import argparse
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from src.data.acdc import recording_group
from src.data.dataset import ManifestSegmentationDataset, evaluation_collate
from src.data.validation import validate_manifests
from src.data.preparation import TARGET_DOMAINS
from src.data.replay import BalancedReplayDataset, replay_selection_rank, validate_replay_loader_options, validate_source_monitor
from src.evaluation.metrics import confusion_matrix, scores
from src.models.segformer import make_model, load_segformer_checkpoint
from src.training.optim import make_optimizer
from src.training.strategies import set_training_mode
from src.training.losses import make_criterion


def read_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _bootstrap_miou(per_image_confusions, samples=1000, seed=42):
    if not per_image_confusions or samples <= 0:
        return [None, None]
    generator = np.random.default_rng(seed)
    count = len(per_image_confusions)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = generator.integers(0, count, size=count)
        aggregate = torch.stack([per_image_confusions[i] for i in chosen]).sum(0)
        values[index] = scores(aggregate)["miou"]
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def evaluate_model(model, loader, device, n, bootstrap_samples=0, seed=42, prediction_callback=None,
                   compute_loss=False):
    """Evaluate once and aggregate metrics by manifest domain.

    Bootstrap intervals resample cached per-image confusion matrices; they do
    not run inference again or select a model using target-domain labels.
    """
    model.eval()
    domain_confusions = {}
    per_image_confusions = {}
    loss_totals = {}
    with torch.inference_mode():
        for batch in loader:
            labels = batch["labels"]
            logits = model(pixel_values=batch["pixel_values"].to(device)).logits
            domains = batch["domain"]
            for index, (target, domain) in enumerate(zip(labels, domains)):
                sample_logits = logits[index:index + 1]
                if "fit_box" in batch:
                    sample_logits = F.interpolate(sample_logits, size=batch["pixel_values"].shape[-2:], mode="bilinear", align_corners=False)
                    top, left, height, width = batch["fit_box"][index]
                    sample_logits = sample_logits[:, :, top:top + height, left:left + width]
                sample_logits = F.interpolate(sample_logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
                if compute_loss:
                    valid_pixels = int((target != 255).sum())
                    total, count = loss_totals.get(domain, (0., 0))
                    # Score the same unpadded, native-size logits as mIoU. An
                    # all-ignore image contributes neither loss nor pixels.
                    if valid_pixels:
                        loss = F.cross_entropy(sample_logits.float(), target.unsqueeze(0).to(device),
                                               ignore_index=255, reduction="sum")
                        if not torch.isfinite(loss):
                            raise RuntimeError("Non-finite validation loss")
                        total += float(loss)
                    loss_totals[domain] = (total, count + valid_pixels)
                prediction = sample_logits.argmax(1)[0].cpu()
                conf = confusion_matrix(prediction, target, n)
                if prediction_callback is not None:
                    prediction_callback(domain, prediction, target, conf)
                domain_confusions.setdefault(domain, torch.zeros((n, n), dtype=torch.int64))
                domain_confusions[domain] += conf
                per_image_confusions.setdefault(domain, []).append(conf)

    result = {"by_domain": {}}
    all_confusions = []
    for domain, conf in sorted(domain_confusions.items()):
        per_image = per_image_confusions[domain]
        domain_result = scores(conf)
        domain_result["n_images"] = len(per_image)
        domain_result["miou_ci95"] = _bootstrap_miou(per_image, bootstrap_samples, seed)
        if compute_loss:
            total, count = loss_totals[domain]
            domain_result.update(cross_entropy=total / count if count else None, loss_valid_pixels=count)
        result["by_domain"][domain] = domain_result
        all_confusions.append(conf)

    overall = scores(torch.stack(all_confusions).sum(0)) if all_confusions else scores(torch.zeros((n, n)))
    result.update(overall)
    result["n_images"] = sum(len(items) for items in per_image_confusions.values())
    if len(result["by_domain"]) == 1:
        result["miou_ci95"] = next(iter(result["by_domain"].values()))["miou_ci95"]
    if compute_loss:
        count = sum(item[1] for item in loss_totals.values())
        total = sum(item[0] for item in loss_totals.values())
        result.update(cross_entropy=total / count if count else None, loss_valid_pixels=count)
    return result


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _save_checkpoint(model, directory, training_state, transactional=False):
    if transactional:
        from src.training.checkpoints import save_transactional
        save_transactional(model, directory, training_state)
        return
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    torch.save(training_state, directory / "training_state.pt")


def validate_train_eval_subset(samples, fit_samples):
    """A same-mode training diagnostic must never use dev, monitor or official val."""
    def identity(row):
        return (str(Path(row['image']).resolve()), str(Path(row['mask']).resolve()), row['domain'], row['split'])
    allowed = {identity(row) for row in fit_samples}
    selected = [identity(row) for row in samples]
    if (not selected or len(set(selected)) != len(selected)
            or any(row['split'] != 'train' for row in samples) or not set(selected) <= allowed):
        raise ValueError('Train-eval manifest must be a unique subset of the actual training fit rows, never dev/monitor/val')


def make_train_eval_loader(dataset, loader_args, seed):
    # A DataLoader without its own generator consumes the global torch RNG
    # even with shuffle=False, changing future dropout when diagnostics change.
    return DataLoader(dataset, shuffle=False, collate_fn=evaluation_collate,
                      generator=torch.Generator().manual_seed(int(seed) + 2020003), **loader_args)


class ReplayEarlyStopping:
    """Plateau detection separate from exact constrained-checkpoint selection.

    min_delta is in mIoU fractions. Before any feasible epoch exists, source
    change (logged as pp) is converted to fractions for this comparison.
    """
    def __init__(self, settings):
        if not isinstance(settings, dict) or set(settings) - {'patience', 'min_delta', 'min_epochs'}:
            raise ValueError('early_stopping accepts patience, min_delta and min_epochs')
        self.patience = settings.get('patience', 3)
        self.min_epochs = settings.get('min_epochs', 3)
        self.min_delta = float(settings.get('min_delta', .001))
        if (isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience < 1
                or isinstance(self.min_epochs, bool) or not isinstance(self.min_epochs, int) or self.min_epochs < 1
                or not math.isfinite(self.min_delta) or self.min_delta < 0):
            raise ValueError('Require positive integer patience/min_epochs and finite nonnegative min_delta')
        self.reference_rank = None
        self.bad_epochs = 0
        self.epoch = 0
        self.stopped = False

    def update(self, rank, epoch):
        if self.stopped or int(epoch) != self.epoch + 1:
            raise ValueError('Early-stopping epochs must continue sequentially from a non-stopped state')
        rank = tuple(rank)
        if len(rank) != 3 or not all(math.isfinite(float(v)) for v in rank):
            raise ValueError('Expected finite replay selection rank')
        if self.reference_rank is None:
            substantial = True
        elif bool(rank[0]) != bool(self.reference_rank[0]):
            substantial = bool(rank[0])
        else:
            difference = rank[1] - self.reference_rank[1]
            if not rank[0]:
                difference /= 100.
            substantial = difference > 0 and difference >= self.min_delta
        if substantial:
            self.reference_rank = rank
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.epoch = int(epoch)
        self.stopped = self.epoch >= self.min_epochs and self.bad_epochs >= self.patience
        return self.stopped

    def state_dict(self):
        return dict(patience=self.patience, min_delta=self.min_delta, min_epochs=self.min_epochs,
                    reference_rank=self.reference_rank, bad_epochs=self.bad_epochs, epoch=self.epoch, stopped=self.stopped)

    def load_state_dict(self, state):
        for key in ('patience', 'min_delta', 'min_epochs'):
            if state[key] != getattr(self, key):
                raise ValueError('Early-stopping settings changed when resuming')
        self.reference_rank = tuple(state['reference_rank']) if state['reference_rank'] is not None else None
        self.bad_epochs, self.epoch, self.stopped = int(state['bad_epochs']), int(state['epoch']), bool(state['stopped'])
        if self.bad_epochs < 0 or self.epoch < 0 or self.stopped:
            raise ValueError('Cannot resume invalid or already early-stopped state')


def validate_target_pilot(train_rows, dev_rows, num_labels):
    """Keep supervised adaptation separate from the source-only experiment."""
    if int(num_labels) != 19:
        raise ValueError("Supervised ACDC requires 19 Cityscapes classes")
    if any(row["domain"] not in TARGET_DOMAINS or row["split"] != "train"
           for row in train_rows + dev_rows):
        raise ValueError("Supervised ACDC pilot requires fit/dev subsets of the official train split")
    train_groups = {recording_group(row["image"]) for row in train_rows}
    dev_groups = {recording_group(row["image"]) for row in dev_rows}
    if train_groups & dev_groups:
        raise ValueError("Supervised ACDC fit/dev video groups overlap")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Resume from a checkpoint directory saved by this trainer")
    parser.add_argument('--allow-supervised-acdc', action='store_true',
                        help='Historical supervised experiment only; incompatible with DG')
    parser.add_argument("--stop-after-epoch", type=int, help="Save and stop after this epoch while preserving the original scheduler horizon")
    args = parser.parse_args()
    if args.stop_after_epoch is not None and args.stop_after_epoch < 1:
        raise ValueError("stop-after-epoch must be positive")

    cfg = read_cfg(args.config)
    from src.experiments.protocol import require_training_permission
    require_training_permission(cfg, args.allow_supervised_acdc)
    transactional = bool(cfg.get('checkpointing', {}).get('transactional', False))
    if transactional and args.resume:
        from src.training.checkpoints import recover_run
        if Path(args.resume).resolve() != (Path(cfg['output_dir']) / 'last').resolve():
            raise ValueError('Transactional training resumes from this run\'s last/ checkpoint')
        if recover_run(cfg['output_dir']) == 0:
            args.resume = None
    protocol = cfg.get("protocol", "source_only")
    if protocol not in {"source_only", "supervised_acdc", "supervised_acdc_replay"}:
        raise ValueError("Unknown training protocol")
    replay = protocol == "supervised_acdc_replay"
    target_training = protocol in {"supervised_acdc", "supervised_acdc_replay"}
    freeze_bn = cfg['train'].get('freeze_batch_norm_stats', False)
    if not isinstance(freeze_bn, bool):
        raise ValueError('train.freeze_batch_norm_stats must be a boolean')
    stopping_settings = cfg.get('early_stopping')
    if stopping_settings is not None and not replay:
        raise ValueError('Constrained early stopping is supported for supervised_acdc_replay only')
    stopping = ReplayEarlyStopping(stopping_settings) if stopping_settings is not None else None
    output_dir = Path(cfg["output_dir"])
    if not args.resume and (output_dir / "history.jsonl").exists():
        raise ValueError("Run already exists; use --resume or a new output_dir")
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.get('training_policy') == 'supervised_comparison_20e_v1':
        if cfg['train']['epochs'] != 20 or cfg.get('early_stopping'):
            raise ValueError('Research comparisons require20 epochs without early stopping')

    manifests = [cfg["data"]["train_manifest"], cfg["data"]["val_manifest"]]
    if replay:
        manifests += [cfg["data"]["source_train_manifest"], cfg["data"]["source_monitor_manifest"]]
    validate_manifests(
        manifests, check_files=True,
        allow_target_train=target_training,
    )
    train = ManifestSegmentationDataset(
        cfg["data"]["train_manifest"],
        cfg["data"]["image_size"],
        training=True,
        augment=cfg["data"].get("augment", {}),
        preprocessing=cfg["data"].get("preprocessing"),
        seed=seed,
        allow_target_train=target_training,
    )
    val = ManifestSegmentationDataset(
        cfg["data"]["val_manifest"],
        cfg["data"]["image_size"],
        native_labels=True,
        preprocessing=cfg["data"].get("preprocessing"),
        allow_target_train=target_training,
    )
    if target_training:
        validate_target_pilot(train.items, val.items, cfg["model"]["num_labels"])
    if train.gamma and train.gamma.probability:
        if any(row["domain"] != "cityscapes" for row in train.items) or any(
            row["domain"] != "cityscapes" or row["split"] != "val" for row in val.items
        ):
            raise ValueError("Gamma runs require Cityscapes train and Cityscapes val for checkpoint selection")
    target_train_count = len(train)
    source_monitor = None
    if cfg['data'].get('target_sampling_weights') and not replay:
        raise ValueError('Target sampling weights require supervised balanced replay')
    if replay:
        validate_replay_loader_options(cfg["train"])
        replay_selection_rank(0.0, 0.0, float(cfg["selection"]["max_source_drop_pp"]))
        source = ManifestSegmentationDataset(cfg["data"]["source_train_manifest"], cfg["data"]["image_size"],
                                             training=True, augment=cfg["data"].get("augment", {}),
                                             preprocessing=cfg["data"].get("preprocessing"), seed=seed)
        source_monitor = ManifestSegmentationDataset(cfg["data"]["source_monitor_manifest"], cfg["data"]["image_size"],
                                                     native_labels=True, preprocessing=cfg["data"].get("preprocessing"))
        validate_source_monitor(source.items, source_monitor.items)
        sampling_path=cfg['data'].get('target_sampling_weights')
        sampling_weights=None
        if sampling_path:
            sampling=json.loads(Path(sampling_path).read_text())
            if hashlib.sha256(Path(cfg['data']['train_manifest']).read_bytes()).hexdigest()!=sampling['fit_manifest_sha256']:
                raise ValueError('Sampling weights belong to a different target-fit manifest')
            sampling_weights=sampling['weights']
        train = BalancedReplayDataset(train, source, seed, target_weights=sampling_weights)
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": int(cfg["train"]["batch_size"]),
        "num_workers": int(cfg["train"]["num_workers"]),
        "worker_init_fn": _seed_worker,
    }
    trainloader = DataLoader(train, shuffle=not replay, generator=generator,
                             drop_last=bool(cfg['train'].get('drop_last', False)), **loader_args)
    criterion = None
    if 'loss' in cfg:
        fit_manifests = [cfg['data']['train_manifest']]
        if replay:
            fit_manifests.append(cfg['data']['source_train_manifest'])
        criterion, provenance = make_criterion(cfg['loss'], fit_manifests)
        criterion = criterion.to(device)
        cfg['loss_provenance'] = provenance
    valloader = DataLoader(val, shuffle=False, collate_fn=evaluation_collate, **loader_args)
    source_loader = DataLoader(source_monitor, shuffle=False, collate_fn=evaluation_collate, **loader_args) if replay else None
    train_eval_loader = None
    train_eval_manifest = cfg.get('monitoring', {}).get('train_eval_manifest')
    if train_eval_manifest:
        validate_manifests([train_eval_manifest], check_files=True, allow_target_train=target_training)
        train_eval = ManifestSegmentationDataset(train_eval_manifest, cfg['data']['image_size'], native_labels=True,
                                                preprocessing=cfg['data'].get('preprocessing'), allow_target_train=target_training)
        fit_rows = train.target.items + train.source.items if replay else train.items
        validate_train_eval_subset(train_eval.items, fit_rows)
        train_eval_loader = make_train_eval_loader(train_eval, loader_args, seed)

    if target_training and not args.resume:
        initial = Path(cfg["model"]["checkpoint"])
        with (initial / "model.safetensors").open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != cfg["model"]["checkpoint_sha256"]:
                raise ValueError("Initial source checkpoint hash mismatch")
        if cfg['model'].get('name'):
            from src.models.registry import load_checkpoint
            model = load_checkpoint(initial, cfg['model']['name'])
        else:
            model = load_segformer_checkpoint(initial, local_files_only=True, variant=cfg['model'].get('variant'))
        if model.config.num_labels != 19 or model.config.semantic_loss_ignore_index != 255:
            raise ValueError("Initial checkpoint must use 19 classes and ignore index 255")
    else:
        if cfg['model'].get('name'):
            from src.models.registry import build_model, load_checkpoint
            model = load_checkpoint(args.resume, cfg['model']['name']) if args.resume else build_model(cfg)
        else:
            model = (
                load_segformer_checkpoint(args.resume,variant=cfg['model'].get('variant'))
                if args.resume else make_model(cfg)
            )
    model = model.to(device)
    if args.resume:
        # Detach CPU safetensors mappings before overwriting last/ on Windows.
        model.load_state_dict({key: value.clone() for key, value in model.state_dict().items()}, assign=True)
    optimizer = make_optimizer(model, cfg["train"])

    max_steps = cfg["train"].get("max_steps")
    batches_per_epoch = min(len(trainloader), int(max_steps)) if max_steps is not None else len(trainloader)
    accumulation = max(1, int(cfg["train"].get("gradient_accumulation_steps", 1)))
    updates_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_updates = max(1, updates_per_epoch * int(cfg["train"]["epochs"]))
    warmup_updates = min(int(cfg["train"].get("warmup_steps", 0)), total_updates - 1)
    power = float(cfg["train"].get("poly_power", 1.0))

    def lr_multiplier(step):
        if warmup_updates and step < warmup_updates:
            return float(step + 1) / warmup_updates
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return max(0.0, 1.0 - progress) ** power

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    amp_enabled = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    amp_name = cfg["train"].get("amp_dtype", "float16")
    if amp_name not in {"float16", "bfloat16"}:
        raise ValueError("amp_dtype must be float16 or bfloat16")
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg['model'].get('name') and not args.resume:
        from src.models.registry import parameter_report
        model.save_pretrained(output_dir / 'initial')
        (output_dir / 'model_report.json').write_text(json.dumps(dict(
            name=cfg['model']['name'], initialization=cfg['model'].get('initialization', 'random'),
            parameters=parameter_report(model), torch=torch.__version__,
            implementation=f'{type(model).__module__}.{type(model).__name__}'), indent=2), encoding='utf-8')
    start_epoch = 0
    global_step = 0
    best_miou = -math.inf
    best_epoch = None
    best_rank = None
    source_reference = None
    if args.resume:
        state_path = Path(args.resume) / "training_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"Resume state not found: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        comparable = lambda config: {key: value for key, value in config.items() if key != "output_dir"}
        if "config" not in state or comparable(state["config"]) != comparable(cfg):
            raise ValueError("Resume requires the original config and scheduler horizon; legacy state has no verified config")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        best_miou = float(state["best_miou"])
        best_epoch = state.get('best_epoch')
        if replay:
            best_rank = tuple(state["best_selection_rank"])
            source_reference = float(state["source_monitor_reference_miou"])
        if stopping:
            if 'early_stopping_state' not in state:
                raise ValueError('Resume checkpoint lacks early-stopping state')
            stopping.load_state_dict(state['early_stopping_state'])
            if stopping.epoch != start_epoch:
                raise ValueError('Early-stopping state epoch disagrees with checkpoint')
        rng = state.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            generator.set_state(rng["loader"])
            if device.type == "cuda" and "cuda" in rng:
                torch.cuda.set_rng_state_all(rng["cuda"])

    if args.stop_after_epoch is not None and start_epoch >= args.stop_after_epoch:
        raise ValueError("Checkpoint already reached stop-after-epoch; choose a later stop or omit the option")
    # Optional diagnostics use the existing inference pass. Keeping the default
    # off preserves historical logs and model-selection behavior.
    loss_monitoring = cfg.get("monitoring", {}).get("validation_loss", False)
    if not isinstance(loss_monitoring, bool):
        raise ValueError("monitoring.validation_loss must be a boolean")
    if replay and not args.resume:
        initial_target = evaluate_model(model, valloader, device, 19, compute_loss=loss_monitoring)
        initial_source = evaluate_model(model, source_loader, device, 19, compute_loss=loss_monitoring)
        source_reference = initial_source["miou"]
        if not math.isfinite(source_reference):
            raise ValueError("Source monitoring baseline must be finite")
        monitor_note = ("Source monitor is withheld from the matched Cityscapes source fit and replay"
                        if cfg.get('training_policy') == 'supervised_comparison_20e_v1' else
                        "Source monitor is withheld from replay, but was seen during the original Cityscapes pretraining")
        (output_dir / "initial_validation.json").write_text(json.dumps(
            dict(target=initial_target, source=initial_source,
                 note=monitor_note), indent=2) + "\n")

    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    if float(cfg["train"].get("head_lr_multiplier", 1.)) != 1.:
        groups = [dict(name=g["name"], base_lr=g["initial_lr"], weight_decay=g["weight_decay"],
                       parameters=sum(p.numel() for p in g["params"])) for g in optimizer.param_groups]
        (output_dir / "optimizer_groups.json").write_text(json.dumps(groups, indent=2)+"\n")
    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        train.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        epoch_started = time.perf_counter()
        step_started = global_step
        set_training_mode(model, freeze_bn)
        losses = []
        target_losses, source_losses = [], []
        crop_exposure = dict(selected=[0] * 19, met=[0] * 19, pixels=[0] * 19)
        optimizer.zero_grad(set_to_none=True)
        epoch_batches = min(len(trainloader), int(max_steps)) if max_steps is not None else len(trainloader)
        for step, batch in enumerate(trainloader):
            if step >= epoch_batches:
                break
            if 'crop_class_id' in batch:
                for chosen, met, pixels in zip(batch['crop_class_id'].tolist(),
                                               batch['crop_constraint_met'].tolist(), batch['crop_class_pixels'].tolist()):
                    if chosen >= 0:
                        crop_exposure['selected'][chosen] += 1
                        crop_exposure['met'][chosen] += int(met)
                        crop_exposure['pixels'][chosen] += pixels
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, epoch_batches - group_start)
            context = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
            with context:
                if criterion is None:
                    # Preserve historical models' exact built-in loss path.
                    result = model(pixel_values=batch['pixel_values'].to(device), labels=batch['labels'].to(device))
                    loss = result.loss
                else:
                    result = model(pixel_values=batch['pixel_values'].to(device))
                    loss = criterion(result.logits, batch['labels'].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; check labels, all-ignore crops and AMP before resuming")
            losses.append(float(loss.detach()))
            if replay:
                (source_losses if batch["domain"][0] == "cityscapes" else target_losses).append(float(loss.detach()))
            scaler.scale(loss / group_size).backward()
            is_update = (step + 1) % accumulation == 0 or step + 1 == epoch_batches
            if is_update:
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    global_step += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.perf_counter() - epoch_started
        metrics = evaluate_model(model, valloader, device, cfg["model"]["num_labels"], compute_loss=loss_monitoring)
        source_metrics = evaluate_model(model, source_loader, device, 19, compute_loss=loss_monitoring) if replay else None
        train_eval_metrics = evaluate_model(model, train_eval_loader, device, cfg['model']['num_labels'],
                                            compute_loss=loss_monitoring) if train_eval_loader is not None else None
        if device.type == "cuda":
            torch.cuda.synchronize()
        validation_seconds = time.perf_counter() - epoch_started - train_seconds
        record = {
            "protocol": protocol,
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val": metrics,
            "seed": seed,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "amp_dtype": amp_name if amp_enabled else "float32",
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        current_rank = None
        if criterion is not None:
            record['training_loss_spec'] = cfg['loss']
        if sum(crop_exposure['selected']):
            record['crop_exposure'] = crop_exposure
        if train_eval_metrics is not None:
            record['train_eval'] = train_eval_metrics
        if float(cfg["train"].get("head_lr_multiplier", 1.)) != 1.:
            record["learning_rates"] = {g["name"]: g["lr"] for g in optimizer.param_groups}
        if replay:
            source_delta = 100 * (source_metrics["miou"] - source_reference)
            current_rank = replay_selection_rank(metrics["miou"], source_delta,
                                                  float(cfg["selection"]["max_source_drop_pp"]))
            record.update(source_val=source_metrics, source_delta_pp=source_delta,
                          source_monitor_gate_pass=bool(current_rank[0]),
                          target_train_loss=float(np.mean(target_losses)), source_train_loss=float(np.mean(source_losses)))
        early_stop = stopping.update(current_rank, epoch + 1) if stopping else False
        if stopping:
            record['early_stopping'] = stopping.state_dict()
        print(json.dumps(record, allow_nan=False))
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps(record, allow_nan=False) + "\n")

        val_miou = metrics["miou"]
        is_best = (best_rank is None or current_rank > best_rank) if replay else math.isfinite(val_miou) and val_miou > best_miou
        if is_best:
            best_miou = val_miou
            best_rank = current_rank
            best_epoch = epoch + 1
        state = {
            "config": cfg,
            "epoch": epoch + 1,
            "global_step": global_step,
            "best_miou": best_miou,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "loader": generator.get_state(),
            },
        }
        if replay:
            state.update(best_selection_rank=best_rank, source_monitor_reference_miou=source_reference)
        if stopping:
            state['early_stopping_state'] = stopping.state_dict()
        if transactional:
            state['best_epoch'] = best_epoch
        _save_checkpoint(model, output_dir / "last", state, transactional)
        if is_best:
            _save_checkpoint(model, output_dir / "best", state, transactional)
        resources = {
            "epoch": epoch + 1, "device": str(device),
            "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "amp_enabled": amp_enabled,
            "train_seconds": train_seconds, "validation_seconds": validation_seconds,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "batches": len(losses), "optimizer_updates": global_step - step_started,
            "train_images": len(train), "val_images": len(val),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        }
        if replay:
            resources.update(target_train_images=target_train_count, source_pool_images=len(train.source),
                             source_replay_images_per_epoch=target_train_count, source_monitor_images=len(source_monitor))
        with (output_dir / "resources.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(resources, allow_nan=False) + "\n")
        if stopping:
            end_status = 'stopped' if early_stop else 'horizon_reached' if epoch + 1 == int(cfg['train']['epochs']) else 'running'
            stop_report = dict(stopping.state_dict(), status=end_status, planned_epochs=int(cfg['train']['epochs']),
                               reason='replay_selection_plateau' if early_stop else 'horizon_reached' if end_status == 'horizon_reached' else None)
            (output_dir/'early_stopping.json').write_text(json.dumps(stop_report, indent=2)+'\n')
        if early_stop:
            print(f'Early stopping after epoch {epoch + 1}; fixed scheduler horizon remains {cfg["train"]["epochs"]}')
            break
        if args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch:
            print(f"Saved epoch {epoch + 1} of {cfg['train']['epochs']}; resume with the original config")
            break

    if not (output_dir / "best").is_dir():
        raise RuntimeError("Training completed without a finite validation mIoU checkpoint")
    print(f"Best checkpoint: {output_dir / 'best'}")


if __name__ == "__main__":
    main()
