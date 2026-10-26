import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from src.data.dataset import ManifestSegmentationDataset
from src.evaluation.metrics import confusion_matrix, scores
from src.models.segformer import make_model


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


def evaluate_model(model, loader, device, n, bootstrap_samples=0, seed=42):
    """Evaluate once and aggregate metrics by manifest domain.

    Bootstrap intervals resample cached per-image confusion matrices; they do
    not run inference again or select a model using target-domain labels.
    """
    model.eval()
    domain_confusions = {}
    per_image_confusions = {}
    with torch.inference_mode():
        for batch in loader:
            labels = batch["labels"]
            logits = model(pixel_values=batch["pixel_values"].to(device)).logits
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predictions = logits.argmax(1).cpu()
            domains = batch["domain"]
            for prediction, target, domain in zip(predictions, labels, domains):
                conf = confusion_matrix(prediction, target, n)
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
        result["by_domain"][domain] = domain_result
        all_confusions.append(conf)

    overall = scores(torch.stack(all_confusions).sum(0)) if all_confusions else scores(torch.zeros((n, n)))
    result.update(overall)
    result["n_images"] = sum(len(items) for items in per_image_confusions.values())
    if len(result["by_domain"]) == 1:
        result["miou_ci95"] = next(iter(result["by_domain"].values()))["miou_ci95"]
    return result


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _save_checkpoint(model, directory, training_state):
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    torch.save(training_state, directory / "training_state.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Resume from a checkpoint directory saved by this trainer")
    args = parser.parse_args()

    cfg = read_cfg(args.config)
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = ManifestSegmentationDataset(
        cfg["data"]["train_manifest"],
        cfg["data"]["image_size"],
        training=True,
        augment=cfg["data"].get("augment", {}),
    )
    val = ManifestSegmentationDataset(
        cfg["data"]["val_manifest"],
        cfg["data"]["image_size"],
    )
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": int(cfg["train"]["batch_size"]),
        "num_workers": int(cfg["train"]["num_workers"]),
        "worker_init_fn": _seed_worker,
    }
    trainloader = DataLoader(train, shuffle=True, generator=generator, **loader_args)
    valloader = DataLoader(val, shuffle=False, **loader_args)

    model = (
        SegformerForSemanticSegmentation.from_pretrained(args.resume)
        if args.resume
        else make_model(cfg)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.01)),
    )

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
    amp_dtype = torch.bfloat16 if amp_name == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    global_step = 0
    best_miou = -math.inf
    if args.resume:
        state_path = Path(args.resume) / "training_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"Resume state not found: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        best_miou = float(state["best_miou"])
        rng = state.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            generator.set_state(rng["loader"])
            if device.type == "cuda" and "cuda" in rng:
                torch.cuda.set_rng_state_all(rng["cuda"])

    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        epoch_batches = min(len(trainloader), int(max_steps)) if max_steps is not None else len(trainloader)
        for step, batch in enumerate(trainloader):
            if step >= epoch_batches:
                break
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, epoch_batches - group_start)
            context = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
            with context:
                result = model(
                    pixel_values=batch["pixel_values"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = result.loss
            losses.append(float(loss.detach()))
            scaler.scale(loss / group_size).backward()
            is_update = (step + 1) % accumulation == 0 or step + 1 == epoch_batches
            if is_update:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        metrics = evaluate_model(model, valloader, device, cfg["model"]["num_labels"])
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val": metrics,
            "seed": seed,
            "device": str(device),
            "amp_enabled": amp_enabled,
            "amp_dtype": amp_name if amp_enabled else "float32",
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(record, allow_nan=False))
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps(record, allow_nan=False) + "\n")

        val_miou = metrics["miou"]
        is_best = math.isfinite(val_miou) and val_miou > best_miou
        if is_best:
            best_miou = val_miou
        state = {
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
        _save_checkpoint(model, output_dir / "last", state)
        if is_best:
            _save_checkpoint(model, output_dir / "best", state)

    if not (output_dir / "best").is_dir():
        raise RuntimeError("Training completed without a finite validation mIoU checkpoint")
    print(f"Best checkpoint: {output_dir / 'best'}")


if __name__ == "__main__":
    main()
