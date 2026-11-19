"""Cityscapes-only DG arms for B2/DeepLab: train2975, select val500, no target input."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from scripts.run_research import committed_state, fingerprints, fits_session, snapshot_source
from src.data.preparation import create_manifest, write_manifest, convert_label_ids
from src.data.validation import validate_manifests
from src.experiments.dg import compose_dg
from src.models.registry import MODEL_NAMES
from src.training.dg import ARMS
from src.experiments.packaging import export_workspace
from src.experiments.protocol import DG_PROTOCOL, require_training_permission
from src.utils.artifacts import sha256_file, records, write_json


def prepare_cityscapes(city, output):
    """The interface deliberately has no ACDC root or target split argument."""
    city, output = Path(city).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    for split, count in [('train', 2975), ('val', 500)]:
        masks = city
        available = list((city / 'gtFine' / split).rglob('*_gtFine_labelTrainIds.png'))
        if len(available) != count:
            raw = sorted((city / 'gtFine' / split).rglob('*_gtFine_labelIds.png'))
            if len(raw) != count:
                raise ValueError(f'Cityscapes {split}: expected {count} labelIds/trainIds masks')
            masks = output / 'converted'
            def convert(path):
                relative = path.relative_to(city)
                destination = masks / relative.parent / relative.name.replace('_labelIds.png', '_labelTrainIds.png')
                convert_label_ids(path, destination)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(convert, raw))
        destination = output / f'cityscapes_{split}.csv'
        write_manifest(create_manifest(city, 'cityscapes', split, mask_root=masks), destination)
        paths['source_train_manifest' if split == 'train' else 'source_monitor_manifest'] = str(destination)
    validate_manifests(list(paths.values()), check_files=True)
    return paths


def prepare(city, root, seed=42, model='segformer_b2', arm='baseline', pair_probability=.5, consistency_weight=.1):
    root, city = Path(root).resolve(), Path(city).resolve()
    # Validate run choices before converting thousands of source masks.
    compose_dg(model,arm,dict(source_train_manifest='',source_monitor_manifest=''),
               root/'run',seed,pair_probability,consistency_weight)
    plan_path = root / 'dg_plan.json'
    identity = dict(experiment_protocol=DG_PROTOCOL, city_root=str(city), seed=seed,
                    source_hashes=fingerprints(), target_data_access='none', epochs=20,
                    model=model, arm=arm, method='source_ce_baseline' if arm == 'baseline' else 'source_paired_style_v1',
                    pair_probability=0. if arm == 'baseline' else pair_probability,
                    consistency_weight=consistency_weight if arm == 'consistency' else 0.)
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if any(plan.get(k) != v for k, v in identity.items()):
            raise ValueError('Resume with the original DG source/data mount/seed')
        for name, checksum in plan['frozen_files'].items():
            if sha256_file(root / name) != checksum:
                raise ValueError('Frozen DG config or Cityscapes manifests changed')
        if not (root / 'source.zip').is_file():
            # Kaggle can expand nested ZIPs. Current code already matched the
            # per-file source hashes above, so rebuilding preserves code identity.
            snapshot_source(root / 'source.zip', identity['source_hashes'])
        cfg = yaml.safe_load((root / 'train.yaml').read_text())
        require_training_permission(cfg)
        return cfg
    if root.exists() and any(root.iterdir()):
        raise ValueError('Preserve existing outputs; choose a fresh DG workspace')
    root.mkdir(parents=True, exist_ok=True)
    paths = prepare_cityscapes(city, root / 'data')
    cfg = compose_dg(model, arm, paths, root / 'run', seed, pair_probability, consistency_weight)
    require_training_permission(cfg)
    validate_manifests(list(paths.values()), check_files=True)
    (root / 'train.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    snapshot_source(root / 'source.zip', identity['source_hashes'])
    # Source identity uses per-file hashes, not ZIP container timestamps.
    files = ['train.yaml', 'data/cityscapes_train.csv', 'data/cityscapes_val.csv']
    write_json(plan_path, dict(identity, frozen_files={p: sha256_file(root / p) for p in files},
        selection='Cityscapes official val500 mIoU only; ACDC evaluation after checkpoint freeze',
        target_evaluation_limitation='ACDC development scores have been viewed historically; not a fresh independent test'))
    return cfg


def run(city, root, archive, seed=42, session_hours=10., prepare_only=False,
        model='segformer_b2', arm='baseline', pair_probability=.5, consistency_weight=.1):
    if not 0 < session_hours <= 10:
        raise ValueError('Session window must be in (0,10] hours')
    root, archive = Path(root).resolve(), Path(archive).resolve()
    if archive.is_relative_to(root):
        raise ValueError('Save result ZIP outside the DG workspace')
    started = time.monotonic()
    cfg = prepare(city, root, seed, model, arm, pair_probability, consistency_weight)
    if prepare_only:
        print('DG config prepared; no training or target data accessed:', root / 'train.yaml')
        export_workspace(root, archive)
        return
    import torch
    if not torch.cuda.is_available():
        raise ValueError('Enable GPU for full DG training; --prepare-only is available on CPU')
    from src.training.checkpoints import recover_run
    run_dir = Path(cfg['output_dir'])
    recover_run(run_dir)
    status = 'running'
    try:
        for epoch in range(1, 21):
            metadata = run_dir / 'last/checkpoint.json'
            saved = json.loads(metadata.read_text())['epoch'] if metadata.is_file() else 0
            if saved >= epoch:
                committed_state(run_dir, saved)
                continue
            usage = records(run_dir / 'resources.jsonl')
            if not fits_session(time.monotonic(), started + session_hours * 3600,
                                [r['epoch_seconds'] for r in usage]):
                status = 'session_checkpoint_ready'
                break
            command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(root / 'train.yaml'),
                       '--stop-after-epoch', str(epoch)]
            if saved:
                command += ['--resume', str(run_dir / 'last')]
            subprocess.run(command, check=True)
            committed_state(run_dir, epoch)
            from scripts.plot_research_run import plot
            plot(run_dir)
            write_json(root / 'status.json', dict(protocol=DG_PROTOCOL, status='running', completed_epochs=epoch))
            export_workspace(root, archive)
            print('DG checkpoint ZIP ready:', archive, flush=True)
        else:
            committed_state(run_dir, 20)
            status = 'training_complete'
            write_json(root / 'frozen_candidate.json', dict(protocol=DG_PROTOCOL, selected_by='Cityscapes val500 only',
                checkpoint='run/best', weights_sha256=sha256_file(run_dir / 'best/model.safetensors'),
                model_config_sha256=sha256_file(run_dir / 'best/config.json'),
                training_config_sha256=sha256_file(run_dir / 'config.json'),
                initialization=json.loads((run_dir / 'initialization.json').read_text()),
                model=model, arm=arm, completed_epochs=20,
                plan_sha256=sha256_file(root / 'dg_plan.json'), target_training=False,
                target_evaluation='not run by this launcher'))
    except Exception:
        status = 'failed_preserve_outputs'
        raise
    finally:
        write_json(root / 'status.json', dict(protocol=DG_PROTOCOL, status=status, elapsed_seconds=time.monotonic()-started))
        export_workspace(root, archive)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--session-hours', type=float, default=10.)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--model', choices=MODEL_NAMES, default='segformer_b2')
    parser.add_argument('--arm', choices=ARMS, default='baseline')
    parser.add_argument('--pair-probability', type=float, default=.5)
    parser.add_argument('--consistency-weight', type=float, default=.1)
    a = parser.parse_args()
    run(a.city, a.root, a.archive, a.seed, a.session_hours, a.prepare_only,
        a.model, a.arm, a.pair_probability, a.consistency_weight)
