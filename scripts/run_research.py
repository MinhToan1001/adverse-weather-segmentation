"""Current entry point: swap --model; keep data, training and evaluation shared."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import yaml

from src.data.class_weights import fit_class_weights
from src.experiments.config import compose_experiment
from src.models.registry import MODEL_NAMES, build_model, parameter_report
from src.training.losses import LOSS_NAMES
from src.utils.artifacts import sha256_file, write_json, records, load_training_state as load_state


REPO = Path(__file__).resolve().parents[1]


def fingerprints():
    return {p.relative_to(REPO).as_posix(): sha256_file(p) for directory in ('src', 'scripts', 'configs')
            for p in sorted((REPO / directory).rglob('*')) if p.is_file() and p.suffix in ('.py', '.yaml', '.json')}


def snapshot_source(destination, hashes=None):
    hashes = hashes or fingerprints()
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(REPO / name, name)
        for name in ('requirements.txt', 'THIRD_PARTY_NOTICES.md', 'third_party/transformers/LICENSE'):
            archive.write(REPO / name, name)


def committed_state(run_dir, expected_epoch=None):
    """Only a checkpoint with matching config and log commits is complete."""
    run_dir = Path(run_dir)
    state = load_state(run_dir / 'last/training_state.pt')
    config = json.loads((run_dir / 'config.json').read_text())
    epoch = state['epoch']
    usage, history = records(run_dir / 'resources.jsonl'), records(run_dir / 'history.jsonl')
    if ((expected_epoch is not None and epoch != expected_epoch)
            or [r['epoch'] for r in usage] != list(range(1, epoch + 1))
            or [r['epoch'] for r in history] != list(range(1, epoch + 1))
            or state['config'] != config
            or state['global_step'] != sum(r['optimizer_updates'] for r in usage)):
        raise ValueError('Checkpoint/config/logs do not form a committed run')
    return config


def fits_session(now, deadline, recent_epoch_seconds):
    estimate = max(recent_epoch_seconds[-3:] or [2400])
    return now + estimate * 1.5 + 900 <= deadline


def prepare(args):
    root = Path(args.output).resolve()
    hashes = fingerprints()
    if (root / 'experiment.json').exists():
        saved = json.loads((root / 'experiment.json').read_text())
        if (saved['source_hashes'] != hashes or saved['model'] != args.model
                or saved['stage'] != args.stage or saved['loss'] != args.loss
                or saved['class_crop'] != args.class_crop or saved['seed'] != args.seed
                or saved['data_config_sha256'] != sha256_file(args.data)
                or saved['initialization'] != args.initialization):
            raise ValueError('Resume with the exact experiment code/model/loss/seed; keep each run separate')
        if saved['config_sha256'] != sha256_file(root / 'train.yaml'):
            raise ValueError('Frozen training config changed')
        for path, checksum in saved['fit_and_dev_hashes'].items():
            if sha256_file(path) != checksum:
                raise ValueError('Experiment data manifests changed')
        for path, checksum in saved.get('auxiliary_hashes', {}).items():
            if sha256_file(path) != checksum:
                raise ValueError('Frozen loss metadata changed')
        cfg = yaml.safe_load((root / 'train.yaml').read_text())
        if cfg['model'].get('checkpoint_sha256') and sha256_file(
                Path(cfg['model']['checkpoint']) / 'model.safetensors') != cfg['model']['checkpoint_sha256']:
            raise ValueError('Frozen source-stage dependency changed')
        return cfg
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh experiment directory')
    data = json.loads(Path(args.data).read_text())['paths']
    if args.stage == 'adapt':
        if not args.checkpoint:
            raise ValueError('Provide --checkpoint from this model\'s source-stage best/')
        source = Path(args.checkpoint).resolve().parent
        config = committed_state(source, 20)
        if config['protocol'] != 'source_only' or config['model'].get('name') != args.model:
            raise ValueError('Matched comparison requires this model\'s complete Cityscapes source run')
        if config['train']['epochs'] != 20 or config.get('training_policy') != 'supervised_comparison_20e_v1':
            raise ValueError('Complete the matched20epoch source stage before adaptation')
        if config['loss'] != dict(name='ce') or config['data']['augment'].get('class_aware_crop'):
            raise ValueError('Adaptation ablations start from the common source CE control')
        if (sha256_file(config['data']['train_manifest']) != sha256_file(data['source_train_manifest'])
                or sha256_file(config['data']['val_manifest']) != sha256_file(data['source_monitor_manifest'])):
            raise ValueError('Source run must use the same Cityscapes fit/monitor split')
        from safetensors import safe_open
        selected = source / 'best/model.safetensors'
        if Path(args.checkpoint).resolve() != selected.parent.resolve():
            raise ValueError('Use the source-stage selected best/ checkpoint')
        with safe_open(str(selected), framework='pt'):
            pass
    root.mkdir(parents=True)
    cfg = compose_experiment(args.model, args.stage, data, root / 'run', args.loss,
                             args.checkpoint, args.initialization, args.seed, args.class_crop)
    if args.loss == 'weighted_ce':
        manifests = [cfg['data']['train_manifest']]
        if args.stage == 'adapt':
            manifests.append(cfg['data']['source_train_manifest'])
        artifact = root / 'fit_class_weights.json'
        fit_class_weights(manifests, artifact)
        cfg['loss']['class_weights_file'] = str(artifact)
    path = root / 'train.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    write_json(root / 'experiment.json', dict(model=args.model, stage=args.stage, loss=args.loss,
        class_crop=args.class_crop, seed=args.seed, initialization=args.initialization,
        data_config_sha256=sha256_file(args.data), config_sha256=sha256_file(path), source_hashes=hashes,
        auxiliary_hashes={cfg['loss']['class_weights_file']: sha256_file(cfg['loss']['class_weights_file'])}
            if 'class_weights_file' in cfg['loss'] else {},
        fit_and_dev_hashes={p: sha256_file(p) for key, p in cfg['data'].items() if key.endswith('_manifest')},
        selection='source-monitor for source stage; constrained target-dev/source-monitor for adaptation; official-val later'))
    snapshot_source(root / 'source.zip', hashes)
    return cfg


def run(args, on_epoch=None, deadline=None):
    started = time.monotonic()
    cfg = prepare(args)
    root = Path(args.output).resolve()
    if args.prepare_only:
        print(root / 'train.yaml')
        return
    import torch
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise ValueError('Enable GPU; --allow-cpu is for small plumbing probes only')
    if args.inspect:
        model = build_model(dict(model=dict(name=args.model, initialization='random')))
        print(json.dumps(parameter_report(model)))
        return
    stop_at = deadline if deadline is not None else started + args.session_hours * 3600
    run_dir = Path(cfg['output_dir'])
    if cfg.get('checkpointing', {}).get('transactional'):
        from src.training.checkpoints import recover_run
        recover_run(run_dir)
    try:
        for epoch in range(1, 21):
            last = run_dir / 'last/training_state.pt'
            state = load_state(last) if last.exists() else {}
            saved = state.get('epoch', 0)
            if saved >= epoch:
                continue
            # Roll back uncommitted log rows after an interrupted epoch save.
            for name in ('history.jsonl', 'resources.jsonl'):
                path = run_dir / name
                rows = records(path)
                if rows and rows[-1]['epoch'] > saved:
                    shutil.copy2(path, path.with_suffix('.interrupted.jsonl'))
                    path.write_text(''.join(json.dumps(r) + '\n' for r in rows if r['epoch'] <= saved))
            usage = records(run_dir / 'resources.jsonl')
            if not fits_session(time.monotonic(), stop_at, [r['epoch_seconds'] for r in usage]):
                write_json(root / 'status.json', dict(status='session_checkpoint_ready', epoch=saved, planned_epochs=20))
                print('Session window reached; resume the same experiment from last/.', flush=True)
                return
            command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(root / 'train.yaml'),
                       '--stop-after-epoch', str(epoch)]
            if saved:
                command += ['--resume', str(run_dir / 'last')]
            subprocess.run(command, cwd=REPO, check=True)
            committed_state(run_dir, epoch)
            if on_epoch is not None:
                on_epoch(epoch)
        committed_state(run_dir, 20)
        write_json(root / 'status.json', dict(status='training_complete', planned_epochs=20,
                                             final_validation='pending_frozen_candidate_evaluation'))
    except Exception as error:
        write_json(root / 'status.json', dict(status='training_failed', error=str(error), planned_epochs=20,
                                             action='Preserve outputs; audit failure before resume'))
        raise
    finally:
        write_json(root / 'invocation.json', dict(elapsed_seconds=time.monotonic() - started,
                                                 session_hours=args.session_hours, total_hours_cap=None))
        print('Preserve experiment + shared data directory before ending the session:', root, flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODEL_NAMES, default='segformer_b2')
    p.add_argument('--stage', choices=('source', 'adapt'), default='source')
    p.add_argument('--loss', choices=LOSS_NAMES, default='ce')
    p.add_argument('--data', required=True, help='Shared data.json from prepare_research_data')
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--initialization', choices=('imagenet_encoder', 'random'), default='imagenet_encoder')
    p.add_argument('--class-crop', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--session-hours', type=float, default=10., help='Per-invocation window, not a total experiment budget')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--allow-cpu', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if not 0 < args.session_hours <= 10:
        raise ValueError('Use a session window in(0,10] hours, leaving room before Kaggle\'s12hour cutoff')
    run(args)
