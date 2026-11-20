"""Inspect K2 results, export a small inference ZIP, and audit matched controls.

No dataset is opened and no training-state pickle is deserialized. Full result
ZIPs/directories are required for comparison; share ZIPs are for inference/review.
"""
import argparse
from contextlib import AbstractContextManager
import csv
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

import yaml


ARMS = ('baseline', 'film_clean', 'aug', 'film', 'k2')
PROTOCOL = 'weatherseg_k2_source_v1'


class ResultFiles(AbstractContextManager):
    def __init__(self, source):
        self.source = Path(source)
        self.archive = None
        if self.source.is_file():
            self.archive = zipfile.ZipFile(self.source)
            members = self.archive.namelist()
            plans = [n for n in members if PurePosixPath(n).name == 'k2_plan.json']
            if len(plans) != 1 or len(members) != len(set(members)):
                self.archive.close()
                raise ValueError('Choose exactly one K2 result, without duplicate ZIP members')
            self.prefix = plans[0][:-len('k2_plan.json')]
            self.names = {n[len(self.prefix):] for n in members if n.startswith(self.prefix)}
        else:
            plans = ([self.source/'k2_plan.json'] if (self.source/'k2_plan.json').is_file()
                     else list(self.source.rglob('k2_plan.json')))
            if len(plans) != 1:
                raise ValueError('Choose one full K2 workspace or result ZIP')
            self.source = plans[0].parent
            self.names = {p.relative_to(self.source).as_posix() for p in self.source.rglob('*') if p.is_file()}

    def read(self, name):
        if name not in self.names:
            raise ValueError('Missing result artifact: ' + name)
        return self.archive.read(self.prefix + name) if self.archive else (self.source/name).read_bytes()

    def json(self, name):
        return json.loads(self.read(name))

    def __exit__(self, *args):
        if self.archive:
            self.archive.close()


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _summary(files):
    plan = files.json('k2_plan.json')
    if plan.get('protocol') != PROTOCOL or plan.get('arm') not in ARMS:
        raise ValueError('Not a supported source-only K2 result')
    for name, digest in plan['frozen_files'].items():
        if _digest(files.read(name)) != digest:
            raise ValueError('Frozen K2 artifact changed: ' + name)
    cfg = yaml.safe_load(files.read('train.yaml'))
    status = files.json('status.json')
    rows = [json.loads(line) for line in files.read('run/history.jsonl').decode().splitlines() if line.strip()]
    if not rows or [r['epoch'] for r in rows] != list(range(1, len(rows)+1)):
        raise ValueError('Need contiguous committed history from epoch 1')
    if any(not math.isfinite(r['val']['miou']) for r in rows):
        raise ValueError('Nonfinite Cityscapes selection score')
    if status.get('completed_epochs') != len(rows):
        raise ValueError('Status and history disagree; use a committed result ZIP')
    best = max(rows, key=lambda r: r['val']['miou'])
    weights_hash = _digest(files.read('run/best/model.safetensors'))
    config_hash = _digest(files.read('run/best/config.json'))
    checkpoint = files.json('run/best/checkpoint.json')
    if (checkpoint['epoch'] != best['epoch']
            or checkpoint['files']['model.safetensors'] != weights_hash
            or checkpoint['files']['config.json'] != config_hash):
        raise ValueError('Best checkpoint does not match its hashes/selection history')
    complete = (status.get('status') == 'training_complete' and len(rows) == plan['epochs'])
    if complete:
        frozen = files.json('frozen_candidate.json')
        if (frozen['weights_sha256'] != weights_hash or frozen['config_sha256'] != config_hash
                or frozen['plan_sha256'] != _digest(files.read('k2_plan.json'))
                or frozen['arm'] != plan['arm'] or frozen['seed'] != plan['seed']
                or frozen['completed_epochs'] != plan['epochs'] or frozen['target_training'] is not False):
            raise ValueError('Completed candidate seal is invalid')
    weather = files.json('weather_preflight.json')
    model_report = files.json('run/model_report.json') if 'run/model_report.json' in files.names else {}
    resources = ([json.loads(line) for line in files.read('run/resources.jsonl').decode().splitlines() if line.strip()]
                 if 'run/resources.jsonl' in files.names else [])
    return dict(protocol=PROTOCOL, arm=plan['arm'], seed=plan['seed'], complete=complete,
        recipe=plan.get('recipe', 'legacy_v1'), image_size=cfg['data']['image_size'],
        completed_epochs=len(rows), planned_epochs=plan['epochs'], best_epoch=best['epoch'],
        cityscapes_val_miou=best['val']['miou'], cityscapes_val_miou_percent=100*best['val']['miou'],
        last_train_loss=rows[-1]['train_loss'], best_per_class_iou=best['val']['per_class_iou'],
        last_condition_train=rows[-1].get('condition_train'),
        parameters=model_report.get('parameters'), conditions=weather['conditions'],
        fog_enabled=weather['fog_enabled'], optimizer_updates=sum(r['optimizer_updates'] for r in resources),
        train_validation_hours=sum(r['epoch_seconds'] for r in resources)/3600,
        peak_allocated_bytes=max((r.get('max_memory_allocated_bytes') or 0 for r in resources), default=0),
        checkpoint_sha256=weights_hash, config_sha256=config_hash,
        plan_sha256=_digest(files.read('k2_plan.json')),
        curve=[dict(epoch=r['epoch'], train_loss=r['train_loss'], val_miou=r['val']['miou']) for r in rows],
        metric_scope='Cityscapes val500 checkpoint selection; no ACDC result or improvement claim',
        checkpoint_scope='Best inference weights; share ZIP cannot resume training')


def summarize(source):
    with ResultFiles(source) as files:
        return _summary(files)


def export_share(workspace, destination):
    """Leave the full resume archive untouched. Publish only after a committed epoch."""
    workspace, destination = Path(workspace), Path(destination)
    if not (workspace/'run/best/model.safetensors').is_file():
        return None
    if destination.resolve().is_relative_to(workspace.resolve()):
        raise ValueError('Share ZIP must be outside the workspace')
    with ResultFiles(workspace) as files:
        report = _summary(files)
        (workspace/'result_summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        keep = {'run/best/model.safetensors', 'run/best/config.json', 'run/best/checkpoint.json', 'run/history.jsonl',
                'run/resources.jsonl', 'run/model_report.json', 'run/initialization.json',
                'train.yaml', 'k2_plan.json', 'status.json', 'weather_preflight.json',
                'frozen_candidate.json', 'source.zip', 'result_summary.json',
                'data/cityscapes_train.csv', 'data/cityscapes_val.csv'}
        keep.update(p.relative_to(workspace).as_posix() for p in (workspace/'figures').rglob('*')
                    if p.is_file() and p.suffix in {'.png', '.svg', '.json'})
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.partial.zip')
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(keep):
                path = workspace/name
                if path.is_file():
                    archive.write(path, name)
        temporary.replace(destination)
    print(f'SHARE ZIP (inference/review, not resume): {destination} '
          f'({destination.stat().st_size/1024**2:.1f} MiB)', flush=True)
    return report


def _comparison_record(source):
    with ResultFiles(source) as files:
        report = _summary(files)
        if not report['complete']:
            raise ValueError('Compare only completed, frozen runs')
        plan = files.json('k2_plan.json')
        cfg = yaml.safe_load(files.read('train.yaml'))
        from scripts.run_k2 import compose
        expected = compose(plan['arm'], dict(source_train_manifest=cfg['data']['train_manifest'],
            source_monitor_manifest=cfg['data']['val_manifest']), cfg['output_dir'],
            plan['seed'], plan['epochs'], plan['batch_size'], cfg['data']['augment'].get('weather'),
            recipe=plan.get('recipe', 'legacy_v1'))
        # A Kaggle POSIX output path must not become a Windows absolute path
        # during an offline audit. Output locations do not change the recipe.
        expected['output_dir'] = cfg['output_dir']
        if cfg != expected:
            raise ValueError('Run differs from its declared K2 arm/recipe')
        normalized = deepcopy(cfg)
        for key in ('model', 'condition_training', 'output_dir', 'experiment'):
            normalized.pop(key, None)
        normalized['data']['augment'].pop('weather', None)
        for key in ('train_manifest', 'val_manifest'):
            normalized['data'].pop(key)
        manifests = {}
        for split in ('train', 'val'):
            rows = list(csv.DictReader(io.StringIO(files.read(f'data/cityscapes_{split}.csv').decode())))
            if len(rows) != (2975 if split == 'train' else 500):
                raise ValueError('Comparison needs full Cityscapes train2975/val500 manifests')
            if any(r['domain'] != 'cityscapes' or r['split'] != split for r in rows):
                raise ValueError('Comparison contains non-source or wrong-split records')
            manifests[split] = [tuple('/'.join(r[k].replace('\\', '/').split('/')[-3:])
                                      for k in ('image', 'mask')) for r in rows]
        # Changes to launcher, heartbeat, notebook or reporting are allowed;
        # model/trainer/data/loss/pretraining source and all configs must match.
        core = {n: h for n, h in plan['source_hashes'].items() if n.startswith(('src/', 'configs/'))}
        required_core = {'src/training/train.py', 'src/models/registry.py',
                         'src/models/proposed/weatherseg_b0.py',
                         'src/data/augmentations/weather/generator.py',
                         'configs/proposed/k2_film_b0.yaml'}
        if not required_core.issubset(core):
            raise ValueError('Incomplete core source provenance')
        with zipfile.ZipFile(io.BytesIO(files.read('source.zip'))) as snapshot:
            if not core or any(_digest(snapshot.read(n)) != h for n, h in core.items()):
                raise ValueError('Core source snapshot checksum mismatch')
            # Verify each original byte hash first, then ignore platform-only
            # CRLF differences when comparing a Git clone to an uploaded ZIP.
            core = {n: _digest(snapshot.read(n).replace(b'\r\n', b'\n')) for n in core}
        # Compare the actual common backbone+decoder initialization tensors.
        # This is safetensors only; optimizer/RNG pickle is never loaded here.
        from safetensors.torch import load
        initial_bytes = files.read('run/initial/model.safetensors')
        initialization = files.json('run/initialization.json')
        if initialization['weights_sha256'] != _digest(initial_bytes) or initialization['target_data_access'] != 'none':
            raise ValueError('Initial weights provenance mismatch')
        tensor_hash = hashlib.sha256()
        for name, value in sorted(load(initial_bytes).items()):
            if '.film.' not in name and not name.startswith('condition_head.'):
                tensor_hash.update(name.encode())
                tensor_hash.update(str((value.dtype, tuple(value.shape))).encode())
                tensor_hash.update(value.numpy().tobytes())
        return report, normalized, manifests, core, tensor_hash.hexdigest(), cfg['data']['augment'].get('weather')


def compare(sources, require_all=False):
    results = [_comparison_record(source) for source in sources]
    if len(results) < 2:
        raise ValueError('Comparison requires at least two completed controls')
    arms = [r[0]['arm'] for r in results]
    if len(set(arms)) != len(arms):
        raise ValueError('Compare one seed at a time, without duplicate arms')
    if require_all and set(arms) != set(ARMS):
        raise ValueError('Full K2 ablation requires: ' + ', '.join(ARMS))
    for index, label in [(1, 'recipe/seed'), (2, 'source manifests'), (3, 'core code/configs'), (4, 'B0 initialization')]:
        if any(r[index] != results[0][index] for r in results[1:]):
            raise ValueError('Unmatched ' + label + '; preserve these as separate experiments')
    weather = [r[5] for r in results if r[0]['arm'] not in ('baseline', 'film_clean')]
    if weather and any(w != weather[0] for w in weather[1:]):
        raise ValueError('Weather arms must use identical K1 settings/depth paths')
    reports = {r[0]['arm']: r[0] for r in results}
    questions = [('baseline', 'film_clean', 'FiLM on clean images'), ('baseline', 'aug', 'K1'),
                 ('aug', 'film', 'FiLM with K1'), ('film', 'k2', 'condition CE with K1')]
    deltas = [dict(control=a, method=b, question=q,
                   cityscapes_val_delta_pp=100*(reports[b]['cityscapes_val_miou']-reports[a]['cityscapes_val_miou']))
              for a, b, q in questions if a in reports and b in reports]
    return dict(scope='Matched source-selection comparison, not ACDC generalization',
        full_ablation=set(arms) == set(ARMS), seed=results[0][0]['seed'],
        core_source_and_initialization_verified=True,
        note='Launcher/report code may differ. Raw image contents and GPU nondeterminism are not certified by this audit.',
        runs=reports, comparisons=deltas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    report = commands.add_parser('summarize')
    report.add_argument('--input', required=True)
    report.add_argument('--out', required=True)
    share = commands.add_parser('share')
    share.add_argument('--workspace', required=True)
    share.add_argument('--out', required=True)
    comparison = commands.add_parser('compare')
    comparison.add_argument('--inputs', nargs='+', required=True)
    comparison.add_argument('--out', required=True)
    comparison.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    if args.command == 'share':
        if export_share(args.workspace, args.out) is None:
            raise ValueError('No committed best checkpoint yet')
        return
    result = summarize(args.input) if args.command == 'summarize' else compare(args.inputs, args.require_all)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
