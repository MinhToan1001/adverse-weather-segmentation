"""Active B2/optional DeepLab jobs, per-epoch export and resumable queue."""
import argparse
import json
from pathlib import Path
import time
from statistics import median
from types import SimpleNamespace

from scripts.run_research import run as run_one, fingerprints, snapshot_source
from scripts.plot_research_run import plot
from src.experiments.packaging import export_workspace
from src.models.registry import MODEL_NAMES
from src.utils.artifacts import sha256_file, write_json, records


SCHEMA = 'active_models_fresh_ce_20e_v2'
SELECTED_SCHEMA = 'selected_models_fresh_ce_20e_v1'


def jobs(models=None):
    models = list(MODEL_NAMES if models is None else models)
    if not models or len(set(models)) != len(models) or any(name not in MODEL_NAMES for name in models):
        raise ValueError('Choose unique supported model names')
    return [dict(id=name + '_' + stage, model=name, stage=stage, epochs=20, loss='ce')
            for name in models for stage in ('source', 'adapt')]


def prepare_campaign(root, data, seed=42, models=None):
    root, data = Path(root).resolve(), Path(data).resolve()
    plan_file = root / 'campaign.json'
    current = dict(schema=SCHEMA if models is None else SELECTED_SCHEMA, seed=seed, jobs=jobs(models), source_hashes=fingerprints(),
                   data_config_sha256=sha256_file(data), initialization='imagenet_encoder',
                   old_project_checkpoints='excluded', class_crop=False, epochs_per_job=20)
    if plan_file.exists():
        if json.loads(plan_file.read_text()) != current:
            raise ValueError('Resume the original campaign code/data/seed; old experiments cannot be mixed into this plan')
    else:
        if (root / 'jobs').exists() and any((root / 'jobs').iterdir()):
            raise ValueError('A fresh campaign must have an empty jobs/ directory')
        root.mkdir(parents=True, exist_ok=True)
        snapshot_source(root / 'source.zip', current['source_hashes'])
        write_json(plan_file, current)
    return current


def progress(root, plan):
    rows = []
    for job in plan['jobs']:
        run = Path(root) / 'jobs' / job['id'] / 'run'
        metadata = run / 'last/checkpoint.json'
        epoch = json.loads(metadata.read_text())['epoch'] if metadata.is_file() else 0
        history = records(run / 'history.jsonl')
        usage = records(run / 'resources.jsonl')
        committed = (len(history) >= epoch and len(usage) >= epoch
                     and [r['epoch'] for r in history[:epoch]] == list(range(1, epoch + 1))
                     and [r['epoch'] for r in usage[:epoch]] == list(range(1, epoch + 1)))
        if not committed: epoch = min(len(history), len(usage), max(0, epoch - 1))
        row = dict(job, completed_epochs=epoch, status='complete' if epoch == 20 else 'pending' if epoch == 0 else 'in_progress')
        timing = [r['epoch_seconds'] for r in usage[-3:] if r.get('epoch_seconds')]
        if timing:
            row.update(observed_epoch_seconds=median(timing),
                       estimated_remaining_hours=(20-epoch)*median(timing)/3600,
                       eta_scope='Current stage only;excludes ZIP export/setup;future unmeasured stages remain unknown')
        best = run / 'best/checkpoint.json'
        if best.is_file() and epoch:
            best_epoch = json.loads(best.read_text())['epoch']
            if best_epoch <= epoch:
                selected = history[best_epoch - 1]
                row.update(best_epoch=best_epoch, selection_miou=selected['val']['miou'],
                           source_gate_pass=selected.get('source_monitor_gate_pass'),
                           source_delta_pp=selected.get('source_delta_pp'))
        rows.append(row)
    return dict(schema=plan['schema'], jobs=rows, completed_jobs=sum(r['status'] == 'complete' for r in rows),
                total_jobs=len(rows), old_results_used=False,
                accuracy_scope='Monitor/internal-dev selection only; official validation is pending',
                resume_granularity='Completed epochs; repeat an interrupted unfinished epoch')


def run_campaign(root, data, archive, seed=42, session_hours=10., allow_cpu=False,
                 runner=run_one, clock=time.monotonic, exporter=export_workspace, models=None,
                 allow_supervised_acdc=False):
    if not allow_supervised_acdc:
        raise ValueError('This source+ACDC campaign is supervised, not DG. Use scripts.run_dg; '
                         'historical runs require --allow-supervised-acdc.')
    if not 0 < session_hours <= 10:
        raise ValueError('Per-session window must be in (0,10] hours')
    root, archive = Path(root).resolve(), Path(archive).resolve()
    started = clock()
    deadline = started + session_hours * 3600
    plan = prepare_campaign(root, data, seed, models)
    status = 'running'
    snapshot_count = 0

    def snapshot(epoch=None):
        nonlocal snapshot_count
        try:
            report = progress(root, plan)
            report['status'] = status
        except Exception as error:
            # Even truncated logs must not prevent export of valid older slots.
            report = dict(status='recovery_required', completed_jobs=0, total_jobs=len(plan['jobs']), error=str(error))
        write_json(root / 'status.json', report)
        for job in plan['jobs']:
            path = root / 'jobs' / job['id'] / 'run'
            if (path / 'history.jsonl').exists():
                try: plot(path)
                except Exception as error: print('Plot skipped; checkpoint remains valid:', error, flush=True)
        exporter(root, archive)
        snapshot_count += 1
        print(f'CHECKPOINT ZIP ready: {archive}; completed {report["completed_jobs"]}/{len(plan["jobs"])} jobs; epoch {epoch}', flush=True)
        for row in report.get('jobs', []):
            if row['status']=='in_progress' and 'estimated_remaining_hours' in row:
                print(f'ETA {row["id"]}: {row["estimated_remaining_hours"]:.1f}h + export/setup overhead',flush=True)

    try:
        for job in plan['jobs']:
            output = root / 'jobs' / job['id']
            source = root / 'jobs' / (job['model'] + '_source') / 'run/best' if job['stage'] == 'adapt' else None
            print('CAMPAIGN JOB:', job['id'], '20 epochs; fresh campaign only', flush=True)
            args = SimpleNamespace(model=job['model'], stage=job['stage'], loss='ce', data=str(Path(data).resolve()),
                output=str(output), checkpoint=str(source) if source else None, initialization='imagenet_encoder',
                class_crop=False, seed=seed, session_hours=session_hours, allow_cpu=allow_cpu, inspect=False,
                prepare_only=False, allow_supervised_acdc=True)
            runner(args, on_epoch=snapshot, deadline=deadline)
            job_status = json.loads((output / 'status.json').read_text())
            if job_status['status'] != 'training_complete':
                status = 'session_checkpoint_ready'
                break
        else:
            status = 'training_complete'
    except Exception as error:
        status = 'failed_preserve_and_resume'
        write_json(root / 'failure.json', dict(error=str(error)))
        raise
    finally:
        write_json(root / 'invocation.json', dict(seconds=clock() - started, session_hours=session_hours,
            total_hours_cap=None, snapshots_written=snapshot_count, status=status))
        snapshot()
    return json.loads((root / 'status.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--session-hours', type=float, default=10.)
    parser.add_argument('--allow-cpu', action='store_true', help='Only for small integration QA')
    parser.add_argument('--allow-supervised-acdc', action='store_true', help='Legacy supervised experiment, not DG')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, help='Select models;each runs source20 then adapt20')
    a = parser.parse_args()
    run_campaign(a.root, a.data, a.archive, a.seed, a.session_hours, a.allow_cpu, models=a.models,
                 allow_supervised_acdc=a.allow_supervised_acdc)
