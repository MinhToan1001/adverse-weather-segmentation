"""Summarize source warm-up without certifying a full-run budget automatically."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def summarize(run_dir, full_config, out):
    import yaml
    run_dir,out=Path(run_dir),Path(out)
    history=[json.loads(line) for line in (run_dir/'history.jsonl').read_text().splitlines() if line.strip()]
    resources=[json.loads(line) for line in (run_dir/'resources.jsonl').read_text().splitlines() if line.strip()]
    if not history or not resources:
        raise ValueError('Warm-up history and resource records are required')
    losses=[row['train_loss'] for row in history]
    if any(value is None or not isinstance(value,(int,float)) for value in losses):
        raise ValueError('Invalid loss record')
    import math
    if not all(math.isfinite(value) for value in losses):
        raise ValueError('Nonfinite loss')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'loss.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle);writer.writerow(['epoch','train_loss','source_val_miou'])
        writer.writerows((row['epoch'],row['train_loss'],row['val']['miou']) for row in history)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([row['epoch'] for row in history],losses,marker='o')
    ax.set(xlabel='Epoch',ylabel='Training loss',title='Source-only warm-up (not full baseline)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out/'loss.svg')
    fig.savefig(out/'loss.png',dpi=160)
    plt.close(fig)
    cfg=yaml.safe_load(Path(full_config).read_text())
    from src.data.validation import read_manifest
    source_train=read_manifest(cfg['data']['train_manifest'])
    source_val=read_manifest(cfg['data']['val_manifest'])
    if any(row['domain'] not in {'cityscapes','synthetic'} for row in source_train+source_val):
        raise ValueError('Budget estimation must use source-only manifests')
    full_train=len(source_train)
    full_val=len(source_val)
    # Discard epoch 1 startup where possible. Estimate train and val separately.
    steady=resources[1:] or resources
    projected=[]
    for row in steady:
        if row['batches']<=0 or row['val_images']<=0:
            raise ValueError('No valid batches/validation images')
        batch_size=int(cfg['train']['batch_size'])
        full_batches=math.ceil(full_train/batch_size)
        projected.append(row['train_seconds']/row['batches']*full_batches + row['validation_seconds']/row['val_images']*full_val)
    gpu_verified=all(row['device']=='cuda' for row in resources)
    if gpu_verified:
        measured=json.loads((run_dir/'config.json').read_text())
        if measured.get('model') != cfg.get('model'):
            raise ValueError('Model architecture/checkpoint differs from full config')
        for key in ['image_size','preprocessing','augment']:
            if measured['data'].get(key) != cfg['data'].get(key):
                raise ValueError(f'Preprocessing differs: {key}')
        for key in ['batch_size','gradient_accumulation_steps','num_workers','amp','amp_dtype']:
            if measured['train'].get(key) != cfg['train'].get(key):
                raise ValueError(f'Training settings differ: {key}')
    report=dict(status='GPU_MEASURED_REVIEW_REQUIRED' if gpu_verified else 'CPU_ONLY_NOT_A_GPU_BUDGET',
        first_loss=losses[0],last_loss=losses[-1],loss_decreased=losses[-1]<losses[0],
        gpu_name=resources[-1]['gpu_name'],peak_allocated_bytes=max((row['max_memory_allocated_bytes'] or 0) for row in resources),
        full_train_images=full_train,full_val_images=full_val,
        projected_epoch_seconds=statistics.median(projected) if gpu_verified else None,
        projected_total_hours=statistics.median(projected)*int(cfg['train']['epochs'])/3600 if gpu_verified else None,
        budget_frozen=False,
        assumptions=['Same hardware, resolution, batch, precision and loader settings as warm-up','Linear extrapolation from a small source-only subset; val scaling estimated separately','Checkpoint/I/O overhead, session quotas and runtime variance require manual review','No ACDC input or metric may inform this budget'])
    (out/'warmup_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True);p.add_argument('--full-config',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();print(json.dumps(summarize(args.run_dir,args.full_config,args.out),indent=2))
