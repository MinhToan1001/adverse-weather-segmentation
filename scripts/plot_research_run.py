"""Shared loss/metric curves and per-class learning heatmap; no raw dataset imagery."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.utils.artifacts import records
from src.data.labels import CLASS_NAMES
from src.utils.artifacts import write_json


def plot(run):
    run = Path(run)
    rows = records(run / 'history.jsonl')
    if not rows:
        return
    out = run / 'report'
    out.mkdir(exist_ok=True)
    x = [r['epoch'] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    dg = any('dg_train' in r for r in rows)
    selection = 'Cityscapes val (selection)' if dg else 'Dev'
    specs = [('train_loss', 'Training objective (loss preset)'),
             ('dev_ce', 'Dev unweighted CE (diagnostic)'),
             ('dev_miou', selection+' mIoU (%)'),
             ('dg_components' if dg else 'source', 'DG training components' if dg else 'Source retention (pp)')]
    for ax, (key, title) in zip(axes.flat, specs):
        if key=='dg_components':
            for component in ['clean_ce','styled_ce','consistency_kl']:
                y=[r.get('dg_train',{}).get(component) for r in rows]
                y=[np.nan if value is None else value for value in y]
                if np.isfinite(y).any():ax.plot(x,y,marker='.',label=component)
            ax.legend();ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2)
            continue
        if key == 'train_loss': y = [r['train_loss'] for r in rows]
        elif key == 'dev_ce': y = [r['val'].get('cross_entropy', np.nan) for r in rows]
        elif key == 'dev_miou': y = [100 * r['val']['miou'] for r in rows]
        else: y = [r.get('source_delta_pp', np.nan) for r in rows]
        if np.isfinite(np.asarray(y, dtype=float)).any():
            ax.plot(x, y, marker='o', label='Dev' if key == 'dev_ce' else None)
        else:
            ax.text(.5, .5, 'Not applicable: source-only stage', transform=ax.transAxes, ha='center')
        if key == 'dev_ce' and any('train_eval' in r for r in rows):
            fit = [r.get('train_eval', {}).get('cross_entropy', np.nan) for r in rows]
            ax.plot(x, fit, marker='s', linestyle='--', label='Fit subset (diagnostic)')
            ax.legend()
        if key == 'source' and np.isfinite(np.asarray(y, dtype=float)).any():
            ax.axhline(-1., color='tab:red', linestyle='--', label='Source gate -1pp')
            ax.legend()
        if dg:title=title.replace('Dev unweighted CE',selection+' unweighted CE')
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'learning_curves.png', dpi=140); plt.close(fig)
    values = np.array([[np.nan if v is None else v for v in r['val']['per_class_iou']] for r in rows])
    fig, ax = plt.subplots(figsize=(11, 7))
    image = ax.imshow(values.T * 100, aspect='auto', vmin=0, vmax=100, cmap='viridis')
    ax.set_yticks(np.arange(19), CLASS_NAMES); ax.set_xticks(np.arange(len(x)), x)
    ax.set(xlabel='Epoch', title='Per-class dev IoU; undefined (zero-union) classes blank')
    fig.colorbar(image, ax=ax, label='IoU (%)'); fig.tight_layout()
    fig.savefig(out / 'class_learning.png', dpi=140); plt.close(fig)
    write_json(out / 'summary.json', dict(epochs_completed=x[-1], planned_epochs=20,
        final_dev_miou=rows[-1]['val']['miou'],
        losses_are_different_objectives='Training focal/Dice/weighted objectives are not directly comparable to unweighted dev CE.',
        accuracy_scope='Dev only; CPU/smoke data are never a benchmark result.'))
    (out / 'README.md').write_text('# Learning diagnostics\n\n'
        'learning_curves.png: objective, dev CE/mIoU and source retention.\n\n'
        'class_learning.png: all19 classes across epochs; zero union is blank, false-positive-only class is IoU0.\n\n'
        'Do not subtract focal/Dice training loss from dev CE to estimate an overfitting gap.\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run-dir', required=True)
    plot(parser.parse_args().run_dir)
