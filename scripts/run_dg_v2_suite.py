"""Run the four source-development arms; no target or official-val argument."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from src.data.preparation import convert_label_ids,create_manifest,write_manifest
from src.data.validation import validate_manifests
from src.experiments.dg_v2 import (PROTOCOL,ARMS,ARM_LABELS,GROUPS,ROOT,split_cities,fit_statistics,
    compose,validate,source_fingerprints,verify_reference_split,verify_reference_weights,portable_rows,validate_source_pairs)
from src.experiments.run_support import snapshot_source,fits_session,committed_state
from src.experiments.packaging import export_workspace
from src.utils.artifacts import sha256_file,write_json,records


def prepare(city,root,model='segformer_b2',seed=42,group='all'):
    city,root=Path(city).resolve(),Path(root).resolve()
    if group not in GROUPS:raise ValueError('Unknown experiment group')
    arms=list(GROUPS[group])
    policy=yaml.safe_load((ROOT/'configs/dg_v2/development.yaml').read_text())
    if (policy['protocol']!=PROTOCOL or policy['arms']!=list(ARMS) or policy['epochs']!=20
        or policy['dev_fraction']!=.2 or policy['split_seed']!=42 or policy['pair_probability']!=.5
        or policy['fit_style_mixture']!=dict(original=.5,low_light=.3,haze=.2)
        or policy['class_weights']!=dict(minimum=.75,maximum=2.,exponent=.5)
        or policy['corruptions']!=['low_light','noise','haze'] or policy['target_data_access']!='none'):
        raise ValueError('The policy file must describe the implemented predeclared v2 matrix')
    identity=dict(protocol=PROTOCOL,model=model,seed=seed,group=group,arms=arms,city_root=str(city),
        source_hashes=source_fingerprints(),target_data_access='none',policy=policy)
    plan_path=root/'suite_plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text())
        if any(plan[k]!=v for k,v in identity.items()):raise ValueError('Resume requires original source/model/seed/mount')
        for name,h in plan['frozen_files'].items():
            if sha256_file(root/name)!=h:raise ValueError('Frozen v2 input/config changed: '+name)
        return plan
    if root.exists() and any(root.iterdir()):raise ValueError('Preserve existing workspace; prepare a fresh root')
    root.mkdir(parents=True,exist_ok=True);data=root/'data';data.mkdir()
    # Only train2975 is opened; official val500/test and ACDC never participate.
    images=sorted((city/'leftImg8bit/train').rglob('*_leftImg8bit.png'))
    if len(images)!=2975:raise ValueError('Need full Cityscapes train2975')
    masks=data/'converted'
    def convert(image):
        name=image.name.replace('_leftImg8bit.png','_gtFine_labelTrainIds.png')
        native=city/'gtFine/train'/image.parent.name/name
        target=masks/'gtFine/train'/image.parent.name/name
        if native.is_file():
            import shutil
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(native,target)
        else:convert_label_ids(native.with_name(name.replace('_labelTrainIds.png','_labelIds.png')),target)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(convert,images))
    rows=create_manifest(city,'cityscapes','train',masks)
    fit,dev=split_cities(rows,seed=policy['split_seed'])
    verify_reference_split(fit,dev)
    fit_path=data/'fit.csv';dev_path=data/'dev.csv'
    write_manifest(fit,fit_path);write_manifest(dev,dev_path)
    validate_manifests([fit_path,dev_path],check_files=True)
    validate_source_pairs(fit,dev)
    stats=fit_statistics(fit_path);verify_reference_weights(stats)
    write_json(data/'class_weights.json',stats)
    for name,subset in [('fit',fit),('dev',dev)]:write_manifest(portable_rows(subset),data/f'{name}_portable.csv')
    write_json(data/'split.json',dict(fit_images=len(fit),dev_images=len(dev),fit_cities=sorted({Path(r['image']).parent.name for r in fit}),
        dev_cities=sorted({Path(r['image']).parent.name for r in dev}),seed=policy['split_seed'],target_data_access='none',group='city',requested_dev_fraction=.2))
    preview(data,fit)
    for arm in arms:
        job=root/'jobs'/arm;job.mkdir(parents=True)
        cfg=compose(model,arm,fit_path,dev_path,data/'class_weights.json',job/'run',seed)
        validate(cfg)
        (job/'train.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    frozen={str(p.relative_to(root)).replace(chr(92),'/'):sha256_file(p) for folder in [data,root/'jobs'] for p in sorted(folder.rglob('*')) if p.is_file()}
    plan=dict(identity,schema='automatic_dg_v2_development_suite',epochs=20,
        portable_manifests={name:sha256_file(data/f'{name}_portable.csv') for name in ['fit','dev']},
        frozen_files=frozen,selection='0.5 clean-dev + 0.5 mean fixed corruption-dev',
        historical_feedback_limit='ACDC v1 feedback informed v2 design; not a fresh independent target test')
    snapshot_source(root/'source.zip',identity['source_hashes']);write_json(plan_path,plan)
    return plan


def preview(data,fit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from src.data.augmentations.weather_v2 import WeatherStyle
    from src.data.labels import CLASS_NAMES
    options=yaml.safe_load((ROOT/'configs/dg/paired_source.yaml').read_text())['source_style']
    image=Image.open(fit[0]['image']).convert('RGB');image.thumbnail((768,384))
    fig,axes=plt.subplots(1,4,figsize=(14,4))
    variants=[image,*[WeatherStyle(options,corruption=k).apply(image,42,0,0) for k in ['low_light','noise','haze']]]
    for ax,im,title in zip(axes,variants,['Source','Fixed dev: low light','Fixed dev: noise','Fixed dev: haze']):
        ax.imshow(im);ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(data/'preview.png',dpi=120);plt.close(fig)
    stats=json.loads((data/'class_weights.json').read_text())
    with (data/'class_audit.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['class','fit_pixels','fit_images','loss_weight'])
        w.writerows(zip(CLASS_NAMES,stats['pixel_counts'],stats['image_counts'],stats['weights']))


def report(root):
    root=Path(root);rows=[];histories={}
    plan=json.loads((root/'suite_plan.json').read_text())
    arms=plan['arms']
    for arm in arms:
        run_root=root/'jobs'/arm/'run'
        history=records(run_root/'history.jsonl');usage=records(run_root/'resources.jsonl')
        committed=0
        last=run_root/'last/checkpoint.json';best_file=run_root/'best/checkpoint.json'
        if last.is_file() and best_file.is_file():
            last_meta=json.loads(last.read_text());best_meta=json.loads(best_file.read_text())
            epoch=last_meta['epoch'];prefix=list(range(1,epoch+1))
            if (best_meta['epoch']==last_meta['best_epoch']
                and [r['epoch'] for r in history[:epoch]]==prefix
                and [r['epoch'] for r in usage[:epoch]]==prefix):committed=epoch
        history=history[:committed]
        histories[arm]=history
        best=max(history,key=lambda r:r['selection_score']) if history else None
        rows.append(dict(arm=arm,label=ARM_LABELS[arm],epochs=len(history),best_epoch=best['epoch'] if best else None,
            selection_score=best['selection_score'] if best else None,
            clean_dev_miou=best['dev']['clean']['miou'] if best else None,
            corruption_dev_miou={k:v['miou'] for k,v in best['dev'].items() if k!='clean'} if best else None))
    if all(r['epochs']==20 for r in rows):
        initial=[json.loads((root/'jobs'/a/'run/initialization.json').read_text())['weights_sha256'] for a in arms]
        if len(set(initial))!=1:raise ValueError('Arms have different initial weights')
    result=dict(protocol=PROTOCOL,group=plan['group'],status='suite_complete' if all(r['epochs']==20 for r in rows) else 'session_checkpoint_ready',
        jobs=rows,target_data_access='none',qa=False)
    write_json(root/'suite_status.json',result)
    with (root/'suite_summary.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['arm','label','epochs','best_epoch','selection_score','clean_dev_miou','corruption_dev_miou']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    if any(histories.values()):plot_history(root,histories)
    return result


def plot_history(root,histories):
    """Plot committed epochs only; dev curves are not target accuracy results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,len(histories),figsize=(6*len(histories),7),squeeze=False)
    for column,(arm,history) in enumerate(histories.items()):
        axes[0,column].set_title(ARM_LABELS[arm])
        if history:
            epochs=[r['epoch'] for r in history]
            if all('train_loss' in r for r in history):
                axes[0,column].plot(epochs,[r['train_loss'] for r in history],label='Train loss')
            axes[1,column].plot(epochs,[100*r['dev']['clean']['miou'] for r in history],label='Clean dev')
            axes[1,column].plot(epochs,[100*sum(r['dev'][k]['miou'] for k in ['low_light','noise','haze'])/3
                for r in history],label='Mean corrupted dev')
            axes[1,column].plot(epochs,[100*r['selection_score'] for r in history],label='Selection score')
            axes[1,column].legend()
        for row in range(2):axes[row,column].set_xlabel('Committed epoch');axes[row,column].grid(alpha=.25)
        axes[0,column].set_ylabel('Loss');axes[1,column].set_ylabel('mIoU / score (%)')
    fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=120);plt.close(fig)


def run(city,root,archive,model='segformer_b2',seed=42,session_hours=10,prepare_only=False,group='all'):
    root,archive=Path(root).resolve(),Path(archive).resolve()
    if archive.is_relative_to(root) or not 0<session_hours<=10:raise ValueError('Invalid ZIP location or session window')
    started=time.monotonic();plan=prepare(city,root,model,seed,group)
    try:
        if prepare_only:return report(root)
        import torch
        if not torch.cuda.is_available():raise ValueError('Enable GPU; --prepare-only supports CPU')
        from src.training.checkpoints import recover_run
        for arm in plan['arms']:
            print('EXPERIMENT:',ARM_LABELS[arm],flush=True)
            job=root/'jobs'/arm;run_root=job/'run';saved=recover_run(run_root)
            if saved:committed_state(run_root,saved)
            for epoch in range(saved+1,21):
                usage=records(run_root/'resources.jsonl')
                if not fits_session(time.monotonic(),started+session_hours*3600,[r['epoch_seconds'] for r in usage]):
                    print('Session window: preserve ZIP and resume in a new session',flush=True);return report(root)
                subprocess.run([sys.executable,'-u','-m','src.training.dg_v2','--config',str(job/'train.yaml'),
                    '--stop-after-epoch',str(epoch)],check=True)
                committed_state(run_root,epoch);report(root);export_workspace(root,archive)
        return report(root)
    finally:
        report(root);export_workspace(root,archive)
        print('Download:',archive,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['city','root','archive']:p.add_argument('--'+n,required=True)
    p.add_argument('--model',default='segformer_b2',choices=['segformer_b2','deeplabv3plus_resnet50'])
    p.add_argument('--seed',type=int,default=42);p.add_argument('--session-hours',type=float,default=10.)
    p.add_argument('--group',choices=list(GROUPS),default='all',help='Two parallel pairs, or all four locally')
    p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    run(a.city,a.root,a.archive,a.model,a.seed,a.session_hours,a.prepare_only,a.group)
