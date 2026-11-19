"""GPU timing pilot on Cityscapes only; discard all pilot weights afterwards."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
from statistics import median
import time

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
import yaml

from src.data.dataset import ManifestSegmentationDataset,evaluation_collate
from src.experiments.dg import compose_dg
from src.experiments.protocol import require_training_permission
from src.models.registry import build_model
from src.training.dg import backward_batch
from src.training.losses import SegmentationLoss
from src.training.optim import make_optimizer
from src.utils.artifacts import write_json,sha256_file


def estimated_hours(unpaired_seconds,paired_seconds,pair_probability,batches,validation_seconds,epochs=20):
    values=[unpaired_seconds,paired_seconds,pair_probability,batches,validation_seconds,epochs]
    if not all(math.isfinite(x) and x>=0 for x in values) or not 0<=pair_probability<=1:
        raise ValueError('Invalid profiling observations')
    seconds=(1-pair_probability)*unpaired_seconds+pair_probability*paired_seconds
    return epochs*(batches*seconds+validation_seconds)/3600


def profile(workspace,out,warmup=4,steps=12,val_samples=8,pair_probability=.5,consistency_weight=.1):
    # Refuse before touching data if this is not a GPU timing session.
    if not torch.cuda.is_available():raise ValueError('GPU profiling needs CUDA; CPU smoke timings are not GPU ETAs')
    if warmup<4 or steps<4 or steps%4 or val_samples<1:raise ValueError('Use warmup>=4, steps multiple of4, val_samples>=1')
    if not 0<pair_probability<=1 or not 0<=consistency_weight<=10:raise ValueError('Invalid candidate DG settings')
    root=Path(workspace).resolve();out=Path(out)
    if out.exists():raise ValueError('Preserve the previous profile report')
    cfg=yaml.safe_load((root/'train.yaml').read_text());require_training_permission(cfg)
    plan=json.loads((root/'dg_plan.json').read_text())
    if any(sha256_file(root/n)!=h for n,h in plan['frozen_files'].items()):raise ValueError('Frozen profiling data/config changed')
    paths=dict(source_train_manifest=cfg['data']['train_manifest'],source_monitor_manifest=cfg['data']['val_manifest'])
    measurements={};validation_seconds=None
    modes=[('baseline',False),('aug',False),('aug',True),('consistency',False),('consistency',True)]
    for arm,paired in modes:
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
        local=compose_dg(cfg['model']['name'],arm,paths,root/'unused',cfg['seed'],pair_probability,consistency_weight)
        ds=ManifestSegmentationDataset(paths['source_train_manifest'],cfg['data']['image_size'],training=True,
            preprocessing=cfg['data'].get('preprocessing'),augment=cfg['data']['augment'],seed=cfg['seed'],
            paired_style=local['dg']['source_style'])
        loader=DataLoader(ds,batch_size=2,drop_last=True,num_workers=2,shuffle=False)
        iterator=iter(loader)
        model=build_model(local).to('cuda').train();criterion=SegmentationLoss()
        optimizer=make_optimizer(model,cfg['train']);scaler=torch.amp.GradScaler('cuda')
        optimizer.zero_grad(set_to_none=True)
        times=[];torch.cuda.reset_peak_memory_stats()
        for index in range(warmup+steps):
            torch.cuda.synchronize();started=time.perf_counter()
            batch=next(iterator)
            backward_batch(model,batch,criterion,scaler,4,'cuda',paired,
                           local['dg']['consistency_weight'],True,torch.float16)
            if (index+1)%4==0:
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if index>=warmup:times.append(time.perf_counter()-started)
        key=f'{arm}_{"paired" if paired else "single"}'
        measurements[key]=dict(median_batch_seconds=median(times),mean_batch_seconds=sum(times)/len(times),
            batch_count=len(times),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if validation_seconds is None:
            from src.training.train import evaluate_model
            val=ManifestSegmentationDataset(paths['source_monitor_manifest'],cfg['data']['image_size'],
                native_labels=True,preprocessing=cfg['data'].get('preprocessing'))
            selected=Subset(val,np.linspace(0,len(val)-1,min(val_samples,len(val)),dtype=int).tolist())
            vl=DataLoader(selected,batch_size=2,collate_fn=evaluation_collate,num_workers=0)
            torch.cuda.synchronize();started=time.perf_counter()
            evaluate_model(model,vl,'cuda',19,compute_loss=True)
            torch.cuda.synchronize();validation_seconds=(time.perf_counter()-started)*len(val)/len(selected)
        del iterator,loader,model,optimizer,scaler
        torch.cuda.empty_cache()
        print('Timing pilot complete:',key,flush=True)
    forecasts={}
    for arm in ['baseline','aug','consistency']:
        a=measurements[f'{arm}_single']['mean_batch_seconds']
        b=a if arm=='baseline' else measurements[f'{arm}_paired']['mean_batch_seconds']
        p=0 if arm=='baseline' else pair_probability
        forecasts[arm]=estimated_hours(a,b,p,len(ds)//2,validation_seconds)
    report=dict(status='PASS',model=cfg['model']['name'],gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        measurements=measurements,estimated_20epoch_hours_excluding_checkpoint_zip_setup=forecasts,
        validation500_seconds_extrapolated=validation_seconds,pair_probability_used_for_forecast=pair_probability,
        consistency_weight_used_for_paired_pilot=consistency_weight,
        target_data_access='none',pilot_weights_discarded=True,
        limitations='Short source-only pilot; estimates exclude checkpoint/ZIP/setup and hardware/runtime variation. Reserve time before admitting full runs.')
    write_json(out,report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True);p.add_argument('--out',required=True)
    p.add_argument('--warmup',type=int,default=4);p.add_argument('--steps',type=int,default=12)
    p.add_argument('--pair-probability',type=float,default=.5);p.add_argument('--consistency-weight',type=float,default=.1)
    a=p.parse_args();profile(a.workspace,a.out,a.warmup,a.steps,pair_probability=a.pair_probability,consistency_weight=a.consistency_weight)
