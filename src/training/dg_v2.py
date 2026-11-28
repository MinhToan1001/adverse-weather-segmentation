"""DG-v2 epoch trainer; reuse v1 models, loss, optimizer, metrics and transactions."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import ManifestSegmentationDataset, evaluation_collate
from src.data.augmentations.weather_v2 import WeatherStyle
from src.experiments.dg_v2 import validate, selection, ROOT, CORRUPTIONS
from src.models.registry import build_model, load_checkpoint
from src.training.optim import make_optimizer
from src.training.losses import make_criterion
from src.training.dg import backward_batch
from src.training.train import evaluate_model, _seed_worker
from src.training.checkpoints import recover_run, save_transactional
from src.utils.artifacts import load_training_state, sha256_file, write_json


class V2Dataset(ManifestSegmentationDataset):
    def __init__(self, cfg, training=False, corruption=None):
        super().__init__(cfg['data']['train_manifest' if training else 'val_manifest'],
            cfg['data']['image_size'],training=training,native_labels=not training,
            preprocessing=cfg['data']['preprocessing'],augment=cfg['data']['augment'],seed=cfg['seed'])
        original=yaml.safe_load((ROOT/'configs/dg/paired_source.yaml').read_text())['source_style']
        self.corruption=corruption
        if training:
            self.source_style=WeatherStyle(original,expanded=cfg['v2']['expanded_style'])
        elif corruption:
            self.dev_style=WeatherStyle(original,corruption=corruption)

    def __getitem__(self, index):
        item=super().__getitem__(index)
        if self.corruption:
            p=self.preprocessor
            rgb=np.clip(item['pixel_values'].numpy().transpose(1,2,0)*p.std+p.mean,0,1)
            image=Image.fromarray(np.rint(rgb*255).astype(np.uint8))
            item['pixel_values']=p.to_tensor(self.dev_style.apply(image,self.seed,0,index))
        return item


def train_epoch_run(cfg, stop_after=None, qa=False):
    validate(cfg,qa=qa)
    if bool(cfg.get('_qa',False))!=bool(qa):
        raise ValueError('CPU QA must be explicitly marked, never a production result')
    root=Path(cfg['output_dir']);root.mkdir(parents=True,exist_ok=True)
    config_path=root/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=cfg:
        raise ValueError('Preserve original resume config/horizon')
    start=recover_run(root)
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not qa:
        raise ValueError('Enable GPU; CPU is reserved for marked QA')
    generator=torch.Generator().manual_seed(seed)
    fit=V2Dataset(cfg,training=True)
    settings=cfg['train']
    loader=DataLoader(fit,batch_size=settings['batch_size'],shuffle=True,generator=generator,
        num_workers=settings['num_workers'],worker_init_fn=_seed_worker,drop_last=settings['drop_last'])
    batches=min(len(loader),settings.get('max_steps') or len(loader))
    if batches<1:raise ValueError('Empty training loader')
    dev_loaders={}
    for i,kind in enumerate(('clean',*CORRUPTIONS)):
        dataset=V2Dataset(cfg,corruption=None if kind=='clean' else kind)
        dev_loaders[kind]=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0,
            collate_fn=evaluation_collate,generator=torch.Generator().manual_seed(seed+10000+i))
    model=(load_checkpoint(root/'last',cfg['model']['name']) if start else build_model(cfg)).to(device)
    # Avoid safetensors memory mappings across Windows transactional renames.
    if start:model.load_state_dict({k:v.clone() for k,v in model.state_dict().items()},assign=True)
    criterion,provenance=make_criterion(cfg['loss'],[cfg['data']['train_manifest']]);criterion=criterion.to(device)
    optimizer=make_optimizer(model,settings)
    accumulation=settings['gradient_accumulation_steps']
    horizon=math.ceil(batches/accumulation)*settings['epochs'];warmup=min(settings['warmup_steps'],horizon-1)
    def schedule(step):
        if warmup and step<warmup:return (step+1)/warmup
        return max(0.,1-(step-warmup)/max(1,horizon-warmup))**settings['poly_power']
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    amp=bool(settings['amp']) and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    step=0;best=-math.inf;best_epoch=None
    if start:
        state=load_training_state(root/'last/training_state.pt')
        if state['config']!=cfg:raise ValueError('Checkpoint config changed')
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);scaler.load_state_dict(state['scaler'])
        step=state['global_step'];best=state['best_miou'];best_epoch=state['best_epoch']
        rng=state['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch']);generator.set_state(rng['loader'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(rng['cuda'])
    else:
        model.save_pretrained(root/'initial')
        write_json(root/'initialization.json',dict(weights_sha256=sha256_file(root/'initial/model.safetensors'),
            seed=seed,model=cfg['model']['name'],target_data_access='none',qa=qa))
    write_json(config_path,cfg);write_json(root/'loss_provenance.json',provenance)
    for epoch in range(start,min(stop_after or settings['epochs'],settings['epochs'])):
        started=time.monotonic();before=step;fit.set_epoch(epoch);model.train();losses=[];paired_count=0
        optimizer.zero_grad(set_to_none=True)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        for batch_index,batch in enumerate(loader):
            if batch_index>=batches:break
            divisor=min(accumulation,batches-(batch_index//accumulation)*accumulation)
            paired=random.Random(f'dg-pair:{seed}:{epoch}:{batch_index}').random()<.5
            info=backward_batch(model,batch,criterion,scaler,divisor,device,paired=paired,amp_enabled=amp)
            losses.append(info['objective']);paired_count+=int(paired)
            if (batch_index+1)%accumulation==0 or batch_index+1==batches:
                old=scaler.get_scale();scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale()>=old:scheduler.step();step+=1
        train_seconds=time.monotonic()-started
        metrics={kind:evaluate_model(model,dl,device,19,compute_loss=True) for kind,dl in dev_loaders.items()}
        score=selection(metrics['clean']['miou'],{k:metrics[k]['miou'] for k in CORRUPTIONS})
        is_best=score>best
        if is_best:best=score;best_epoch=epoch+1
        record=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),dev=metrics,selection_score=score,
            paired_batches=paired_count,global_step=step,qa=qa)
        usage=dict(epoch=epoch+1,optimizer_updates=step-before,epoch_seconds=time.monotonic()-started,
            train_seconds=train_seconds,device=str(device),fit_images=len(fit),dev_images=len(dev_loaders['clean'].dataset),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
        # Log writes precede checkpoint commit; recover_run rolls an interrupted
        # epoch back until both logs and last/best generations agree.
        for name,row in [('history.jsonl',record),('resources.jsonl',usage)]:
            with (root/name).open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        state=dict(config=cfg,epoch=epoch+1,global_step=step,best_miou=best,best_epoch=best_epoch,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                loader=generator.get_state(),cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
        save_transactional(model,root/'last',state)
        if is_best:save_transactional(model,root/'best',state)
        print(json.dumps(dict(epoch=epoch+1,selection_score=score,clean=metrics['clean']['miou'],
            corruptions={k:metrics[k]['miou'] for k in CORRUPTIONS})),flush=True)
    return best


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--stop-after-epoch',type=int);p.add_argument('--qa',action='store_true')
    a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text())
    train_epoch_run(cfg,a.stop_after_epoch,a.qa)
