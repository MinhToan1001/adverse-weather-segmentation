"""Native 19-class evaluation of different model families, with calibration and caches."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from src.models.registry import load_checkpoint
from src.evaluation.semantic import native_logits
from src.utils.artifacts import write_json, sha256_file
from src.data.dataset import ManifestSegmentationDataset,evaluation_collate
from src.data.labels import PALETTE
from src.data.preparation import resolve_data_path
from src.evaluation.calibration import CalibrationBins
from src.evaluation.metrics import confusion_matrix,scores


def evaluate(checkpoint,family,manifest,out,canvas=(512,1024),device=None,expected_counts=None,expected_split=None):
    out=Path(out)
    if out.exists():raise ValueError('Preserve existing evaluation')
    device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ds=ManifestSegmentationDataset(manifest,canvas,native_labels=True,allow_target_train=True)
    if expected_counts and Counter(r['domain'] for r in ds.items)!=expected_counts:raise ValueError('Incomplete evaluation split')
    if expected_split and any(r['split']!=expected_split for r in ds.items):raise ValueError('Wrong evaluation split')
    if family not in ('segformer','deeplabv3plus'):raise ValueError('Unsupported model family')
    model=load_checkpoint(checkpoint,'deeplabv3plus_resnet50' if family=='deeplabv3plus' else None).to(device).eval()
    if model.config.num_labels!=19:raise ValueError('Need19classes')
    out.mkdir(parents=True);(out/'confusions').mkdir();(out/'panels').mkdir()
    sha=sha256_file
    import shutil
    shutil.copy2(manifest,out/'manifest.csv')
    weights=Path(checkpoint)/('model.safetensors' if (Path(checkpoint)/'model.safetensors').exists() else 'pytorch_model.bin')
    state=dict(status='started',family=family,checkpoint_sha256=sha(weights),config_sha256=sha(Path(checkpoint)/'config.json'),manifest_sha256=sha(manifest),
        image_size=list(canvas),tta=False,dtype='float32',device=device,
        parameters=sum(p.numel() for p in model.parameters()),
        model_implementation=f'{type(model).__module__}.{type(model).__name__}',
        torch=torch.__version__,scope='shared geometry; not upstream paper inference reproduction')
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    write_json(out/'run-state.json',state)
    stats={};confusions={};seen=set();started=time.perf_counter();count=0
    with torch.inference_mode():
        for batch in DataLoader(ds,batch_size=1,collate_fn=evaluation_collate,num_workers=0):
            result=model(pixel_values=batch['pixel_values'].to(device))
            target=batch['labels'][0];domain=batch['domain'][0]
            logits=native_logits(result,0,batch['pixel_values'].shape[-2:],batch['fit_box'][0],target.shape,family)
            if logits.shape!=(1,19,*target.shape) or not torch.isfinite(logits).all():raise ValueError('Invalid native model output')
            probability=logits.softmax(1);confidence,prediction=probability.max(1)
            confidence=confidence[0].cpu().numpy();prediction=prediction[0].cpu()
            conf=confusion_matrix(prediction,target,19);np.save(out/'confusions'/f'{count:04d}.npy',conf.numpy(),allow_pickle=False)
            confusions.setdefault(domain,torch.zeros((19,19),dtype=torch.int64));confusions[domain]+=conf
            stats.setdefault(domain,CalibrationBins());stats[domain].update(confidence,prediction.numpy(),target.numpy())
            if domain not in seen:
                seen.add(domain);rgb=Image.open(resolve_data_path(ds.items[count]['image'])).convert('RGB')
                truth=np.full((*target.shape,3),35,dtype=np.uint8);valid=target.numpy()!=255;truth[valid]=PALETTE[target.numpy()[valid]]
                error=(prediction.numpy()!=target.numpy()) & valid
                fig,axes=plt.subplots(1,5,figsize=(15,3.5))
                panels=[rgb,truth,PALETTE[prediction.numpy()],confidence,error]
                for ax,p,title in zip(axes,panels,['RGB','GT','Prediction','Confidence','Error; void excluded']):
                    if title=='Confidence':
                        shown=ax.imshow(p,cmap='magma',vmin=0,vmax=1);fig.colorbar(shown,ax=ax,fraction=.04,pad=.02)
                    elif title.startswith('Error'):
                        colors=plt.get_cmap('gray').copy();colors.set_bad('#808080')
                        ax.imshow(np.where(valid,p.astype(float),np.nan),cmap=colors,vmin=0,vmax=1)
                    else:ax.imshow(p)
                    ax.set_title(title);ax.axis('off')
                fig.tight_layout();fig.savefig(out/'panels'/(domain+'.png'),dpi=130);plt.close(fig)
            count+=1
            if count%50==0:print(f'Evaluation {out.name}: {count}/{len(ds)}',flush=True)
    total=sum(confusions.values());report=scores(total)
    report.update(n_images=count,by_domain={d:dict(scores(c),n_images=sum(r['domain']==d for r in ds.items),calibration=stats[d].report()) for d,c in confusions.items()},
        evaluation_seconds=time.perf_counter()-started,checkpoint_sha256=state['checkpoint_sha256'],parameters=state['parameters'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None,
        timing_scope='Includes data I/O, native metrics, calibration, cache and panels; not pure inference FPS')
    pooled=CalibrationBins()
    for bins in stats.values():
        for key in ('count','confidence','correct','accepted','accepted_correct'):setattr(pooled,key,getattr(pooled,key)+getattr(bins,key))
    report['calibration']=pooled.report()
    write_json(out/'metrics.json',report)
    cache_hashes={str(p.relative_to(out)):sha(p) for p in sorted((out/'confusions').glob('*.npy'))}
    write_json(out/'run-state.json',dict(state,status='complete',n_images=count,cache_sha256=cache_hashes,metrics_sha256=sha(out/'metrics.json')))
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return report
