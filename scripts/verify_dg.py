"""Real Cityscapes CPU plumbing checks, never a convergence/accuracy benchmark."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file
import torch
import yaml

from src.data.dataset import ManifestSegmentationDataset
from src.data.labels import PALETTE
from src.data.preparation import write_manifest
from src.experiments.dg import compose_dg
from src.models.registry import MODEL_NAMES, build_model
from src.utils.artifacts import sha256_file, records, write_json

ROOT=Path(__file__).resolve().parents[1]


def verify(data_root, output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    paths={}
    for split,count,key in [('train',4,'source_train_manifest'),('val',2,'source_monitor_manifest')]:
        with (Path(data_root)/f'cityscapes_{split}.csv').open(encoding='utf-8') as handle:
            rows=list(csv.DictReader(handle))[:count]
        if len(rows)!=count or any(r['domain']!='cityscapes' or r['split']!=split for r in rows):
            raise ValueError('QA requires real Cityscapes train/val manifests')
        path=output/f'{split}.csv';write_manifest(rows,path);paths[key]=str(path)
    reports=[]
    for name in MODEL_NAMES:
        resumed=output/name
        cfg=compose_dg(name,'consistency',paths,resumed)
        cfg.pop('experiment_protocol')  # Tiny QA must not bypass full DG2975/500 guards.
        cfg.update(training_policy='cpu_plumbing_only_not_benchmark')
        cfg['model']['initialization']='random'
        cfg['data']['image_size']=[64,96]
        cfg['data']['augment'].update(min_scale=1.,max_scale=1.)
        cfg['train'].update(num_workers=0,max_steps=2,gradient_accumulation_steps=2,amp=False)
        cfg['dg']['pair_probability']=1.  # Ensure the smoke check exercises the paired path.
        config=output/f'{name}.yaml';config.write_text(yaml.safe_dump(cfg),encoding='utf-8')
        with (output/f'{name}.log').open('w',encoding='utf-8') as log:
            for epoch in [1,2]:
                command=[sys.executable,'-u','-m','src.training.train','--config',str(config),'--stop-after-epoch',str(epoch)]
                if epoch==2:command+=['--resume',str(resumed/'last')]
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        continuous=deepcopy(cfg);continuous['output_dir']=str(output/(name+'_continuous'))
        continuous['checkpointing']['transactional']=False  # Fewer redundant QA artifacts; identical training math.
        other=output/f'{name}_continuous.yaml';other.write_text(yaml.safe_dump(continuous),encoding='utf-8')
        with (output/f'{name}_continuous.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u','-m','src.training.train','--config',str(other),'--stop-after-epoch','2'],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        left=load_file(str(resumed/'last/model.safetensors'))
        right=load_file(str(Path(continuous['output_dir'])/'last/model.safetensors'))
        assert left.keys()==right.keys() and all(torch.equal(v,right[k]) for k,v in left.items())
        initial=load_file(str(resumed/'initial/model.safetensors'))
        assert any(not torch.equal(initial[k],v) for k,v in left.items())
        del left,right,initial
        digest=sha256_file(resumed/'initial/model.safetensors')
        # Architecture construction ignores the arm. Verify actual serialized
        # initialization for C0/C1 against the C2 initialization used above.
        init_hashes={'consistency':digest}
        for arm in ['baseline','aug']:
            c=compose_dg(name,arm,paths,output/'unused')
            c['model']['initialization']='random';torch.manual_seed(42)
            model=build_model(c);directory=output/f'{name}_{arm}_initial'
            model.save_pretrained(directory)
            init_hashes[arm]=sha256_file(directory/'model.safetensors')
            del model
        assert len(set(init_hashes.values()))==1
        history=records(resumed/'history.jsonl');usage=records(resumed/'resources.jsonl')
        assert [r['epoch'] for r in history]==[1,2]
        assert all(r['optimizer_updates']==1 and r['dg']['paired_batches']==2 for r in usage)
        assert all(r['dg_train']['consistency_kl'] is not None and np.isfinite(r['train_loss']) for r in history)
        reports.append(dict(model=name,status='PASS',two_epoch_interruption_resume=True,
            continuous_and_resumed_weights_bitwise_equal=True,all_arm_initial_hashes=init_hashes,
            optimizer_updates=[r['optimizer_updates'] for r in usage],random_weights_changed=True,
            loss_and_kl_finite=True,accuracy_claim=False))
        print('CPU DG QA PASS:',name,flush=True)
    preview_cfg=compose_dg('segformer_b2','aug',paths,output/'unused')
    dataset=ManifestSegmentationDataset(paths['source_train_manifest'],(512,1024),training=True,
        augment=dict(min_scale=1.,max_scale=1.,horizontal_flip=0.),paired_style=preview_cfg['dg']['source_style'],seed=42)
    sample=dataset[0];pre=dataset.preprocessor
    def rgb(tensor):
        return np.clip(tensor.numpy().transpose(1,2,0)*pre.std+pre.mean,0,1)
    labels=sample['labels'].numpy();colors=np.full((*labels.shape,3),35,dtype=np.uint8)
    valid=labels!=255;colors[valid]=PALETTE[labels[valid]]
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,im,title in zip(axes,[rgb(sample['pixel_values']),rgb(sample['styled_pixel_values']),colors],
                           ['Cityscapes source view','RGB-only source style','Same GT mask; not model predictions']):
        ax.imshow(im);ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(output/'paired_source_preview.png',dpi=130);plt.close(fig)
    report=dict(status='PASS',scope='CPU only; random initialization,4 Cityscapes train/2val,64x96,2batches/epoch; not GPU validation or quality results',
                reports=reports,target_data_access='none',candidate_pair_probability=.5,probe_pair_probability=1.)
    write_json(output/'qa.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True);parser.add_argument('--out',required=True)
    a=parser.parse_args();verify(a.data_root,a.out)
