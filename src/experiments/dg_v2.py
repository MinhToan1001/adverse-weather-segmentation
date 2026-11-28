"""Source-only group split, fit-only class audit and frozen DG-v2 composition."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import yaml

from src.data.preparation import write_manifest, resolve_data_path
from src.data.validation import read_manifest
from src.experiments.config import compose_experiment
from src.utils.artifacts import sha256_file, write_json

PROTOCOL = 'cityscapes_groupdev_dg_v2'
ARMS = ('baseline','class_weights','weather_aug','weather_aug_class_weights')
ARM_LABELS = dict(baseline='B2 baseline', class_weights='B2 + class weights',
    weather_aug='B2 + weather augmentation', weather_aug_class_weights='B2 + weather augmentation + class weights')
GROUPS = dict(baseline_weights=ARMS[:2], weather_weights=ARMS[2:], all=ARMS)
WEIGHTED_ARMS = ('class_weights','weather_aug_class_weights')
WEATHER_ARMS = ('weather_aug','weather_aug_class_weights')
CORRUPTIONS = ('low_light','noise','haze')
ROOT = Path(__file__).resolve().parents[2]


def source_fingerprints():
    from src.experiments.run_support import fingerprints
    return dict(fingerprints(),**{p.relative_to(ROOT).as_posix():sha256_file(p)
        for p in sorted((ROOT/'configs/dg_v2/splits').glob('*.csv'))})


def portable_rows(rows):
    """Dataset-root-relative sample identities, independent of OS/mount/output."""
    return sorted([dict(image=f"leftImg8bit/train/{city(r)}/{Path(r['image']).name}",
        mask=f"gtFine/train/{city(r)}/{Path(r['mask']).name}",domain=r['domain'],split=r['split'])
        for r in rows],key=lambda r:r['image'])


def verify_reference_split(fit,dev):
    for name,rows in [('fit',fit),('dev',dev)]:
        expected=read_manifest(ROOT/f'configs/dg_v2/splits/{name}.csv')
        if portable_rows(rows)!=expected:
            raise ValueError('Source split differs from the shared portable '+name+' manifest')


def verify_reference_weights(stats):
    reference=json.loads((ROOT/'configs/dg_v2/class_weights_reference.json').read_text())
    if (stats['pixel_counts']!=reference['pixel_counts'] or stats['image_counts']!=reference['image_counts']
        or not np.allclose(stats['weights'],reference['weights'],rtol=0,atol=1e-12)):
        raise ValueError('Source fit label statistics differ from the shared class-weight reference')


def validate_source_pairs(fit,dev):
    """Check every pair's geometry; fit pixels are checked by fit_statistics.

    Read image headers instead of decoding all RGB pixels a second time. Dev
    masks are checked here because they never enter fit-only class statistics.
    """
    for rows,check_pixels in [(fit,False),(dev,True)]:
        for row in rows:
            with Image.open(resolve_data_path(row['image'])) as image, Image.open(resolve_data_path(row['mask'])) as mask:
                if mask.mode!='L' or image.size!=mask.size:
                    raise ValueError('Invalid source image/mask geometry or mask mode')
                if check_pixels and not np.isin(np.asarray(mask),[*range(19),255]).all():
                    raise ValueError('Invalid source dev trainIds')


def city(row):
    return Path(row['image']).parent.name


def split_cities(rows, fraction=.2, seed=42):
    if not 0 < fraction < 1 or any(r['domain']!='cityscapes' or r['split']!='train' for r in rows):
        raise ValueError('Split only official Cityscapes train, with a proper dev fraction')
    groups = sorted({city(r) for r in rows})
    if len(groups)<2:
        raise ValueError('Need at least two source cities')
    random.Random(seed).shuffle(groups)
    # Subset sum chooses a whole-city split near the requested image count.
    choices = {0:()}
    for group in groups:
        count=sum(city(r)==group for r in rows)
        for total,chosen in list(choices.items()):
            choices.setdefault(total+count,chosen+(group,))
    total=min((n for n in choices if 0<n<len(rows)),key=lambda n:(abs(n-len(rows)*fraction),n))
    dev_cities=set(choices[total])
    return ([r for r in rows if city(r) not in dev_cities], [r for r in rows if city(r) in dev_cities])


def fit_statistics(manifest):
    rows=read_manifest(manifest)
    if any(r['domain']!='cityscapes' or r['split']!='train' for r in rows):
        raise ValueError('Weights must only use source fit labels')
    pixels=np.zeros(19,dtype=np.int64);images=np.zeros(19,dtype=np.int64)
    for row in rows:
        with Image.open(row['mask']) as im:
            x=np.asarray(im)
            if im.mode!='L' or x.ndim!=2 or not np.isin(x,[*range(19),255]).all():
                raise ValueError('Invalid source trainId mask')
            counts=np.bincount(x.ravel(),minlength=256)[:19]
            pixels+=counts; images+=counts>0
    if (pixels==0).any():
        raise ValueError('Fit split is missing classes; do not invent weights for unseen classes')
    frequency=pixels/pixels.sum()
    weights=np.clip(np.sqrt(np.median(frequency)/(frequency+1e-12)),.75,2.)
    return dict(fit_manifest_sha256=[sha256_file(manifest)],pixel_counts=pixels.tolist(),
        image_counts=images.tolist(),weights=weights.tolist(),
        formula='clip(sqrt(median(frequency)/(frequency+1e-12)),0.75,2.0)',target_data_access='none')


def compose(model, arm, fit, dev, weights, output, seed=42):
    if model not in ('segformer_b2','deeplabv3plus_resnet50') or arm not in ARMS:
        raise ValueError('Unknown v2 model/arm')
    cfg=compose_experiment(model,'source',dict(source_train_manifest=str(fit),source_monitor_manifest=str(dev)),output,seed=seed)
    if model=='segformer_b2':
        cfg['model']=yaml.safe_load((ROOT/'configs/dg_v2/segformer_b2.yaml').read_text())['model']
    cfg.update(experiment_protocol=PROTOCOL,training_policy=PROTOCOL,arm=arm,
        experiment=f'{model}-dg-v2-{arm}-seed{seed}',
        selection=dict(metric='0.5_clean_plus_0.5_mean_corruptions',target_data_access='none'),
        v2=dict(expanded_style=arm in WEATHER_ARMS,pair_probability=.5,corruptions=list(CORRUPTIONS)))
    if arm in WEIGHTED_ARMS:
        cfg['loss']=dict(name='weighted_ce',class_weights_file=str(weights))
    return cfg


def validate(cfg, qa=False):
    if cfg.get('experiment_protocol')!=PROTOCOL or cfg.get('protocol')!='source_only':
        raise ValueError('V2 is a separate source-only protocol')
    if cfg.get('arm') not in ARMS or cfg.get('selection')!=dict(metric='0.5_clean_plus_0.5_mean_corruptions',target_data_access='none'):
        raise ValueError('Unexpected arm/selection rule')
    if cfg.get('v2')!=dict(expanded_style=cfg['arm'] in WEATHER_ARMS,pair_probability=.5,corruptions=list(CORRUPTIONS)):
        raise ValueError('V2 matrix policy changed')
    if set(cfg['data'])!={'train_manifest','val_manifest','image_size','preprocessing','augment'}:
        raise ValueError('No extra target/data strategies in the v2 matrix')
    if cfg['model'].get('name') not in ('segformer_b2','deeplabv3plus_resnet50') or cfg['model'].get('num_labels')!=19:
        raise ValueError('Require a declared19-class model')
    fit=read_manifest(cfg['data']['train_manifest']);dev=read_manifest(cfg['data']['val_manifest'])
    if any(r['domain']!='cityscapes' or r['split']!='train' for r in fit+dev):
        raise ValueError('Fit and dev must both be held within Cityscapes train')
    if {city(r) for r in fit}&{city(r) for r in dev}:
        raise ValueError('Source fit/dev cities overlap')
    if len({r['image'] for r in fit+dev})!=len(fit)+len(dev):
        raise ValueError('Fit/dev images overlap')
    weighted=cfg['arm'] in WEIGHTED_ARMS
    if cfg['loss'].get('name')!=('weighted_ce' if weighted else 'ce'):
        raise ValueError('Loss must match the predeclared arm')
    if not qa and (len(fit)+len(dev)!=2975 or cfg['train']['epochs']!=20
        or cfg['model'].get('initialization')!='imagenet_encoder' or cfg['model'].get('checkpoint')
        or cfg.get('early_stopping') or cfg['train'].get('max_steps') is not None
        or cfg['data']['image_size']!=[512,1024]):
        raise ValueError('Production v2 requires all2975 source train images,20 epochs and ImageNet initialization')
    if not qa:
        verify_reference_split(fit,dev)
        recipe=yaml.safe_load((ROOT/'configs/recipes/source_20epoch.yaml').read_text())
        processing=yaml.safe_load((ROOT/'configs/preprocessing/shared_512.yaml').read_text())
        if (cfg['train']!=recipe['train'] or any(cfg['data'][k]!=v for k,v in processing.items())
            or cfg['data']['augment']!=dict(min_scale=.5,max_scale=2.,horizontal_flip=.5)):
            raise ValueError('Keep the shared optimizer/processing/crop recipe across v2 arms')
    if weighted:
        artifact=json.loads(Path(cfg['loss']['class_weights_file']).read_text())
        if artifact['fit_manifest_sha256']!=[sha256_file(cfg['data']['train_manifest'])]:
            raise ValueError('Weights are not bound to the actual fit manifest')
        if not qa:verify_reference_weights(artifact)
    return fit,dev


def selection(clean, corruptions):
    if set(corruptions)!=set(CORRUPTIONS):
        raise ValueError('Selection requires all three fixed corruption dev passes')
    values=[float(clean),*[float(corruptions[k]) for k in CORRUPTIONS]]
    if not all(np.isfinite(values)):
        raise ValueError('Non-finite dev score')
    return .5*values[0]+.5*float(np.mean(values[1:]))
