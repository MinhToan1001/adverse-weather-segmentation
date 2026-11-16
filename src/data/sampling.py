"""Fit-only class-presence weights, separated from dev and official labels."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.validation import read_manifest
from src.utils.artifacts import sha256_file as sha, write_json
from src.data.preparation import resolve_data_path


def presence_weights(presence,cap=5.):
    presence=np.asarray(presence,dtype=bool)
    if not np.isfinite(cap) or cap<1:raise ValueError('Sampling max/min ratio must be finite and >=1')
    if presence.ndim!=2 or not len(presence) or presence.shape[1]!=19 or not presence.any(1).all():raise ValueError('Each fit image needs at least one present class')
    frequency=presence.mean(0)
    rarity=1/np.sqrt(np.maximum(frequency,1/len(presence)))
    weights=(presence*rarity).sum(1)/presence.sum(1)
    weights=np.minimum(weights,weights.min()*cap);weights/=weights.mean()
    return weights,frequency


def build_weights(manifest,out,min_pixels=64,cap=5.):
    if not isinstance(min_pixels,int) or min_pixels<1:raise ValueError('Require positive pixel threshold')
    items=read_manifest(manifest, allow_target_train=True)
    if not items or any(r['split']!='train' or r['domain'] not in ('fog','night','rain','snow') for r in items):
        raise ValueError('Class sampling requires target official-train fit only')
    def scan(row):
        with Image.open(resolve_data_path(row['mask'])) as m:
            a=np.asarray(m)
            if a.ndim!=2 or not np.isin(a,[*range(19),255]).all():raise ValueError('Invalid trainIds')
            counts=np.bincount(a.reshape(-1).astype(np.int64),minlength=256)
        present=counts[:19]>=min_pixels
        if not present.any():present=counts[:19]>0
        return present
    with ThreadPoolExecutor(max_workers=4) as pool:presence=np.stack(list(pool.map(scan,items)))
    weights,frequency=presence_weights(presence,cap)
    report=dict(fit_manifest_sha256=sha(manifest),weights=weights.tolist(),class_image_frequency=frequency.tolist(),
        expected_sampled_class_frequency=np.average(presence,axis=0,weights=weights).tolist(),
        minimum_class_pixels=min_pixels,max_to_min_weight_ratio=cap,method='Mean inverse-square-root image-frequency of present classes; capped max/min ratio, mean1',
        limit='Image-level resampling; not DAFormer reproduction or guaranteed rare-object recall')
    write_json(out,report);return report
