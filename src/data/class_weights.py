"""Class loss weights computed from fit labels only, never evaluation scores."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.validation import read_manifest
from src.data.preparation import resolve_data_path
from src.utils.artifacts import sha256_file, write_json


def fit_class_weights(manifests, destination, cap=5.):
    if not np.isfinite(cap) or cap < 1 or not manifests:
        raise ValueError('Require fit manifests and max/min weight ratio >=1')
    frequencies, counts = [], []
    for manifest in manifests:
        rows = read_manifest(manifest, allow_target_train=True)
        if not rows or any(row['split'] != 'train' for row in rows):
            raise ValueError('Class loss weighting accepts training fit labels only')
        def scan(row):
            with Image.open(resolve_data_path(row['mask'])) as mask:
                values = np.asarray(mask)
                if values.ndim != 2 or not np.isin(values, [*range(19), 255]).all():
                    raise ValueError('Require valid19-class trainIds masks')
                return np.bincount(values.reshape(-1), minlength=256)[:19]
        with ThreadPoolExecutor(max_workers=4) as pool:
            pixels = np.stack(list(pool.map(scan, rows))).sum(0)
        if not pixels.sum():
            raise ValueError('No valid fit pixels')
        counts.append(pixels.tolist())
        frequencies.append(pixels / pixels.sum())
    # Equal mixture of target/source priors for the1:1 replay recipe.
    frequency = np.mean(frequencies, axis=0)
    present = frequency > 0
    weights = np.zeros(19)
    inverse = 1 / np.sqrt(frequency[present])
    inverse = np.minimum(inverse, inverse.min() * cap)
    weights[present] = inverse / inverse.mean()
    report = dict(fit_manifest_sha256=[sha256_file(p) for p in manifests],
                  counts_by_fit_manifest=counts, frequency=frequency.tolist(), weights=weights.tolist(),
                  method='equal-domain-mixture inverse-square-root pixel frequency; capped max/min; mean-present1',
                  max_min_ratio=cap, absent_classes=np.flatnonzero(~present).tolist(),
                  limitation='Not effective-number Class-Balanced Loss; full-mask priors differ from augmented-crop exposure.')
    write_json(destination, report)
    return report
