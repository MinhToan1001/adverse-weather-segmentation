"""Training-label-guided crop; only fit labels may reach this function."""
import random

import numpy as np


def choose_class_crop(mask, crop_hw, options, rng=random):
    if set(options) - {'probability', 'min_pixels', 'attempts'}:
        raise ValueError('Unknown class-aware crop setting')
    probability = float(options.get('probability', .5))
    minimum, attempts = options.get('min_pixels', 64), options.get('attempts', 10)
    if not 0 <= probability <= 1 or not isinstance(minimum, int) or minimum < 1 or not isinstance(attempts, int) or attempts < 1:
        raise ValueError('Invalid crop probability/min_pixels/attempts')
    values = np.asarray(mask)
    height, width = values.shape
    crop_h, crop_w = min(crop_hw[0], height), min(crop_hw[1], width)
    if rng.random() >= probability:
        return None
    counts = np.bincount(values.reshape(-1), minlength=256)[:19]
    available = np.flatnonzero(counts >= minimum)
    if not len(available):
        return None
    chosen = rng.choices(available.tolist(), weights=(1 / np.sqrt(counts[available])).tolist(), k=1)[0]
    ys, xs = np.where(values == chosen)
    best = None
    for _ in range(attempts):
        pixel = rng.randrange(len(ys))
        y, x = int(ys[pixel]), int(xs[pixel])
        top = rng.randint(max(0, y - crop_h + 1), min(y, height - crop_h))
        left = rng.randint(max(0, x - crop_w + 1), min(x, width - crop_w))
        seen = int((values[top:top + crop_h, left:left + crop_w] == chosen).sum())
        item = dict(top=top, left=left, class_id=chosen, pixels=seen, met=seen >= minimum)
        if best is None or seen > best['pixels']:
            best = item
        if item['met']:
            break
    return best
