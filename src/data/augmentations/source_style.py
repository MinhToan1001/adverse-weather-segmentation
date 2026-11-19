"""Source-only RGB perturbations; no target images, masks or style statistics."""
import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class SourceStyle:
    KEYS = {'brightness', 'contrast', 'saturation', 'gamma', 'blur_radius', 'noise_std'}

    def __init__(self, options):
        if set(options) != self.KEYS:
            raise ValueError('Require the six declared source-style ranges')
        self.ranges = {}
        for key, values in options.items():
            lo, hi = map(float, values)
            minimum = 0. if key in {'blur_radius', 'noise_std'} else 1e-6
            if not all(math.isfinite(x) for x in (lo, hi)) or not minimum <= lo <= hi:
                raise ValueError('Invalid source-style range: ' + key)
            if key == 'noise_std' and hi > .1:
                raise ValueError('Noise std must be <=0.1 on RGB [0,1]')
            self.ranges[key] = (lo, hi)

    def apply(self, image, seed, epoch, index):
        if image.mode != 'RGB':
            raise ValueError('SourceStyle expects RGB')
        rng = random.Random(f'dg-style:{seed}:{epoch}:{index}')
        result = image.copy()
        for key, operation in [('brightness', ImageEnhance.Brightness),
                               ('contrast', ImageEnhance.Contrast), ('saturation', ImageEnhance.Color)]:
            result = operation(result).enhance(rng.uniform(*self.ranges[key]))
        gamma = rng.uniform(*self.ranges['gamma'])
        lut = [round(255 * (v / 255) ** gamma) for v in range(256)]
        result = result.point(lut * 3)
        radius = rng.uniform(*self.ranges['blur_radius'])
        if radius:
            result = result.filter(ImageFilter.GaussianBlur(radius))
        std = rng.uniform(*self.ranges['noise_std'])
        if std:
            noise = np.random.default_rng(rng.getrandbits(64)).normal(0., std, (*result.size[::-1], 3))
            pixels = np.asarray(result, dtype=np.float32) / 255.
            result = Image.fromarray(np.rint(np.clip(pixels + noise, 0, 1) * 255).astype(np.uint8))
        return result
