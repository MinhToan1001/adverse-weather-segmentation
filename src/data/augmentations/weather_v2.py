"""Deterministic RGB-only source styles; never read target statistics."""
import random
from pathlib import Path
import numpy as np
from PIL import Image
import yaml

from src.data.augmentations.source_style import SourceStyle


class WeatherStyle:
    def __init__(self, original, expanded=False, corruption=None):
        if corruption not in (None, 'low_light', 'noise', 'haze'):
            raise ValueError('Unknown development corruption')
        self.original = SourceStyle(original)
        self.expanded, self.corruption = expanded, corruption
        self.policy=yaml.safe_load((Path(__file__).resolve().parents[3]/'configs/dg_v2/appearance.yaml').read_text())

    def apply(self, image, seed, epoch, index):
        if image.mode != 'RGB':
            raise ValueError('Need RGB')
        # Identical arm seeds produce identical branch/field draws. Dev uses a
        # distinct key and more severe held-out parameter ranges, fixed per image.
        dev = self.corruption is not None
        key = f'weather-v2:{"dev" if dev else "fit"}:{seed}:{0 if dev else epoch}:{index}'
        rng = random.Random(key)
        kind = self.corruption
        if kind is None:
            if not self.expanded:
                return self.original.apply(image, seed, epoch, index)
            draw = rng.random()
            mix=self.policy['fit_mixture']
            kind = 'original' if draw < mix['original'] else 'low_light' if draw < mix['original']+mix['low_light'] else 'haze'
        if kind == 'original':
            return self.original.apply(image, seed, epoch, index)
        x = np.asarray(image, dtype=np.float32) / 255.
        noise = np.random.default_rng(rng.getrandbits(64))
        height, width = x.shape[:2]
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        xx /= max(width-1, 1); yy /= max(height-1, 1)
        if kind == 'low_light':
            policy=self.policy['low_light'];prefix='dev' if dev else 'fit'
            exposure = rng.uniform(*policy[prefix+'_exposure'])
            gamma = rng.uniform(*policy[prefix+'_gamma'])
            field = policy['spatial_base'] + policy['spatial_amplitude']*np.cos((xx-rng.random())*np.pi)*np.cos((yy-rng.random())*np.pi)
            cast = np.array([rng.uniform(*policy['color_cast']) for _ in range(3)], dtype=np.float32)
            x = exposure*x**gamma*field[:,:,None]*cast
            # Shot/read-noise proxy, not a calibrated physical camera model.
            x += noise.normal(0,1,x.shape)*(policy['read_noise']+policy['shot_noise']*np.sqrt(np.clip(x,0,1)))
        elif kind == 'noise':
            x += noise.normal(0,self.policy['dev_noise_std'],x.shape)
        elif kind == 'haze':
            policy=self.policy['haze']
            alpha = rng.uniform(*policy['dev_alpha' if dev else 'fit_alpha'])
            veil = alpha*(policy['spatial_base']+policy['spatial_amplitude']*(1-yy))[:,:,None]
            x = x*(1-veil)+policy['airlight']*veil
        return Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8))
