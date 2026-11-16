"""Optional RGB-only jitter with an isolated RNG; masks/geometry stay identical."""
import math
import random

from PIL import ImageEnhance


class Photometric:
    def __init__(self,options):
        if set(options)!={'probability','brightness','contrast','saturation','gamma'}:raise ValueError('Invalid photometric keys')
        self.probability=float(options['probability'])
        if not 0<=self.probability<=1:raise ValueError('Invalid photometric probability')
        self.ranges={}
        for key in ('brightness','contrast','saturation','gamma'):
            lo,hi=map(float,options[key])
            if not all(math.isfinite(v) for v in (lo,hi)) or not 0<lo<=hi:raise ValueError('Invalid photometric range')
            self.ranges[key]=(lo,hi)

    def apply(self,image,seed,epoch,index):
        rng=random.Random(f'photo:{seed}:{epoch}:{index}')
        if rng.random()>=self.probability:return image
        for name,operation in [('brightness',ImageEnhance.Brightness),('contrast',ImageEnhance.Contrast),('saturation',ImageEnhance.Color)]:
            image=operation(image).enhance(rng.uniform(*self.ranges[name]))
        exponent=rng.uniform(*self.ranges['gamma'])
        lut=[round(255*(i/255)**exponent) for i in range(256)]
        return image.point(lut*3)
