"""Deterministic condition labels, five severity presets and parameter provenance."""
from dataclasses import dataclass
import hashlib

import numpy as np

from .color import rgb_array
from .effects import atmospheric_fog, sensor_night, streak_rain, layered_snow


CONDITIONS = ('clean', 'fog', 'night', 'rain', 'snow')
VERSION = 'k1_weather_v1'
BETAS = (.005, .01, .02, .03, .05)


def weather_seed(seed, epoch=0, index=0):
    digest = hashlib.sha256(f'{VERSION}:{seed}:{epoch}:{index}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little')


@dataclass
class WeatherSample:
    image: object
    condition_id: int
    severity: int
    metadata: dict


class WeatherGenerator:
    def apply(self, image, condition, severity=1, *, seed=42, distance_m=None):
        rgb_array(image)  # Validate without changing global RNG or the input.
        if condition not in CONDITIONS:
            raise ValueError('Unknown condition; choose clean/fog/night/rain/snow')
        if isinstance(severity, bool) or not isinstance(severity, (int, np.integer)) or not 1 <= severity <= 5:
            raise ValueError('severity must be an integer from1 to5')
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        rng = np.random.default_rng(int(seed))
        level = severity - 1
        if condition == 'clean':
            result, parameters, fidelity = image.copy(), {}, 'identity'
        elif condition == 'fog':
            if distance_m is None:
                raise ValueError('Fog needs metric camera-ray distance; no constant-depth fallback is allowed')
            parameters = dict(beta=BETAS[level], airlight=[.9, .9, .9])
            result = atmospheric_fog(image, distance_m, **parameters)
            fidelity = 'atmospheric scattering with supplied metric distance; not full Foggy Cityscapes reproduction'
        elif condition == 'night':
            parameters = dict(exposure=(.45,.3,.2,.12,.07)[level], gamma=(1.,1.05,1.1,1.15,1.2)[level],
                full_well=(8192,6144,4096,3072,2048)[level], read_noise=(.0005,.001,.0015,.002,.003)[level],
                saturation=(.95,.9,.85,.8,.75)[level], glare=.015*severity)
            result = sensor_night(image, rng, **parameters)
            fidelity = 'linearized-sRGB low-light/shot/read-noise approximation; not calibrated RAW simulation'
        elif condition == 'rain':
            parameters = dict(density_per_mp=(600,1100,1800,2600,3600)[level],
                length=(12,16,20,25,30)[level], opacity=(.3,.4,.5,.6,.7)[level],
                angle=-12., haze=(.0,.015,.03,.05,.08)[level])
            result = streak_rain(image, rng, **parameters)
            fidelity = '2D stochastic streak overlay; no 3D rainfall, wet-road or lens model'
        else:
            parameters = dict(density_per_mp=(350,600,950,1400,2000)[level], radius=(1.,1.2,1.5,1.8,2.2)[level],
                opacity=(.45,.55,.65,.75,.85)[level], haze=(.01,.025,.05,.08,.12)[level])
            result = layered_snow(image, rng, **parameters)
            fidelity = '2D multilayer snow overlay; no accumulation or 3D scattering model'
        return WeatherSample(result, CONDITIONS.index(condition), int(severity),
            dict(version=VERSION, condition=condition, condition_id=CONDITIONS.index(condition),
                 severity=int(severity), seed=int(seed), parameters=parameters,
                 fidelity=fidelity, color_space='linear RGB image operations with sRGB input/output'))
