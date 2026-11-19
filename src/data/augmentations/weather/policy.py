"""Opt-in source-only sampling; depth is loaded before geometric augmentation."""
from pathlib import Path

import numpy as np

from .generator import WeatherGenerator, CONDITIONS, weather_seed
from .depth import cityscapes_distance


class SourceWeather:
    def __init__(self, options):
        if not isinstance(options, dict) or set(options) - {
            'conditions', 'severities', 'probabilities', 'cityscapes_root', 'distance_root',
            'distance_unit', 'invalid_policy', 'max_distance_m'}:
            raise ValueError('Unknown K1 weather policy options')
        self.conditions = options.get('conditions', ['clean', 'night', 'rain', 'snow'])
        self.severities = options.get('severities', [1,2,3,4,5])
        if (not isinstance(self.conditions, list) or not self.conditions
                or any(c not in CONDITIONS for c in self.conditions) or len(set(self.conditions)) != len(self.conditions)):
            raise ValueError('Choose a nonempty unique list of weather conditions')
        if (not isinstance(self.severities, list) or not self.severities
                or any(isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= 5 for s in self.severities)):
            raise ValueError('severities must be integers1..5')
        self.probabilities = np.asarray(options.get('probabilities', [1/len(self.conditions)]*len(self.conditions)), float)
        if (self.probabilities.shape != (len(self.conditions),) or not np.isfinite(self.probabilities).all()
                or (self.probabilities < 0).any() or not np.isclose(self.probabilities.sum(), 1)):
            raise ValueError('Require finite nonnegative condition probabilities summing to1')
        self.probabilities /= self.probabilities.sum()
        self.city = Path(options['cityscapes_root']) if options.get('cityscapes_root') else None
        self.distances = Path(options['distance_root']) if options.get('distance_root') else None
        if self.city and self.distances:
            raise ValueError('Choose one distance source: Cityscapes stereo or completed distance NPY')
        if self.distances and options.get('distance_unit') != 'meters':
            raise ValueError('Explicitly declare distance_unit: meters for camera-ray distance NPY files')
        if 'fog' in self.conditions and not (self.city or self.distances):
            raise ValueError('Fog policy requires cityscapes_root or metric distance_root; explicitly exclude fog otherwise')
        self.invalid_policy = options.get('invalid_policy', 'error')
        self.max_distance_m = options.get('max_distance_m', 200.)
        if self.invalid_policy not in {'error','far'} or not np.isfinite(self.max_distance_m) or self.max_distance_m <= 0:
            raise ValueError('Invalid distance fill policy or max_distance_m')
        self.generator = WeatherGenerator()

    def apply(self, image, row, seed, epoch, index):
        if row['domain'] not in {'cityscapes', 'synthetic'} or row['split'] != 'train':
            raise ValueError('K1 training policy accepts source train images only, never ACDC')
        rng = np.random.default_rng(weather_seed(seed, epoch, index))
        condition = str(rng.choice(self.conditions, p=self.probabilities))
        severity = int(rng.choice(self.severities))
        distance, depth_info = None, None
        if condition == 'fog':
            filename = Path(row['image']).stem
            if self.distances:
                path = self.distances/(filename+'.npy')
                distance = np.load(path, allow_pickle=False)
                depth_info = dict(source=str(path), units='meters', kind='user_supplied_camera_ray_distance')
            else:
                if not filename.endswith('_leftImg8bit'):
                    raise ValueError('Cityscapes stereo depth requires canonical leftImg8bit filenames')
                stem = filename.removesuffix('_leftImg8bit')
                city = stem.rsplit('_',2)[0]
                disparity = self.city/'disparity'/row['split']/city/(stem+'_disparity.png')
                camera = self.city/'camera'/row['split']/city/(stem+'_camera.json')
                loaded = cityscapes_distance(disparity, camera, invalid_policy=self.invalid_policy,
                                             max_distance_m=self.max_distance_m)
                distance, depth_info = loaded.meters, loaded.metadata
        sample = self.generator.apply(image, condition, severity,
            seed=weather_seed(seed, epoch, f'{index}:effect'), distance_m=distance)
        if depth_info is not None:
            sample.metadata['distance'] = depth_info
        return sample
