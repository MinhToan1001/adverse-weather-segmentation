"""Metric camera-ray distance from original Cityscapes disparity + calibration.

Official encoding: disparity_px=(stored-1)/256 for stored>0. Stereo depth Z
is converted to ray distance before atmospheric attenuation. Missing disparity
must be rejected or explicitly filled at max_distance_m (approximation).
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class MetricDistance:
    meters: np.ndarray
    metadata: dict


def validate_distance(distance, shape):
    values = np.asarray(distance, dtype=np.float32)
    if values.shape != tuple(shape) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Fog requires a finite nonnegative HxW camera-ray distance map in meters, aligned to RGB')
    return values


def cityscapes_distance(disparity, camera, *, invalid_policy='error', max_distance_m=200.):
    if invalid_policy not in {'error', 'far'} or not math.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError('Use invalid_policy error/far and a positive finite max_distance_m')
    if isinstance(disparity, (str, Path)):
        with Image.open(disparity) as image:
            stored = np.asarray(image).copy()
    else:
        stored = np.asarray(disparity)
    if (stored.ndim != 2 or not stored.size or stored.dtype.kind not in 'ui' or stored.dtype.itemsize < 2
            or stored.min() < 0 or stored.max() > 65535):
        raise ValueError('Expected original 16-bit Cityscapes disparity encoding')
    calibration = json.loads(Path(camera).read_text()) if isinstance(camera, (str, Path)) else camera
    intrinsic = calibration['intrinsic']
    fx, fy, u0, v0 = (float(intrinsic[name]) for name in ('fx', 'fy', 'u0', 'v0'))
    baseline = float(calibration['extrinsic']['baseline'])
    if not all(math.isfinite(v) for v in (fx, fy, u0, v0, baseline)) or min(fx, fy, baseline) <= 0:
        raise ValueError('Require valid focal lengths, principal point and baseline in meters')
    disparity_px = (stored.astype(np.float32) - 1) / 256.
    valid = (stored > 0) & (disparity_px > 0)
    if not valid.all() and invalid_policy == 'error':
        raise ValueError('Missing/nonpositive disparity: use completed metric distances or explicitly choose invalid_policy=far')
    depth_z = np.full(stored.shape, max_distance_m, dtype=np.float32)
    np.divide(baseline * fx, disparity_px, out=depth_z, where=valid)
    yy, xx = np.indices(stored.shape, dtype=np.float32)
    distance = depth_z * np.sqrt(1 + ((xx-u0)/fx)**2 + ((yy-v0)/fy)**2)
    distance = np.clip(distance, 0, max_distance_m).astype(np.float32)
    distance[~valid] = max_distance_m
    return MetricDistance(distance, dict(source='cityscapes_stereo', units='meters', kind='camera_ray_distance',
        invalid_policy=invalid_policy, invalid_fraction=float((~valid).mean()),
        max_distance_m=max_distance_m, baseline_m=baseline,
        approximation='No edge-aware stereo completion; far-fill is an explicit approximation'))
