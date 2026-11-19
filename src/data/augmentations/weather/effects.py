"""Parameterized weather effects. Fog is depth-based; other effects approximate appearance."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .color import rgb_array, srgb_to_linear, linear_to_srgb, to_image
from .depth import validate_distance


def atmospheric_fog(image, distance_m, beta, airlight=(.9, .9, .9)):
    """I=J exp(-beta*d)+A(1-exp(-beta*d)); beta in m^-1, A in linear RGB."""
    rgb = rgb_array(image)
    distance = validate_distance(distance_m, rgb.shape[:2])
    airlight = np.asarray(airlight, dtype=np.float32)
    if not math.isfinite(beta) or beta < 0 or airlight.shape != (3,) or not np.isfinite(airlight).all() or ((airlight < 0) | (airlight > 1)).any():
        raise ValueError('Require beta>=0 and finite linear RGB airlight in [0,1]')
    transmission = np.exp(-float(beta) * distance)[..., None]
    values = srgb_to_linear(rgb) * transmission + airlight * (1 - transmission)
    return to_image(linear_to_srgb(values))


def sensor_night(image, rng, *, exposure, gamma=1., full_well=4096., read_noise=.002,
                 saturation=.8, glare=0.):
    """Approximate low light on linearized sRGB with Poisson shot + Gaussian read noise.

    This is not RAW unprocessing: demosaicing, camera response and real lamps
    are unknown. Optional bloom uses existing bright pixels, not invented lights.
    """
    parameters = (exposure, gamma, full_well, read_noise, saturation, glare)
    if (not all(math.isfinite(p) for p in parameters) or not 0 < exposure <= 1 or gamma <= 0
            or full_well <= 0 or read_noise < 0 or not 0 <= saturation <= 1 or not 0 <= glare <= 1):
        raise ValueError('Invalid low-light parameters')
    linear = srgb_to_linear(rgb_array(image))
    luminance = (linear * np.array([.2126, .7152, .0722], np.float32)).sum(2, keepdims=True)
    signal = exposure * (saturation * linear + (1 - saturation) * luminance) ** gamma
    values = rng.poisson(signal * full_well).astype(np.float32) / full_well
    values += rng.normal(0, read_noise, signal.shape).astype(np.float32)
    if glare:
        highlights = np.clip((luminance[..., 0] - .85) / .15, 0, 1)
        glow = Image.fromarray(np.rint(highlights * 255).astype(np.uint8))
        glow = np.asarray(glow.filter(ImageFilter.GaussianBlur(max(1., min(image.size)/80))), np.float32) / 255.
        values += glare * glow[..., None]
    return to_image(linear_to_srgb(values))


def streak_rain(image, rng, *, density_per_mp, length, opacity, angle=-12., haze=.0):
    if (not all(math.isfinite(p) for p in (density_per_mp, length, opacity, angle, haze))
            or not 0 <= density_per_mp <= 50000 or length <= 0
            or not 0 <= opacity <= 1 or not 0 <= haze <= 1):
        raise ValueError('Invalid rain parameters')
    width, height = image.size
    scale = min(width, height) / 512.
    layer = Image.new('L', image.size)
    draw = ImageDraw.Draw(layer)
    count = round(density_per_mp * width * height / 1e6)
    dy = max(1, length * scale)
    dx = math.tan(math.radians(angle)) * dy
    for _ in range(count):
        x, y = float(rng.uniform(0, width)), float(rng.uniform(-dy, height))
        draw.line((x, y, x+dx, y+dy), fill=int(rng.integers(100, 256)), width=max(1, round(scale)))
    alpha = np.asarray(layer.filter(ImageFilter.GaussianBlur(max(.25, scale*.5))), dtype=np.float32)[..., None] / 255. * opacity
    background = srgb_to_linear(rgb_array(image)) * (1-haze) + .8*haze
    return to_image(linear_to_srgb(background * (1-alpha) + .8*alpha))


def layered_snow(image, rng, *, density_per_mp, radius, opacity, haze=.0):
    if (not all(math.isfinite(p) for p in (density_per_mp, radius, opacity, haze))
            or not 0 <= density_per_mp <= 50000 or radius <= 0
            or not 0 <= opacity <= 1 or not 0 <= haze <= 1):
        raise ValueError('Invalid snow parameters')
    width, height = image.size
    scale = min(width, height)/512.
    background = srgb_to_linear(rgb_array(image)) * (1-haze) + .9*haze
    # Three appearance layers approximate far/small and near/large particles.
    for size_multiplier, fraction, blur in ((.5, .6, .25), (1., .3, .5), (2., .1, 1.2)):
        layer = Image.new('L', image.size)
        draw = ImageDraw.Draw(layer)
        count = round(density_per_mp * fraction * width * height / 1e6)
        for _ in range(count):
            x, y = float(rng.uniform(0, width)), float(rng.uniform(0, height))
            r = max(.5, radius * scale * size_multiplier * rng.uniform(.6, 1.4))
            draw.ellipse((x-r, y-r, x+r, y+r), fill=int(rng.integers(140, 256)))
        alpha = np.asarray(layer.filter(ImageFilter.GaussianBlur(blur*max(.5, scale))), np.float32)[..., None]/255.*opacity
        background = background*(1-alpha) + alpha
    return to_image(linear_to_srgb(background))
