"""sRGB transfer functions; image formation operations use linear RGB."""
import numpy as np
from PIL import Image


def rgb_array(image):
    if not isinstance(image, Image.Image) or image.mode != 'RGB':
        raise ValueError('Weather input must be a PIL RGB image')
    return np.asarray(image, dtype=np.float32) / 255.


def srgb_to_linear(values):
    values = np.asarray(values, dtype=np.float32)
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def linear_to_srgb(values):
    values = np.clip(np.asarray(values, dtype=np.float32), 0, 1)
    return np.where(values <= .0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - .055)


def to_image(values):
    return Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8))
