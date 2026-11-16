"""Frozen aspect-preserving image preprocessing shared by datasets and inference."""
import numpy as np
from PIL import Image
import torch


class ImagePreprocessor:
    def __init__(self, image_size=(512, 1024), preprocessing=None):
        prep = preprocessing or {}
        fixed = {
            "keep_aspect_ratio": True, "image_interpolation": "bilinear",
            "mask_interpolation": "nearest", "mask_padding": 255,
        }
        if set(prep) - {*fixed, "mean", "std"}:
            raise ValueError("Unknown preprocessing option")
        if any(prep.get(key, value) != value for key, value in fixed.items()):
            raise ValueError("Preprocessing must preserve the frozen geometry/label contract")
        self.mean = np.asarray(prep.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.asarray(prep.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        if (
            self.mean.shape != (3,) or self.std.shape != (3,)
            or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
            or (self.std <= 0).any() or ((self.mean < 0) | (self.mean > 1)).any()
        ):
            raise ValueError("Require three finite mean values in [0,1] and positive std values")
        self.padding_rgb = tuple(int(round(float(value) * 255)) for value in self.mean)
        self.size = tuple(int(value) for value in image_size)
        if len(self.size) != 2 or min(self.size) <= 0:
            raise ValueError("image_size must be [height, width] with positive values")

    def fit_box(self, source_size):
        source_w, source_h = source_size
        height, width = self.size
        scale = min(width / source_w, height / source_h)
        fitted_w, fitted_h = max(1, round(source_w * scale)), max(1, round(source_h * scale))
        return ((height - fitted_h) // 2, (width - fitted_w) // 2, fitted_h, fitted_w)

    def fit_image(self, image):
        top, left, height, width = self.fit_box(image.size)
        canvas = Image.new("RGB", self.size[::-1], self.padding_rgb)
        canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), (left, top))
        return canvas

    def to_tensor(self, image):
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - self.mean) / self.std
        return torch.from_numpy(pixels.transpose(2, 0, 1).copy()).float()
