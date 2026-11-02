import math
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from src.data.preparation import resolve_data_path
from src.data.validation import read_manifest


_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
_MEAN_RGB = tuple(int(round(value * 255)) for value in _MEAN)


class ManifestSegmentationDataset(Dataset):
    """Read manifests whose masks already use trainIds 0..18 and ignore=255."""

    def __init__(self, manifest, image_size=(512, 1024), training=False, augment=None, native_labels=False, preprocessing=None):
        self.items = read_manifest(manifest)
        prep = preprocessing or {}
        fixed = {"keep_aspect_ratio": True, "image_interpolation": "bilinear", "mask_interpolation": "nearest", "mask_padding": 255}
        if set(prep) - {*fixed, "mean", "std"}:
            raise ValueError("Unknown preprocessing option")
        if any(prep.get(key, value) != value for key, value in fixed.items()):
            raise ValueError("Preprocessing must preserve the frozen geometry/label contract")
        self.mean = np.asarray(prep.get("mean", _MEAN), dtype=np.float32)
        self.std = np.asarray(prep.get("std", _STD), dtype=np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any() or ((self.mean < 0) | (self.mean > 1)).any():
            raise ValueError("Require three finite mean values in [0,1] and positive std values")
        self.padding_rgb = tuple(int(round(float(value) * 255)) for value in self.mean)
        if training and any(row["domain"] not in {"cityscapes", "synthetic"} or row["split"] != "train" for row in self.items):
            raise ValueError("Training requires Cityscapes train (or synthetic train for smoke)")
        self.size = tuple(int(value) for value in image_size)
        if len(self.size) != 2 or min(self.size) <= 0:
            raise ValueError("image_size must be [height, width] with positive values")
        self.training = training
        self.augment = augment or {}
        unknown = set(self.augment) - {"min_scale", "max_scale", "horizontal_flip"}
        if unknown:
            raise ValueError(f"Unknown augmentation options: {sorted(unknown)}")
        self.min_scale = float(self.augment.get("min_scale", 0.5))
        self.max_scale = float(self.augment.get("max_scale", 2.0))
        self.flip_probability = float(self.augment.get("horizontal_flip", 0.5))
        if not all(math.isfinite(value) for value in (self.min_scale, self.max_scale, self.flip_probability)):
            raise ValueError("Augmentation values must be finite")
        if not 0 < self.min_scale <= self.max_scale or not 0 <= self.flip_probability <= 1:
            raise ValueError("Require 0 < min_scale <= max_scale and horizontal_flip in [0, 1]")
        self.native_labels = native_labels
        if training and native_labels:
            raise ValueError("native_labels is for evaluation only")

    def __len__(self):
        return len(self.items)

    def _resize_and_fit(self, image, mask):
        target_h, target_w = self.size
        source_w, source_h = image.size
        if mask.size != image.size:
            raise ValueError(f"Image/mask size mismatch: {image.size} vs {mask.size}")

        scale = min(target_w / source_w, target_h / source_h)
        if self.training:
            scale *= random.uniform(self.min_scale, self.max_scale)
        resized_w = max(1, round(source_w * scale))
        resized_h = max(1, round(source_h * scale))
        image = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
        mask = mask.resize((resized_w, resized_h), Image.Resampling.NEAREST)

        crop_left = random.randint(0, max(0, resized_w - target_w)) if self.training else 0
        crop_top = random.randint(0, max(0, resized_h - target_h)) if self.training else 0
        crop_right = min(resized_w, crop_left + target_w)
        crop_bottom = min(resized_h, crop_top + target_h)
        image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        mask = mask.crop((crop_left, crop_top, crop_right, crop_bottom))

        canvas = Image.new("RGB", (target_w, target_h), self.padding_rgb)
        label_canvas = Image.new("L", (target_w, target_h), 255)
        if self.training:
            left = random.randint(0, target_w - image.width)
            top = random.randint(0, target_h - image.height)
        else:
            left = (target_w - image.width) // 2
            top = (target_h - image.height) // 2
        canvas.paste(image, (left, top))
        label_canvas.paste(mask, (left, top))
        return canvas, label_canvas

    def __getitem__(self, idx):
        row = self.items[idx]
        with Image.open(resolve_data_path(row["image"])) as source_image, Image.open(resolve_data_path(row["mask"])) as source_mask:
            if source_mask.mode not in {"L", "I", "I;16", "I;16L", "I;16B"}:
                raise ValueError(
                    f"Mask must be a single-channel integer trainId image, got mode {source_mask.mode}: {row['mask']}"
                )
            original = np.asarray(source_mask, dtype=np.int64).copy()
            if not np.isin(original, [*range(19), 255]).all():
                raise ValueError(f"Invalid trainId mask: {row['mask']}")
            source_w, source_h = source_image.size
            image, mask = self._resize_and_fit(source_image.convert("RGB"), source_mask.convert("L"))
        if self.training and random.random() < self.flip_probability:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - self.mean) / self.std
        labels = original if self.native_labels else np.asarray(mask, dtype=np.int64)
        valid = ((labels >= 0) & (labels < 19)) | (labels == 255)
        if not bool(valid.all()):
            raise ValueError(f"Invalid trainId mask: {row['mask']}")
        result = {
            "pixel_values": torch.from_numpy(pixels.transpose(2, 0, 1).copy()).float(),
            "labels": torch.from_numpy(labels.copy()).long(),
            "domain": row.get("domain", "unknown"),
        }
        if self.native_labels:
            height, width = self.size
            scale = min(width / source_w, height / source_h)
            fitted_w, fitted_h = max(1, round(source_w * scale)), max(1, round(source_h * scale))
            result["fit_box"] = ((height - fitted_h) // 2, (width - fitted_w) // 2, fitted_h, fitted_w)
        return result


def evaluation_collate(samples):
    """Keep original resolution masks as a list, including mixed ACDC sizes."""
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in samples]),
        "labels": [item["labels"] for item in samples],
        "domain": [item["domain"] for item in samples],
        "fit_box": [item["fit_box"] for item in samples],
    }
