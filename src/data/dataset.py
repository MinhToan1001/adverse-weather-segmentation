import csv
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
_MEAN_RGB = tuple(int(round(value * 255)) for value in _MEAN)


class ManifestSegmentationDataset(Dataset):
    """Read manifests whose masks already use trainIds 0..18 and ignore=255."""

    def __init__(self, manifest, image_size=(512, 1024), training=False, augment=None):
        with open(manifest, newline="", encoding="utf-8") as f:
            self.items = list(csv.DictReader(f))
        if not self.items:
            raise ValueError(f"Empty manifest: {manifest}")
        required = {"image", "mask", "domain", "split"}
        if not required.issubset(self.items[0]):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")
        self.size = tuple(int(value) for value in image_size)
        if len(self.size) != 2 or min(self.size) <= 0:
            raise ValueError("image_size must be [height, width] with positive values")
        self.training = training
        self.augment = augment or {}

    def __len__(self):
        return len(self.items)

    def _resize_and_fit(self, image, mask):
        target_h, target_w = self.size
        source_w, source_h = image.size
        if mask.size != image.size:
            raise ValueError(f"Image/mask size mismatch: {image.size} vs {mask.size}")

        scale = min(target_w / source_w, target_h / source_h)
        if self.training:
            scale *= random.uniform(
                float(self.augment.get("min_scale", 0.5)),
                float(self.augment.get("max_scale", 2.0)),
            )
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

        canvas = Image.new("RGB", (target_w, target_h), _MEAN_RGB)
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
        with Image.open(row["image"]) as source_image, Image.open(row["mask"]) as source_mask:
            if source_mask.mode not in {"L", "I", "I;16", "I;16L", "I;16B"}:
                raise ValueError(
                    f"Mask must be a single-channel integer trainId image, got mode {source_mask.mode}: {row['mask']}"
                )
            image, mask = self._resize_and_fit(source_image.convert("RGB"), source_mask.convert("L"))
        if self.training and random.random() < float(self.augment.get("horizontal_flip", 0.5)):
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - _MEAN) / _STD
        labels = np.asarray(mask, dtype=np.int64)
        valid = ((labels >= 0) & (labels < 19)) | (labels == 255)
        if not bool(valid.all()):
            raise ValueError(f"Invalid trainId mask: {row['mask']}")
        return {
            "pixel_values": torch.from_numpy(pixels.transpose(2, 0, 1).copy()).float(),
            "labels": torch.from_numpy(labels.copy()).long(),
            "domain": row.get("domain", "unknown"),
        }
