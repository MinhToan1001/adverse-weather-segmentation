import math
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from src.data.preparation import TARGET_DOMAINS, resolve_data_path
from src.data.preprocessing import ImagePreprocessor
from src.data.validation import read_manifest
from src.data.augmentations import GammaDarkening, darken_rgb, Photometric
from src.data.augmentations.class_aware_crop import choose_class_crop


class ManifestSegmentationDataset(Dataset):
    """Read manifests whose masks already use trainIds 0..18 and ignore=255."""

    def __init__(
        self, manifest, image_size=(512, 1024), training=False, augment=None,
        native_labels=False, preprocessing=None, seed=0, allow_target_train=False, paired_style=None,
    ):
        self.items = read_manifest(manifest, allow_target_train=allow_target_train)
        self.preprocessor = ImagePreprocessor(image_size, preprocessing)
        self.mean, self.std = self.preprocessor.mean, self.preprocessor.std
        self.padding_rgb = self.preprocessor.padding_rgb
        allowed_domains = {"cityscapes", "synthetic"} | (TARGET_DOMAINS if allow_target_train else set())
        if training and any(row["domain"] not in allowed_domains or row["split"] != "train" for row in self.items):
            raise ValueError("Training requires Cityscapes train (or synthetic train for smoke)")
        self.size = self.preprocessor.size
        self.training = training
        self.source_style = None
        if paired_style is not None:
            from src.data.augmentations.source_style import SourceStyle
            if not training or any(r['domain'] != 'cityscapes' or r['split'] != 'train' for r in self.items):
                raise ValueError('Paired style requires Cityscapes train only')
            self.source_style = SourceStyle(paired_style)
        self.augment = augment or {}
        unknown = set(self.augment) - {"min_scale", "max_scale", "horizontal_flip", "gamma_darkening", "photometric", "class_aware_crop", "weather"}
        if unknown:
            raise ValueError(f"Unknown augmentation options: {sorted(unknown)}")
        self.min_scale = float(self.augment.get("min_scale", 0.5))
        self.max_scale = float(self.augment.get("max_scale", 2.0))
        self.flip_probability = float(self.augment.get("horizontal_flip", 0.5))
        if not all(math.isfinite(value) for value in (self.min_scale, self.max_scale, self.flip_probability)):
            raise ValueError("Augmentation values must be finite")
        if not 0 < self.min_scale <= self.max_scale or not 0 <= self.flip_probability <= 1:
            raise ValueError("Require 0 < min_scale <= max_scale and horizontal_flip in [0, 1]")
        self.gamma = (
            GammaDarkening.from_config(self.augment["gamma_darkening"])
            if "gamma_darkening" in self.augment else None
        )
        self.photo=Photometric(self.augment['photometric']) if 'photometric' in self.augment else None
        self.class_crop = self.augment.get('class_aware_crop')
        self.weather = None
        if 'weather' in self.augment:
            from src.data.augmentations.weather.policy import SourceWeather
            if not training or any(row['domain'] not in {'cityscapes', 'synthetic'} for row in self.items):
                raise ValueError('K1 weather is opt-in for source training only')
            if self.source_style or self.gamma or self.photo:
                raise ValueError('K1 condition labels cannot be mixed with other RGB style/gamma/photometric transforms')
            self.weather = SourceWeather(self.augment['weather'])
        self.seed = seed
        self.epoch = 0
        self.native_labels = native_labels
        if training and native_labels:
            raise ValueError("native_labels is for evaluation only")

    def __len__(self):
        return len(self.items)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _resize_and_fit(self, image, mask, gamma_exponent=None, photo_index=None, crop_info=None):
        target_h, target_w = self.size
        source_w, source_h = image.size
        if mask.size != image.size:
            raise ValueError(f"Image/mask size mismatch: {image.size} vs {mask.size}")

        if not self.training:
            top, left, height, width = self.preprocessor.fit_box(image.size)
            label_canvas = Image.new("L", (target_w, target_h), 255)
            label_canvas.paste(mask.resize((width, height), Image.Resampling.NEAREST), (left, top))
            return self.preprocessor.fit_image(image), label_canvas

        scale = min(target_w / source_w, target_h / source_h)
        scale *= random.uniform(self.min_scale, self.max_scale)
        resized_w = max(1, round(source_w * scale))
        resized_h = max(1, round(source_h * scale))
        image = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
        mask = mask.resize((resized_w, resized_h), Image.Resampling.NEAREST)

        crop_left = random.randint(0, max(0, resized_w - target_w))
        crop_top = random.randint(0, max(0, resized_h - target_h))
        if self.class_crop:
            chosen = choose_class_crop(mask, self.size, self.class_crop)
            if chosen is not None:
                crop_top, crop_left = chosen['top'], chosen['left']
                if crop_info is not None:
                    crop_info.update(chosen)
        crop_right = min(resized_w, crop_left + target_w)
        crop_bottom = min(resized_h, crop_top + target_h)
        image = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        mask = mask.crop((crop_left, crop_top, crop_right, crop_bottom))
        if gamma_exponent is not None:
            image = darken_rgb(image, gamma_exponent)
        if self.photo and photo_index is not None:
            image=self.photo.apply(image,self.seed,self.epoch,photo_index)

        canvas = Image.new("RGB", (target_w, target_h), self.padding_rgb)
        label_canvas = Image.new("L", (target_w, target_h), 255)
        left = random.randint(0, target_w - image.width)
        top = random.randint(0, target_h - image.height)
        canvas.paste(image, (left, top))
        label_canvas.paste(mask, (left, top))
        return canvas, label_canvas

    def __getitem__(self, idx):
        row = self.items[idx]
        weather_sample = None
        crop_info = {}
        gamma_exponent = None
        if self.training and self.gamma and self.gamma.probability and row["domain"] == "cityscapes":
            gamma_exponent = self.gamma.exponent_for(self.seed, self.epoch, idx % len(self.items))
        with Image.open(resolve_data_path(row["image"])) as source_image, Image.open(resolve_data_path(row["mask"])) as source_mask:
            if source_mask.mode not in {"L", "I", "I;16", "I;16L", "I;16B"}:
                raise ValueError(
                    f"Mask must be a single-channel integer trainId image, got mode {source_mask.mode}: {row['mask']}"
                )
            original = np.asarray(source_mask, dtype=np.int64).copy()
            if not np.isin(original, [*range(19), 255]).all():
                raise ValueError(f"Invalid trainId mask: {row['mask']}")
            source_w, source_h = source_image.size
            rgb = source_image.convert('RGB')
            if self.weather is not None:
                # The distance map is aligned to native RGB before any crop/flip.
                weather_sample = self.weather.apply(rgb, row, self.seed, self.epoch, idx)
                rgb = weather_sample.image
            image, mask = self._resize_and_fit(
                rgb, source_mask.convert("L"), gamma_exponent, idx if self.training else None, crop_info,
            )
        if self.training and random.random() < self.flip_probability:
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)

        labels = original if self.native_labels else np.asarray(mask, dtype=np.int64)
        valid = ((labels >= 0) & (labels < 19)) | (labels == 255)
        if not bool(valid.all()):
            raise ValueError(f"Invalid trainId mask: {row['mask']}")
        result = {
            "pixel_values": self.preprocessor.to_tensor(image),
            "labels": torch.from_numpy(labels.copy()).long(),
            "domain": row.get("domain", "unknown"),
        }
        if weather_sample is not None:
            result['condition_labels'] = torch.tensor(weather_sample.condition_id, dtype=torch.long)
            result['weather_severity'] = torch.tensor(weather_sample.severity, dtype=torch.long)
        if self.source_style is not None:
            # The clean PIL image already underwent the final crop/pad/flip.
            # RGB-only styling cannot change geometry or the shared mask.
            styled = self.source_style.apply(image, self.seed, self.epoch, idx)
            result['styled_pixel_values'] = self.preprocessor.to_tensor(styled)
        if self.training and self.class_crop:
            result['crop_class_id'] = crop_info.get('class_id', -1)
            result['crop_constraint_met'] = crop_info.get('met', False)
            result['crop_class_pixels'] = crop_info.get('pixels', 0)
        if self.native_labels:
            result["fit_box"] = self.preprocessor.fit_box((source_w, source_h))
        return result


def evaluation_collate(samples):
    """Keep original resolution masks as a list, including mixed ACDC sizes."""
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in samples]),
        "labels": [item["labels"] for item in samples],
        "domain": [item["domain"] for item in samples],
        "fit_box": [item["fit_box"] for item in samples],
    }
