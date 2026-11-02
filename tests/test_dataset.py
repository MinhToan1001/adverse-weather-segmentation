import csv
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ManifestSegmentationDataset, evaluation_collate
from src.data.validation import validate_manifests
from src.training.train import evaluate_model


def make_pair(root, domain="cityscapes", split="val"):
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[:, 20:] = 1
    image = np.repeat((mask * 255)[:, :, None], 3, axis=2)
    Image.fromarray(image).save(root / "image.png")
    Image.fromarray(mask).save(root / "mask.png")
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "mask", "domain", "split"])
        writer.writeheader()
        writer.writerow(dict(image=str(root / "image.png"), mask=str(root / "mask.png"), domain=domain, split=split))
    return manifest


def test_source_train_transforms_are_synchronized(tmp_path):
    manifest = make_pair(tmp_path, split="train")
    data = ManifestSegmentationDataset(manifest, (40, 40), True, dict(min_scale=1, max_scale=1, horizontal_flip=1))
    random.seed(42)
    item = data[0]
    valid = item["labels"] != 255
    assert torch.equal((item["pixel_values"][0][valid] > 0).long(), item["labels"][valid])
    assert (item["labels"] == 255).any()


def test_native_evaluation_and_padding_removal(tmp_path):
    manifest = make_pair(tmp_path)
    data = ManifestSegmentationDataset(manifest, (40, 40), native_labels=True)
    assert data[0]["labels"].shape == (20, 40)
    assert data[0]["fit_box"] == (10, 0, 20, 40)

    class Oracle(torch.nn.Module):
        def forward(self, pixel_values):
            red = pixel_values[:, :1]
            return SimpleNamespace(logits=torch.cat((-red, red), dim=1))

    metrics = evaluate_model(Oracle(), DataLoader(data, collate_fn=evaluation_collate), torch.device("cpu"), 2)
    assert metrics["miou"] == 1.0


def test_raw_mask_rejected_before_resizing(tmp_path):
    manifest = make_pair(tmp_path)
    Image.fromarray(np.full((20, 40), 256, dtype=np.uint16)).save(tmp_path / "mask.png")
    with pytest.raises(ValueError, match="Invalid trainId"):
        ManifestSegmentationDataset(manifest)[0]


def test_validation_checks_labels_and_cross_split_duplicates(tmp_path):
    manifest = make_pair(tmp_path)
    assert validate_manifests([manifest], check_labels=True) == 1
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifests([manifest, manifest])


def test_target_pixel_audit_blocked(tmp_path):
    manifest = make_pair(tmp_path, "fog")
    assert validate_manifests([manifest], check_files=True) == 1
    with pytest.raises(ValueError, match="unopened"):
        validate_manifests([manifest], check_labels=True)
