import numpy as np
import pytest
from PIL import Image

from src.data.dataset import ManifestSegmentationDataset
from src.data.preparation import write_manifest
from src.data.validation import read_manifest, validate_manifests
from src.training.train import validate_target_pilot


def manifest(tmp_path, split="train", label=0):
    folder = tmp_path / "GOPR001"
    folder.mkdir(exist_ok=True)
    image, mask = folder / "a_rgb_anon.png", folder / "a_gt_labelTrainIds.png"
    Image.new("RGB", (128, 64)).save(image)
    Image.fromarray(np.full((64, 128), label, dtype=np.uint8)).save(mask)
    path = tmp_path / "rows.csv"
    write_manifest([dict(image=str(image), mask=str(mask), domain="rain", split=split)], path)
    return path


def test_default_protocol_still_rejects_target_training(tmp_path):
    path = manifest(tmp_path)
    with pytest.raises(ValueError, match="evaluation-only"):
        read_manifest(path)
    with pytest.raises(ValueError, match="evaluation-only"):
        ManifestSegmentationDataset(path, training=True)


def test_explicit_supervised_train_validates_labels(tmp_path):
    path = manifest(tmp_path)
    assert validate_manifests([path], check_labels=True, allow_target_train=True) == 1
    ds = ManifestSegmentationDataset(path, (64, 128), training=True,
                                    allow_target_train=True)
    assert ds[0]["labels"].shape == (64, 128)
    manifest(tmp_path, label=19)
    with pytest.raises(ValueError, match="Invalid dimensions or trainIds"):
        validate_manifests([path], check_labels=True, allow_target_train=True)


def test_opt_in_does_not_open_official_val_labels(tmp_path):
    path = manifest(tmp_path, split="val")
    with pytest.raises(ValueError, match="stay unopened"):
        validate_manifests([path], check_labels=True, allow_target_train=True)


def test_pilot_rejects_official_val_or_correlated_fit_dev():
    fit = [dict(image="rain/train/GOPR001/a.png", domain="rain", split="train")]
    dev = [dict(image="rain/train/GOPR002/b.png", domain="rain", split="train")]
    validate_target_pilot(fit, dev, 19)
    with pytest.raises(ValueError, match="groups overlap"):
        validate_target_pilot(fit, fit, 19)
    with pytest.raises(ValueError, match="official train"):
        validate_target_pilot(fit, [dict(dev[0], split="val")], 19)
    with pytest.raises(ValueError, match="19 Cityscapes"):
        validate_target_pilot(fit, dev, 18)


def test_chapters_of_same_recording_stay_together():
    from src.data.acdc import recording_group
    assert recording_group("fog/train/GOPR0478/a.png") == recording_group("fog/train/GP020478/b.png")
    fit = [dict(image="fog/train/GOPR0478/a.png", domain="fog", split="train")]
    dev = [dict(image="fog/train/GP020478/b.png", domain="fog", split="train")]
    with pytest.raises(ValueError, match="groups overlap"):
        validate_target_pilot(fit, dev, 19)


def test_group_split_uses_counts_without_tiny_dev_or_chapter_leakage():
    import random
    from src.data.partitions import choose_dev_groups
    counts = {"a": 240, "b": 154, "c": 6}
    assert choose_dev_groups(counts, random.Random(42)) == {"b"}
    counts = {"a": 45, "b": 64, "c": 164, "d": 91, "e": 29, "f": 7}
    chosen = choose_dev_groups(counts, random.Random(42))
    assert sum(counts[group] for group in chosen) == 81
