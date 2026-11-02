"""Task03 regression checks using only temporary, synthetic files."""
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from src.data.dataset import ManifestSegmentationDataset, evaluation_collate
from src.data.preparation import REPO_ROOT, create_manifest, write_manifest
from src.data.validation import read_manifest, validate_manifests


def pair(root, split="val", domain="synthetic", height=20, width=40):
    root.mkdir(parents=True, exist_ok=True)
    labels = np.zeros((height, width), dtype=np.uint8)
    labels[:, width // 2:] = 1
    Image.fromarray(np.repeat((labels * 255)[..., None], 3, axis=2)).save(root / "image.png")
    Image.fromarray(labels).save(root / "mask.png")
    return dict(image=str(root / "image.png"), mask=str(root / "mask.png"), domain=domain, split=split)


def manifest(root, rows):
    path = root / "manifest.csv"
    write_manifest(rows, path)
    return path


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("flip", [0.0, 1.0])
def test_scale_crop_pad_and_flip_keep_image_and_mask_aligned(tmp_path, scale, flip):
    path = manifest(tmp_path, [pair(tmp_path, split="train")])
    data = ManifestSegmentationDataset(path, (40, 40), training=True,
        augment=dict(min_scale=scale, max_scale=scale, horizontal_flip=flip))
    random.seed(42)
    sample = data[0]
    valid = sample["labels"] != 255
    assert sample["pixel_values"].shape == (3, 40, 40)
    assert sample["pixel_values"].dtype == torch.float32
    assert sample["labels"].dtype == torch.int64
    assert valid.any()
    assert torch.equal((sample["pixel_values"][0][valid] > 0).long(), sample["labels"][valid])
    if scale < 2:
        assert (~valid).any()


def test_train_augmentation_repeats_with_same_seed(tmp_path):
    path = manifest(tmp_path, [pair(tmp_path, split="train")])
    data = ManifestSegmentationDataset(path, (32, 32), training=True)
    random.seed(17)
    first = data[0]
    random.seed(17)
    second = data[0]
    assert torch.equal(first["pixel_values"], second["pixel_values"])
    assert torch.equal(first["labels"], second["labels"])


def test_eval_preserves_aspect_ratio_normalizes_and_uses_no_rng(tmp_path):
    path = manifest(tmp_path, [pair(tmp_path)])
    data = ManifestSegmentationDataset(path, (40, 40), native_labels=True)
    before = random.getstate()
    sample = data[0]
    assert random.getstate() == before
    assert sample["fit_box"] == (10, 0, 20, 40)
    assert sample["labels"].shape == (20, 40)
    expected_black = -torch.tensor([0.485, 0.456, 0.406]) / torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(sample["pixel_values"][:, 10, 0], expected_black)
    # Mean-colour RGB padding is within rounding error of normalized zero.
    assert sample["pixel_values"][:, :10].abs().max() < 0.01
    assert torch.equal(sample["pixel_values"], data[0]["pixel_values"])


def test_mask_nearest_resize_preserves_ids_zero_18_and_void(tmp_path):
    row = pair(tmp_path, height=3, width=3)
    labels = np.tile(np.array([0, 18, 255], dtype=np.uint8), (3, 1))
    Image.fromarray(labels).save(row["mask"])
    data = ManifestSegmentationDataset(manifest(tmp_path, [row]), (9, 9))
    assert set(data[0]["labels"].unique().tolist()) == {0, 18, 255}
    expected = torch.tensor(labels.astype(np.int64)).repeat_interleave(3, 0).repeat_interleave(3, 1)
    assert torch.equal(data[0]["labels"], expected)


def test_mixed_native_sizes_collate_into_one_image_batch(tmp_path):
    rows = [pair(tmp_path / "a"), pair(tmp_path / "b", height=30, width=10)]
    data = ManifestSegmentationDataset(manifest(tmp_path, rows), (40, 40), native_labels=True)
    batch = evaluation_collate([data[0], data[1]])
    assert batch["pixel_values"].shape == (2, 3, 40, 40)
    assert [tuple(mask.shape) for mask in batch["labels"]] == [(20, 40), (30, 10)]
    assert batch["fit_box"] == [(10, 0, 20, 40), (0, 13, 40, 13)]


@pytest.mark.parametrize("augment", [
    {"min_scale": 0}, {"min_scale": 3, "max_scale": 2},
    {"horizontal_flip": -0.1}, {"horizontal_flip": 1.1},
    {"max_scale": float("inf")}, {"min_scale": float("nan")}, {"flip_typo": 1},
])
def test_invalid_augmentation_fails_at_construction(tmp_path, augment):
    path = manifest(tmp_path, [pair(tmp_path, split="train")])
    with pytest.raises(ValueError):
        ManifestSegmentationDataset(path, training=True, augment=augment)


@pytest.mark.parametrize("domain,split", [("cityscapes", "val"), ("fog", "val")])
def test_training_rejects_validation_and_target_rows(tmp_path, domain, split):
    path = manifest(tmp_path, [pair(tmp_path, domain=domain, split=split)])
    with pytest.raises(ValueError, match="Training requires"):
        ManifestSegmentationDataset(path, training=True)


def test_relative_csv_paths_use_repo_root_from_another_cwd(tmp_path, monkeypatch):
    # A synthetic repository root avoids writing fixtures into the real data tree.
    import src.data.preparation as preparation
    monkeypatch.setattr(preparation, "REPO_ROOT", tmp_path)
    row = pair(tmp_path / "fixtures")
    row.update(image="fixtures/image.png", mask="fixtures/mask.png")
    path = manifest(tmp_path, [row])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert validate_manifests([path], check_labels=True) == 1
    assert ManifestSegmentationDataset(path, (20, 40))[0]["labels"].shape == (20, 40)


@pytest.mark.parametrize("mode", ["RGB", "P"])
def test_color_and_palette_masks_rejected_by_validator_and_loader(tmp_path, mode):
    row = pair(tmp_path)
    Image.new(mode, (40, 20)).save(row["mask"])
    path = manifest(tmp_path, [row])
    with pytest.raises(ValueError, match="integer trainId"):
        validate_manifests([path], check_labels=True)
    with pytest.raises(ValueError, match="single-channel"):
        ManifestSegmentationDataset(path)[0]


@pytest.mark.parametrize("shape,value", [((20, 40), 19), ((20, 40), 256), ((10, 10), 0)])
def test_invalid_original_labels_or_dimensions_rejected(tmp_path, shape, value):
    row = pair(tmp_path)
    Image.fromarray(np.full(shape, value, dtype=np.uint16)).save(row["mask"])
    path = manifest(tmp_path, [row])
    with pytest.raises(ValueError):
        validate_manifests([path], check_labels=True)
    with pytest.raises(ValueError):
        ManifestSegmentationDataset(path)[0]


def test_duplicate_mask_is_rejected_even_with_different_images(tmp_path):
    first, second = pair(tmp_path / "a"), pair(tmp_path / "b")
    second["mask"] = first["mask"]
    path = manifest(tmp_path, [first, second])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifests([path])
    with pytest.raises(ValueError, match="Duplicate"):
        ManifestSegmentationDataset(path)


def test_cross_manifest_mask_reuse_and_missing_files_fail(tmp_path):
    first, second = pair(tmp_path / "a", split="train"), pair(tmp_path / "b")
    first_csv = manifest(tmp_path / "a", [first])
    second["mask"] = first["mask"]
    second_csv = manifest(tmp_path / "b", [second])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifests([first_csv, second_csv], check_files=True)
    Path(first["mask"]).unlink()
    with pytest.raises(FileNotFoundError):
        validate_manifests([first_csv], check_files=True)


def test_corrupt_source_image_fails_pixel_validation(tmp_path):
    row = pair(tmp_path)
    Path(row["image"]).write_bytes(b"not an image")
    with pytest.raises(OSError):
        validate_manifests([manifest(tmp_path, [row])], check_labels=True)


@pytest.mark.parametrize("text", [
    "image,mask,domain\na.png,b.png,synthetic\n",
    "image,mask,domain,split\na.png,,synthetic,val\n",
    "image,mask,domain,split\na.png,b.png,synthetic,val,extra\n",
    "image,mask,domain,split\n",
    "image,mask,domain,split\na.png,b.png,unknown,val\n",
    "image,mask,domain,split\na.png,b.png,fog,train\n",
    "image,mask,domain,split\na.png,b.png,cityscapes,test\n",
])
def test_malformed_csv_and_unsupported_roles_fail(tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        read_manifest(path)


def test_utf8_bom_is_accepted(tmp_path):
    row = pair(tmp_path)
    path = manifest(tmp_path, [row])
    path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8-sig")
    assert validate_manifests([path], check_labels=True) == 1
    assert len(ManifestSegmentationDataset(path)) == 1


def city_pair(root, split="train", name="a_000001_000019", mask_root=None):
    image = root / "leftImg8bit" / split / "a" / f"{name}_leftImg8bit.png"
    mask = (mask_root or root) / "gtFine" / split / "a" / f"{name}_gtFine_labelTrainIds.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    mask.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 4)).save(image)
    Image.new("L", (8, 4), 0).save(mask)
    return image, mask


def test_manifest_sorted_separate_mask_root_and_byte_reproducible(tmp_path):
    root, masks = tmp_path / "raw", tmp_path / "processed"
    city_pair(root, name="a_000002_000019", mask_root=masks)
    city_pair(root, name="a_000001_000019", mask_root=masks)
    rows = create_manifest(root, "cityscapes", "train", masks, allow_subset=True)
    assert rows[0]["image"] < rows[1]["image"]
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    write_manifest(rows, first)
    write_manifest(create_manifest(root, "cityscapes", "train", masks, True), second)
    assert first.read_bytes() == second.read_bytes()
    assert validate_manifests([first], check_labels=True) == 2


def test_orphan_mask_fails_instead_of_silent_omission(tmp_path):
    city_pair(tmp_path)
    orphan = tmp_path / "gtFine/train/a/orphan_gtFine_labelTrainIds.png"
    orphan.touch()
    with pytest.raises(ValueError, match="without a matching image"):
        create_manifest(tmp_path, "cityscapes", "train", allow_subset=True)


def test_canonical_pair_identity_and_split_are_validated(tmp_path):
    image, mask = city_pair(tmp_path)
    row = dict(image=str(image), mask=str(mask), domain="cityscapes", split="val")
    with pytest.raises(ValueError, match="declared domain/split"):
        validate_manifests([manifest(tmp_path, [row])])
    row.update(split="train", mask=str(mask.with_name("different_gtFine_labelTrainIds.png")))
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_manifests([manifest(tmp_path, [row])])
    row["mask"] = str(mask.with_name("a_gtFine_labelIds.png"))
    with pytest.raises(ValueError, match="Raw labelIds"):
        validate_manifests([manifest(tmp_path, [row])])


def test_acdc_builder_and_file_validator_never_decode_pixels(tmp_path, monkeypatch):
    for domain in ("fog", "night", "rain", "snow"):
        for kind, suffix in (("rgb_anon", "rgb_anon"), ("gt", "gt_labelTrainIds")):
            path = tmp_path / kind / domain / "val/seq" / f"frame_{suffix}.png"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"metadata fixture, not an actual PNG")
    def forbidden(*args, **kwargs):
        pytest.fail("ACDC metadata checks must not open image/label pixels")
    monkeypatch.setattr(Image, "open", forbidden)
    rows = create_manifest(tmp_path, "acdc", "val", allow_subset=True)
    path = manifest(tmp_path, rows)
    assert validate_manifests([path], check_files=True) == 4
    with pytest.raises(ValueError, match="unopened"):
        validate_manifests([path], check_labels=True)
    with pytest.raises(ValueError, match="expected 100"):
        create_manifest(tmp_path, "acdc", "val")


def run_cli(script, *args):
    return subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / script), *map(str, args)],
        cwd=REPO_ROOT, capture_output=True, text=True)


def test_manifest_and_validator_cli_end_to_end(tmp_path):
    city_pair(tmp_path)
    output = tmp_path / "manifests"
    args = ("--root", tmp_path, "--dataset", "cityscapes", "--split", "train", "--allow-subset", "--output-dir", output)
    created = run_cli("make_manifest.py", *args)
    assert created.returncode == 0, created.stderr
    csv_path = output / "cityscapes_train.csv"
    valid = run_cli("validate_manifest.py", "--manifest", csv_path, "--check-files", "--check-labels")
    assert valid.returncode == 0, valid.stderr
    assert "rows=1" in valid.stdout
    assert run_cli("make_manifest.py", *args).returncode != 0
    assert run_cli("make_manifest.py", *args, "--overwrite").returncode == 0
    row = read_manifest(csv_path)[0]
    Path(row["mask"]).unlink()
    assert run_cli("validate_manifest.py", "--manifest", csv_path, "--check-files").returncode != 0


def test_cli_preflights_both_splits_before_writing(tmp_path):
    city_pair(tmp_path)  # val is intentionally missing
    output = tmp_path / "manifests"
    result = run_cli("make_manifest.py", "--root", tmp_path, "--dataset", "cityscapes", "--allow-subset", "--output-dir", output)
    assert result.returncode != 0
    assert not (output / "cityscapes_train.csv").exists()
