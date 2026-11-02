from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts.extract_dataset import extract_archive
from src.data.preparation import LABEL_IDS, convert_label_ids, create_manifest


def test_complete_cityscapes_mapping_and_void(tmp_path):
    original = np.array([list(range(34)) + [255]], dtype=np.uint8)
    source, target = tmp_path / "ids.png", tmp_path / "train.png"
    Image.fromarray(original).save(source)
    convert_label_ids(source, target)
    result = np.asarray(Image.open(target))[0]
    for raw in range(34):
        assert result[raw] == (LABEL_IDS.index(raw) if raw in LABEL_IDS else 255)
    assert result[34] == 255
    assert np.array_equal(np.asarray(Image.open(source)), original)


def test_conversion_refuses_rgb_and_in_place(tmp_path):
    source = tmp_path / "rgb.png"
    Image.new("RGB", (4, 4)).save(source)
    with pytest.raises(ValueError):
        convert_label_ids(source, tmp_path / "out.png")
    with pytest.raises(ValueError):
        convert_label_ids(source, source)


def test_cityscapes_pairs_require_converted_labels(tmp_path):
    image = tmp_path / "leftImg8bit/train/a/a_1_leftImg8bit.png"
    mask = tmp_path / "gtFine/train/a/a_1_gtFine_labelTrainIds.png"
    image.parent.mkdir(parents=True)
    image.touch()
    with pytest.raises(FileNotFoundError):
        create_manifest(tmp_path, "cityscapes", "train", allow_subset=True)
    mask.parent.mkdir(parents=True)
    mask.touch()
    rows = create_manifest(tmp_path, "cityscapes", "train", allow_subset=True)
    assert len(rows) == 1 and rows[0]["domain"] == "cityscapes"
    with pytest.raises(ValueError):
        create_manifest(tmp_path, "cityscapes", "train")


def test_acdc_manifest_uses_metadata_only(tmp_path):
    for domain in ("fog", "night", "rain", "snow"):
        for kind, suffix in (("rgb_anon", "rgb_anon"), ("gt", "gt_labelTrainIds")):
            path = tmp_path / kind / domain / "val/seq" / f"seq_frame_000001_{suffix}.png"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"not pixels: manifest must never decode this")
    rows = create_manifest(tmp_path, "acdc", "val", allow_subset=True)
    assert {row["domain"] for row in rows} == {"fog", "night", "rain", "snow"}
    with pytest.raises(ValueError):
        create_manifest(tmp_path, "acdc", "train")


def test_zip_selection_and_traversal(tmp_path):
    archive = tmp_path / "test.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("gtFine/train/city/mask.png", b"source")
        bundle.writestr("gtFine/test/city/mask.png", b"unused")
    destination = tmp_path / "raw"
    assert extract_archive(archive, destination, "cityscapes") == 1
    assert not (destination / "gtFine/test").exists()
    with pytest.raises(FileExistsError):
        extract_archive(archive, destination, "cityscapes")
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.png", b"bad")
    with pytest.raises(ValueError):
        extract_archive(archive, destination, "cityscapes")
