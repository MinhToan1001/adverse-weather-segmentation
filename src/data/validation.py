"""Shared CSV/path checks; target QA stays metadata-only before evaluation."""
import csv

import numpy as np
from PIL import Image

from src.data.preparation import FIELDS, TARGET_DOMAINS, resolve_data_path


def _check_pair_metadata(row, allow_target_train=False):
    image, mask = (resolve_data_path(row[field]) for field in ("image", "mask"))
    domain, split = row["domain"], row["split"]
    if domain not in {"cityscapes", "synthetic", *TARGET_DOMAINS}:
        raise ValueError(f"Unknown domain: {domain}")
    if split not in {"train", "val"}:
        raise ValueError(f"Unsupported split: {split}")
    if domain in TARGET_DOMAINS and split != "val" and not allow_target_train:
        raise ValueError("ACDC is evaluation-only")
    if image == mask:
        raise ValueError("Image and mask must be different files")
    if mask.name.endswith("_labelIds.png"):
        raise ValueError("Raw labelIds are not trainIds; convert Cityscapes labels first")
    if image.name.endswith("_leftImg8bit.png"):
        expected = image.name.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png")
        if domain != "cityscapes" or mask.name != expected or mask.parent.name != image.parent.name:
            raise ValueError(f"Cityscapes image/mask identity mismatch: {image} / {mask}")
    if image.name.endswith("_rgb_anon.png"):
        expected = image.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
        if domain not in TARGET_DOMAINS or mask.name != expected:
            raise ValueError(f"ACDC image/mask identity mismatch: {image} / {mask}")
    # Verify declared split/domain against canonical directory components, if present.
    for path in (image, mask):
        parts = path.parts
        for index, part in enumerate(parts):
            if part in {"leftImg8bit", "gtFine"} and index + 1 < len(parts):
                if parts[index + 1] != split or domain != "cityscapes":
                    raise ValueError(f"Path disagrees with declared domain/split: {path}")
            if part in {"rgb_anon", "gt"} and index + 2 < len(parts) and parts[index + 1] in TARGET_DOMAINS:
                if parts[index + 1] != domain or parts[index + 2] != split:
                    raise ValueError(f"Path disagrees with declared domain/split: {path}")
    return image, mask


def read_manifest(manifest, allow_target_train=False):
    """Read BOM-tolerant CSV and fail on malformed or contradictory metadata."""
    with open(manifest, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        if not set(FIELDS).issubset(names) or len(names) != len(set(names)):
            raise ValueError(f"Missing or duplicate CSV columns: {manifest}")
        rows = []
        seen = set()
        for number, row in enumerate(reader, start=2):
            if None in row or not all(isinstance(row.get(field), str) and row[field].strip() for field in FIELDS):
                raise ValueError(f"Empty field or malformed CSV: {manifest}, row {number}")
            row = {field: row[field].strip() for field in FIELDS}
            paths = _check_pair_metadata(row, allow_target_train=allow_target_train)
            if any(path in seen for path in paths):
                raise ValueError(f"Duplicate image or mask: {manifest}, row {number}")
            seen.update(paths)
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {manifest}")
    return rows


def validate_manifests(manifests, check_files=False, check_labels=False, allow_target_train=False):
    seen = set()
    rows = [row for manifest in manifests for row in read_manifest(manifest, allow_target_train=allow_target_train)]
    if not rows:
        raise ValueError("At least one nonempty manifest is required")
    if check_labels and any(
        row["domain"] in TARGET_DOMAINS and not (allow_target_train and row["split"] == "train")
        for row in rows
    ):
        raise ValueError("ACDC labels stay unopened until the frozen evaluation pass")
    for row in rows:
        paths = [resolve_data_path(row[field]) for field in ("image", "mask")]
        if any(path in seen for path in paths):
            raise ValueError(f"Duplicate image or mask within/across manifests: {paths}")
        seen.update(paths)
        if check_files or check_labels:
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
        if check_labels:
            with Image.open(paths[0]) as image, Image.open(paths[1]) as mask:
                image.load()
                labels = np.asarray(mask)
                if mask.mode not in {"L", "I", "I;16", "I;16L", "I;16B"} or labels.ndim != 2:
                    raise ValueError(f"Not an integer trainId mask: {paths[1]}")
                if image.size != mask.size or not np.isin(labels, [*range(19), 255]).all():
                    raise ValueError(f"Invalid dimensions or trainIds: {paths[1]}")
    return len(rows)
