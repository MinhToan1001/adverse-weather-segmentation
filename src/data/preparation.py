"""Source label conversion and deterministic, metadata-only manifest creation."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

# Official Cityscapes labels.py trainId mapping. All other raw IDs are void.
LABEL_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
FIELDS = ("image", "mask", "domain", "split")
REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_DOMAINS = frozenset({"fog", "night", "rain", "snow"})


def resolve_data_path(value):
    """CSV data paths are relative to the repository, never to the caller CWD."""
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def convert_label_ids(source, destination):
    """Convert raw Cityscapes IDs; never infer encoding from pixel values."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("Output must differ from the original labelIds file")
    with Image.open(source) as image:
        values = np.asarray(image)
        if values.ndim != 2 or image.mode == "P":
            raise ValueError(f"Expected integer labelIds mask: {source}")
        if image.mode not in {"L", "I", "I;16", "I;16L", "I;16B"} or not (((values >= 0) & (values <= 33)) | (values == 255) | (values == -1)).all():
            raise ValueError(f"Unexpected Cityscapes label ID: {source}")
        lookup = np.full(256, 255, dtype=np.uint8)
        for train_id, label_id in enumerate(LABEL_IDS):
            lookup[label_id] = train_id
        result = lookup[values.astype(np.uint8)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(destination)


def path_string(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def create_manifest(root, dataset, split, mask_root=None, allow_subset=False):
    root = Path(root).resolve()
    masks = Path(mask_root).resolve() if mask_root else root
    if dataset == "cityscapes":
        if split not in {"train", "val"}:
            raise ValueError("Cityscapes test is outside this protocol")
        images = sorted(path for path in (root / "leftImg8bit" / split).rglob("*_leftImg8bit.png") if path.is_file())
        pairs = [(image, masks / "gtFine" / split / image.parent.name /
                  image.name.replace("_leftImg8bit.png", "_gtFine_labelTrainIds.png"),
                  "cityscapes") for image in images]
        expected = 2975 if split == "train" else 500
        available_masks = set((masks / "gtFine" / split).rglob("*_gtFine_labelTrainIds.png"))
    elif dataset == "acdc":
        if split != "val":
            raise ValueError("Only ACDC val is allowed; target training is prohibited")
        pairs = []
        available_masks = set()
        for domain, count in (("fog", 100), ("night", 106), ("rain", 100), ("snow", 100)):
            base = root / "rgb_anon" / domain / "val"
            images = sorted(path for path in base.rglob("*_rgb_anon.png") if path.is_file())
            available_masks.update((masks / "gt" / domain / split).rglob("*_gt_labelTrainIds.png"))
            if not allow_subset and len(images) != count:
                raise ValueError(f"ACDC {domain}: expected {count} images, found {len(images)}")
            for image in images:
                relative = image.relative_to(base)
                mask = masks / "gt" / domain / "val" / relative.parent / relative.name.replace(
                    "_rgb_anon.png", "_gt_labelTrainIds.png")
                pairs.append((image, mask, domain))
        expected = 406
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not pairs or (not allow_subset and len(pairs) != expected):
        raise ValueError(f"{dataset}/{split}: expected {expected} pairs, found {len(pairs)}")
    rows = []
    for image, mask, domain in pairs:
        if not mask.is_file():
            raise FileNotFoundError(f"Missing trainId mask: {mask}; convert Cityscapes labelIds first")
        rows.append(dict(image=path_string(image), mask=path_string(mask), domain=domain, split=split))
    orphan_masks = available_masks - {mask for _, mask, _ in pairs}
    if orphan_masks:
        example = min(orphan_masks)
        raise ValueError(f"{len(orphan_masks)} trainId mask(s) without a matching image; first: {example}")
    return rows


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
