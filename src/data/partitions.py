"""Grouped,fit-only Cityscapes/ACDC partitions shared by every model."""
from collections import Counter,defaultdict
import csv,hashlib,json,random
from itertools import combinations
from pathlib import Path
import numpy as np
from PIL import Image
from src.data.acdc import recording_group
from src.data.preparation import path_string,resolve_data_path,write_manifest
from src.data.replay import source_scene,validate_source_monitor
from src.data.validation import read_manifest,validate_manifests
WEATHER=("fog","night","rain","snow")

def prepare_source_split(manifest, output):
    if output.exists():
        raise ValueError("Choose a fresh source replay manifest directory")
    rows = read_manifest(manifest)
    if len(rows) != 2975 or any(r["domain"] != "cityscapes" or r["split"] != "train" for r in rows):
        raise ValueError("Expected all 2975 official Cityscapes train samples")
    groups = sorted({source_scene(row) for row in rows})
    rng = random.Random(42)
    rng.shuffle(groups)
    monitor_groups = set(groups[:max(1, round(.1 * len(groups)))])
    fit = [row for row in rows if source_scene(row) not in monitor_groups]
    monitor = [row for row in rows if source_scene(row) in monitor_groups]
    validate_source_monitor(fit, monitor)
    output.mkdir(parents=True)
    for role, subset in (("fit", fit), ("monitor", monitor)):
        write_manifest(subset, output / f"cityscapes_{role}.csv")
    validate_manifests([output / "cityscapes_fit.csv", output / "cityscapes_monitor.csv"], check_files=True)
    report = dict(status="PASS", seed=42, official_split="train", fit_images=len(fit), monitor_images=len(monitor),
                  group_rule="Parent city plus sequence token in the image filename", scene_groups_disjoint=True,
                  limit="Monitor is withheld from source fit and replay;development monitoring is not an independent test")
    (output / "split_qa.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def choose_dev_groups(counts, rng, fraction=0.2):
    """Approximate a frame fraction without splitting any recording family.

    Membership uses counts only, never labels or model quality. Prefer the
    larger dev set on an equal-distance tie, then fewer groups.
    """
    ordered = sorted(counts)
    candidates = []
    target = sum(counts.values()) * fraction
    for size in range(1, len(ordered)):
        for subset in combinations(ordered, size):
            count = sum(counts[group] for group in subset)
            candidates.append(((abs(count - target), count < target, size, rng.random()), subset))
    if not candidates:
        raise ValueError("At least two recording families per weather are required")
    return set(min(candidates)[1])


def prepare_target_split(rgb_root, gt_root, output, seed=42, check_labels=False):
    if output.exists():
        raise ValueError("Choose a fresh adaptation manifest directory")
    rows, references = [], []
    groups = defaultdict(Counter)
    for domain in WEATHER:
        images = sorted((rgb_root / domain / "train").rglob("*_rgb_anon.png"))
        if len(images) != 400:
            raise ValueError(f"Expected 400 official training images for {domain}")
        for image in images:
            group = image.parent.name
            mask = gt_root / domain / "train" / group / image.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
            reference = rgb_root / domain / "train_ref" / group / image.name.replace("_rgb_anon.png", "_rgb_ref_anon.png")
            if not mask.is_file() or not reference.is_file():
                raise FileNotFoundError(f"Missing label/reference for {image}")
            groups[domain][recording_group(image)] += 1
            rows.append(dict(image=path_string(image), mask=path_string(mask), domain=domain, split="train"))
            references.append(dict(image=path_string(image), reference=path_string(reference), domain=domain,
                                   sequence=group, recording_group=recording_group(image), split="train"))
    rng = random.Random(seed)
    dev_groups = set()
    for domain in WEATHER:
        dev_groups.update(choose_dev_groups(groups[domain], rng))
    fit = [row for row in rows if recording_group(row["image"]) not in dev_groups]
    dev = [row for row in rows if recording_group(row["image"]) in dev_groups]
    fit_groups = {recording_group(row["image"]) for row in fit}
    actual_dev_groups = {recording_group(row["image"]) for row in dev}
    if fit_groups & actual_dev_groups or set(row["domain"] for row in fit) != set(WEATHER) or set(row["domain"] for row in dev) != set(WEATHER):
        raise ValueError("Video-group split failed to preserve four-weather coverage")
    output.mkdir(parents=True)
    manifests = {}
    for name, subset in (("all", rows), ("fit", fit), ("dev", dev)):
        destination = output / f"acdc_train_{name}.csv"
        write_manifest(subset, destination)
        manifests[name] = destination
    with (output / "acdc_train_unlabeled_references.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(references[0]))
        writer.writeheader()
        writer.writerows(references)
    print(f"Prepared {len(fit)} fit / {len(dev)} dev; validating image/mask pairs...", flush=True)
    validate_manifests([manifests["fit"], manifests["dev"]], check_files=True,
                       check_labels=check_labels, allow_target_train=True)
    counts = np.zeros(256, dtype=np.int64)
    if check_labels:
        for index, row in enumerate(fit, start=1):
            with Image.open(resolve_data_path(row["mask"])) as mask:
                counts += np.bincount(np.asarray(mask).reshape(-1), minlength=256)
            if index % 200 == 0:
                print(f"Fit-only class histogram: {index}/{len(fit)}", flush=True)
    report = {
        "status": "PASS", "protocol": "supervised_acdc_internal_split", "seed": seed,
        "official_split": "train", "total_images": len(rows),
        "fit_counts": dict(Counter(row["domain"] for row in fit)),
        "dev_counts": dict(Counter(row["domain"] for row in dev)),
        "fit_groups": sorted(fit_groups), "dev_groups": sorted(actual_dev_groups),
        "grouping": "Conservatively merge GOPRxxxx and GPnnxxxx chapters with the same final four digits",
        "split_policy": "Choose whole recording families nearest 20% frames per weather; count-only, prefer larger dev on ties",
        "recording_family_counts": {domain: dict(counts) for domain, counts in groups.items()},
        "fit_dev_video_groups_disjoint": True, "reference_pairs": len(references),
        "labels_checked": check_labels,
        "fit_class_pixels": counts[:19].tolist() if check_labels else None,
        "fit_ignore_pixels": int(counts[255]) if check_labels else None,
        "manifest_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in manifests.items()},
        "limits": [
            "Internal dev is grouped by GoPro recording-family ID as a conservative proxy for correlated frames; GPS scene separation is not proven.",
            "The internal fit/dev group split is a development protocol, separate from the official paper benchmark.",
            "The reference CSV omits label paths for a future unlabeled-target UDA branch.",
        ],
    }
    (output / "split_qa.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
