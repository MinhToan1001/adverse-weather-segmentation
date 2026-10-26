import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import shlex
import sys

import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from src.data.dataset import ManifestSegmentationDataset
from src.training.train import evaluate_model, read_cfg


CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic_light",
    "traffic_sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


def _write_csv(path, report, dataset_name):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "command", "run_id", "result_status", "dataset", "domain", "class", "iou", "miou", "ci_low", "ci_high",
        "n_images", "manifest_sha256", "config_sha256", "checkpoint_sha256", "commit_hash",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for domain, metrics in sorted(report["by_domain"].items()):
            ci = metrics["miou_ci95"] or [None, None]
            writer.writerow({
                "command": report["command"], "run_id": report["run_id"],
                "result_status": report["result_status"],
                "dataset": dataset_name, "domain": domain,
                "class": "__mean__", "iou": "", "miou": metrics["miou"],
                "ci_low": ci[0], "ci_high": ci[1], "n_images": metrics["n_images"],
                "manifest_sha256": report["manifest_sha256"],
                "config_sha256": report["config_sha256"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "commit_hash": report["commit_hash"],
            })
            for class_id, iou in enumerate(metrics["per_class_iou"]):
                writer.writerow({
                    "command": report["command"], "run_id": report["run_id"],
                    "result_status": report["result_status"],
                    "dataset": dataset_name, "domain": domain,
                    "class": CLASS_NAMES[class_id], "iou": iou, "miou": "",
                    "ci_low": "", "ci_high": "", "n_images": metrics["n_images"],
                    "manifest_sha256": report["manifest_sha256"],
                    "config_sha256": report["config_sha256"],
                    "checkpoint_sha256": report["checkpoint_sha256"],
                    "commit_hash": report["commit_hash"],
                })


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sha256(path):
    root = Path(path)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Evaluate one frozen checkpoint in one inference pass.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", help="Override val_manifest, e.g. the frozen ACDC validation manifest")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--run-id", default="evaluation")
    parser.add_argument(
        "--result-status", choices=("confirmatory", "exploratory"), default="exploratory",
        help="Use confirmatory only for a model frozen without ACDC feedback.",
    )
    parser.add_argument("--output-csv")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    cfg = read_cfg(args.config)
    manifest = args.manifest or cfg["data"]["val_manifest"]
    manifest_path = Path(manifest)
    if not manifest_path.is_absolute() and not manifest_path.is_file():
        manifest_path = Path(__file__).resolve().parents[2] / manifest_path
    manifest = str(manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint).to(device)
    dataset = ManifestSegmentationDataset(manifest, cfg["data"]["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )
    report = evaluate_model(
        model,
        loader,
        device,
        int(cfg["model"]["num_labels"]),
        bootstrap_samples=max(0, args.bootstrap_samples),
        seed=int(cfg["seed"]),
    )
    repo_root = Path(__file__).resolve().parents[2]
    try:
        dirty = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if dirty:
            raise RuntimeError("Commit tracked code/config changes before recording an evaluation result")
        commit_hash = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit_hash = "unknown"
    report.update({
        "run_id": args.run_id,
        "result_status": args.result_status,
        "command": shlex.join([sys.executable, *sys.argv]),
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "checkpoint": str(args.checkpoint),
        "config_sha256": _sha256_file(args.config),
        "checkpoint_sha256": _checkpoint_sha256(args.checkpoint),
        "commit_hash": commit_hash,
    })
    rendered = json.dumps(report, indent=2, allow_nan=False)
    print(rendered)

    dataset_name = Path(manifest).stem
    if args.output_csv:
        _write_csv(args.output_csv, report, dataset_name)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
