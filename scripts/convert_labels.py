"""Convert Cityscapes train/val raw labelIds to a separate processed tree."""
import argparse
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preparation import convert_label_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-root", default="data/processed/cityscapes", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = []
    for split in ("train", "val"):
        for source in sorted((args.root / "gtFine" / split).rglob("*_gtFine_labelIds.png")):
            relative = source.relative_to(args.root)
            destination = args.output_root / relative.parent / relative.name.replace(
                "_labelIds.png", "_labelTrainIds.png")
            if destination.exists():
                parser.error(f"Output already exists: {destination}; use a new output root")
            tasks.append((source, destination))
    if not tasks:
        parser.error("No Cityscapes train/val labelIds found")
    def convert(pair):
        convert_label_ids(*pair)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for count, _ in enumerate(executor.map(convert, tasks), 1):
            if count % 250 == 0:
                print(f"Converted {count}/{len(tasks)}", flush=True)
    count = len(tasks)
    print(f"Converted {count} source masks; originals preserved; test split unused")


if __name__ == "__main__":
    main()
