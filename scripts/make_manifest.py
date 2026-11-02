"""Create sorted Cityscapes/ACDC manifests without opening image or mask pixels."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preparation import create_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", choices=("cityscapes", "acdc"), required=True)
    parser.add_argument("--mask-root", help="Tree containing gtFine/ or gt/; defaults to --root")
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--allow-subset", action="store_true", help="Exploratory subset only")
    parser.add_argument("--split", choices=("train", "val"), help="Build one split; default: both for Cityscapes, val for ACDC")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing generated CSV")
    args = parser.parse_args()
    splits = (args.split,) if args.split else (("train", "val") if args.dataset == "cityscapes" else ("val",))
    outputs = []
    # Check all requested splits before writing any file.
    for split in splits:
        output = args.output_dir / f"{args.dataset}_{split}.csv"
        if output.exists() and not args.overwrite:
            parser.error(f"Refusing to overwrite {output}; use --overwrite after review")
        try:
            rows = create_manifest(args.root, args.dataset, split, args.mask_root, args.allow_subset)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        outputs.append((rows, output))
    for rows, output in outputs:
        write_manifest(rows, output)
        print(f"{output}: {len(rows)} pairs; subset={args.allow_subset}")


if __name__ == "__main__":
    main()
