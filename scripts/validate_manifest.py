import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.validation import validate_manifests


def main():
    parser = argparse.ArgumentParser(description="Validate pairs and cross-manifest path disjointness")
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--check-labels", action="store_true")
    args = parser.parse_args()
    try:
        count = validate_manifests(args.manifest, args.check_files, args.check_labels)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"PASS manifest rows={count}; files={args.check_files}; labels={args.check_labels}")


if __name__ == "__main__":
    main()
