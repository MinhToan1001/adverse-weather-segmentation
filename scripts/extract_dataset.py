"""Extract only protocol splits from an official dataset ZIP, preserving originals."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def extract_archive(archive, destination, dataset):
    destination = Path(destination).resolve()
    count = 0
    with zipfile.ZipFile(archive) as bundle:
        selected = []
        for entry in bundle.infolist():
            relative = PurePosixPath(entry.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in entry.filename:
                raise ValueError(f"Unsafe ZIP path: {entry.filename}")
            parts = relative.parts
            if dataset == "cityscapes":
                keep = len(parts) >= 3 and parts[0] in {"gtFine", "leftImg8bit"} and parts[1] in {"train", "val"}
            else:
                keep = len(parts) >= 4 and parts[0] in {"gt", "rgb_anon"} and parts[1] in {"fog", "night", "rain", "snow"} and parts[2] == "val"
            if keep and not entry.is_dir():
                target = destination.joinpath(*parts).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("ZIP member escapes destination")
                if target.exists():
                    raise FileExistsError(f"Refusing to overwrite {target}")
                selected.append((entry, target))
        if not selected:
            raise ValueError("No expected dataset split found; check the ZIP package")
        for entry, target in selected:
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--dataset", choices=("cityscapes", "acdc"), required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(f"Extracted {extract_archive(args.archive, args.destination, args.dataset)} files")


if __name__ == "__main__":
    main()
