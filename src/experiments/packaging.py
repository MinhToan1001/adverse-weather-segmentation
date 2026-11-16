"""Atomic per-job research snapshots, compatible with Kaggle nested ZIP expansion."""
from pathlib import Path
import zipfile


def export_workspace(workspace, destination):
    workspace, destination = Path(workspace).resolve(), Path(destination).resolve()
    if destination.is_relative_to(workspace):
        raise ValueError('Save the archive outside the directory being archived')
    temporary = destination.with_suffix('.partial.zip')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(workspace.rglob('*')):
            relative = path.relative_to(workspace)
            redundant_source = any(part == 'source' and (workspace.joinpath(*relative.parts[:i]) / 'source.zip').is_file()
                                   for i, part in enumerate(relative.parts))
            if path.is_file() and not redundant_source:
                archive.write(path, relative.as_posix())
    temporary.replace(destination)
    return str(destination)
