"""Standard-library source bootstrap shared by the three model notebooks."""
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


MARKERS=('scripts/run_research_campaign.py','src/training/checkpoints.py','src/models/registry.py')


def valid_source(path):
    path=Path(path)
    if path.is_dir():return all((path/n).is_file() for n in MARKERS)
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:return set(MARKERS).issubset(z.namelist())
    return False


def resolve_source(hint,input_root):
    hint=Path(hint);input_root=Path(input_root)
    if valid_source(hint):return hint
    search_root=hint if hint.is_dir() and hint.resolve()!=input_root.resolve() else input_root
    anchor=PurePosixPath(MARKERS[0])
    folders=sorted({p.parents[len(anchor.parts)-1] for p in search_root.rglob(anchor.name) if valid_source(p.parents[len(anchor.parts)-1])})
    candidates=folders or [p for p in sorted(search_root.rglob('*.zip')) if valid_source(p)]
    if len(candidates)!=1:raise ValueError(f'Need exactly one current code source; found{len(candidates)}: {candidates}. Set CODE_INPUT explicitly.')
    return candidates[0]


def extract_safe(archive,destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        seen=set()
        for item in z.infolist():
            name=PurePosixPath(item.filename.replace('\\','/'))
            if name.is_absolute() or '..' in name.parts or ':' in str(name) or stat.S_IFMT(item.external_attr>>16)==stat.S_IFLNK:
                raise ValueError('Unsafe archive member')
            key=str(name).rstrip('/').casefold()
            if key in seen:raise ValueError('Duplicate archive member')
            seen.add(key)
            if not (destination/str(name)).resolve().is_relative_to(destination):raise ValueError('Unsafe archive destination')
        z.extractall(destination)


def copy_source(source,destination):
    source=Path(source);destination=Path(destination)
    if not valid_source(source):raise ValueError('Stale or missing suite code')
    if source.is_file():extract_safe(source,destination)
    else:
        destination.mkdir(parents=True,exist_ok=True)
        for name in ('src','scripts','configs','tests','third_party'):
            if (source/name).exists():shutil.copytree(source/name,destination/name,dirs_exist_ok=True)
        for name in ('requirements.txt','requirements-demo.txt','pyproject.toml','app.py','THIRD_PARTY_NOTICES.md'):
            if (source/name).is_file():shutil.copy2(source/name,destination/name)
