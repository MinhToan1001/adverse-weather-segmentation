"""Build current notebooks and a code-only Kaggle upload bundle."""
from pathlib import Path
import zipfile
from scripts.build_dg_notebook import build as build_dg
from scripts.build_research_notebooks import build_all as build_legacy
ROOT=Path(__file__).resolve().parents[1]

def build():
    build_legacy()  # Archived files, retained for historical regression coverage only.
    notebooks=[build_dg()]
    destination=ROOT/'outputs/kaggle/project-code.zip'
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('src','scripts','configs'):
            for path in sorted((ROOT/folder).rglob('*')):
                if path.is_file() and path.suffix in {'.py','.yaml','.yml','.json'}:
                    archive.write(path,path.relative_to(ROOT).as_posix())
        for name in ('requirements.txt','requirements-demo.txt','pyproject.toml','THIRD_PARTY_NOTICES.md','third_party/transformers/LICENSE'):
            archive.write(ROOT/name,name)
    print('Created',len(notebooks),'notebooks;source ZIP:',destination)
    return notebooks,destination

if __name__=='__main__':build()
