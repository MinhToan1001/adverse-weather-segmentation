import json
from pathlib import Path
import os
import sys
import zipfile
import pytest

@pytest.mark.parametrize('extracted',[False,True])
def test_kaggle_bootstrap_accepts_zip_or_extracted_input(tmp_path,extracted):
    root=Path(__file__).resolve().parents[1]
    notebook=json.loads((root/'notebooks/02_kaggle_train_segformer_b0.ipynb').read_text(encoding='utf-8'))
    source=tmp_path/'input/code';source.mkdir(parents=True)
    city=tmp_path/'input/cityscapes';(city/'leftImg8bit/train').mkdir(parents=True)
    work=tmp_path/'work'
    archive=source/'project-code.zip'
    if extracted:
        for name in ['src','scripts','configs']:
            (source/name).mkdir()
            (source/name/'test.txt').write_text('source fixture')
    else:
        with zipfile.ZipFile(archive,'w') as bundle:
            for name in ['src','scripts','configs']:
                bundle.writestr(name+'/test.txt','source fixture')
    cell=''.join(notebook['cells'][1]['source'])
    cell=cell.replace('/kaggle/input/YOUR_CODE_DATASET/project-code.zip',archive.as_posix()).replace('/kaggle/input/YOUR_CITYSCAPES_DATASET/cityscapes',city.as_posix()).replace('/kaggle/working/adverse-weather-segmentation',work.as_posix())
    previous=os.getcwd();paths=list(sys.path);environment=dict(os.environ)
    try:
        exec(compile(cell,'kaggle-bootstrap','exec'),{})
        assert (work/'src/test.txt').read_text()=='source fixture'
    finally:
        os.chdir(previous);sys.path[:]=paths;os.environ.clear();os.environ.update(environment)
