"""Exercise actual generated setup cells on fresh/extracted Kaggle-like mounts."""
import json
import os
from pathlib import Path
import shutil
import sys

import pytest
import torch

from scripts.build_kaggle_notebooks import build as package
from scripts.build_dg_notebook import NOTEBOOKS, ROOT
from scripts.kaggle_bootstrap import extract_safe,resolve_source


@pytest.fixture(scope='module')
def source_bundle():
    return package()[1]


@pytest.mark.parametrize('model',list(NOTEBOOKS))
@pytest.mark.parametrize('extracted',[False,True])
def test_fresh_dg_setup_finds_cityscapes_and_code_zip_or_folder(tmp_path,monkeypatch,source_bundle,model,extracted):
    inputs=tmp_path/'input';code_dir=inputs/'datasets/user/code';code_dir.mkdir(parents=True)
    if extracted:extract_safe(source_bundle,code_dir)
    else:shutil.copy2(source_bundle,code_dir/'project-code.zip')
    city=inputs/'datasets/user/city/Cityscape'
    for split in ['train','val']:
        for folder in ['leftImg8bit','gtFine']:(city/folder/split).mkdir(parents=True)
    base=tmp_path/'working'
    repo=base/f'project_{model}_suite'
    notebook=json.loads((ROOT/'notebooks'/NOTEBOOKS[model]).read_text(encoding='utf-8'))
    code=''.join(notebook['cells'][1]['source'])
    replacements={"/kaggle/input":inputs,'/kaggle/working':base}
    for old,new in replacements.items():code=code.replace(f'Path({old!r})',f'Path({str(new)!r})')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *args:'Mock for setup only')
    def execute(text):
        previous=Path.cwd();paths=list(sys.path)
        try:
            namespace={};exec(compile(text,'<actual-notebook-setup>','exec'),namespace);return namespace
        finally:os.chdir(previous);sys.path[:]=paths
    fresh=execute(code)
    assert fresh['CITYSCAPES_ROOT']==city and fresh['MODEL']==model
    assert 'ACDC_ROOT' not in fresh and (repo/'scripts/run_dg_suite.py').is_file()


def test_explicit_code_dataset_zip_wins_over_an_unrelated_extracted_result(tmp_path,source_bundle):
    inputs=tmp_path/'input';code=inputs/'code';code.mkdir(parents=True)
    shutil.copy2(source_bundle,code/'project-code.zip')
    extract_safe(source_bundle,inputs/'unrelated_result/source')
    assert resolve_source(code,inputs)==code/'project-code.zip'
