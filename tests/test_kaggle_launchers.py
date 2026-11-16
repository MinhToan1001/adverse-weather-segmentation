"""Exercise actual generated setup cells on fresh/extracted Kaggle-like mounts."""
import json
import os
from pathlib import Path
import shutil
import sys

import pytest
import torch

from scripts.build_kaggle_notebooks import build as package
from scripts.build_research_notebooks import NOTEBOOKS, ROOT
from scripts.kaggle_bootstrap import extract_safe,resolve_source
from scripts.run_research_campaign import prepare_campaign
from src.experiments.packaging import export_workspace


@pytest.fixture(scope='module')
def source_bundle():
    return package()[1]


@pytest.mark.parametrize('model',list(NOTEBOOKS))
@pytest.mark.parametrize('extracted',[False,True])
def test_setup_finds_datasets_and_restores_original_model_source(tmp_path,monkeypatch,source_bundle,model,extracted):
    inputs=tmp_path/'input';code_dir=inputs/'datasets/user/code';code_dir.mkdir(parents=True)
    if extracted:extract_safe(source_bundle,code_dir)
    else:shutil.copy2(source_bundle,code_dir/'project-code.zip')
    city=inputs/'datasets/user/city/Cityscape';acdc=inputs/'datasets/user/acdc/ACDC'
    for p in (city/'leftImg8bit/train',city/'gtFine/train',acdc/'rgb_anon_trainvaltest/rgb_anon',acdc/'gt_trainval/gt'):p.mkdir(parents=True)
    work,repo,archive=tmp_path/'work',tmp_path/'repo',tmp_path/'result.zip'
    notebook=json.loads((ROOT/'notebooks'/NOTEBOOKS[model]).read_text(encoding='utf-8'))
    code=''.join(notebook['cells'][1]['source'])
    replacements={"/kaggle/input":inputs,f'/kaggle/working/campaign_{model}':work,
                  f'/kaggle/working/project_{model}':repo,f'/kaggle/working/{model}_seed42.zip':archive}
    for old,new in replacements.items():code=code.replace(f'Path({old!r})',f'Path({str(new)!r})')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *args:'Mock for setup only')
    def execute(text):
        previous=Path.cwd();paths=list(sys.path)
        try:
            namespace={};exec(compile(text,'<actual-notebook-setup>','exec'),namespace);return namespace
        finally:os.chdir(previous);sys.path[:]=paths
    fresh=execute(code)
    assert fresh['CITYSCAPES_ROOT']==city and fresh['ACDC_ROOT']==acdc
    (work/'data').mkdir();(work/'data/data.json').write_text('{"paths":{}}')
    plan=prepare_campaign(work,work/'data/data.json',models=[model])
    export_workspace(work,archive)
    previous=inputs/'result';extract_safe(archive,previous)
    if extracted:
        extract_safe(previous/'source.zip',previous/'source');(previous/'source.zip').unlink()
    for directory in (work,repo):
        assert directory.resolve().is_relative_to(tmp_path.resolve());shutil.rmtree(directory)
    restored=execute(code.replace('RESUME_INPUT=None',f'RESUME_INPUT=Path({str(previous)!r})'))
    assert restored['MODEL_NAME']==model and (work/'source.zip').is_file()
    assert prepare_campaign(work,work/'data/data.json',models=[model])==plan


def test_explicit_code_dataset_zip_wins_over_an_unrelated_extracted_result(tmp_path,source_bundle):
    inputs=tmp_path/'input';code=inputs/'code';code.mkdir(parents=True)
    shutil.copy2(source_bundle,code/'project-code.zip')
    extract_safe(source_bundle,inputs/'unrelated_result/source')
    assert resolve_source(code,inputs)==code/'project-code.zip'
