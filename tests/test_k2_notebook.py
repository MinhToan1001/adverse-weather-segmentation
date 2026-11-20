"""Run actual Kaggle setup cells against fresh and resumed source snapshots."""
import ast
import json
import os
from pathlib import Path
import sys

import pytest

from scripts.build_k2_notebook import build
from scripts.build_kaggle_notebooks import build as package
from src.experiments.packaging import export_workspace
from src.experiments.run_support import fingerprints, snapshot_source


def execute_setup(code):
    previous=Path.cwd();paths=list(sys.path)
    try:
        namespace={};exec(compile(code,'<actual-k2-setup>','exec'),namespace);return namespace
    finally:
        os.chdir(previous);sys.path[:]=paths


def setup_code(base,inputs):
    notebook=json.loads(build().read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']))
            assert cell['outputs']==[] and cell['execution_count'] is None
    code=''.join(notebook['cells'][1]['source'])
    return code.replace(repr('/kaggle/working'),repr(str(base))).replace(repr('/kaggle/input'),repr(str(inputs)))


def make_city(inputs):
    city=inputs/'city'
    for split in ('train','val'):
        for name in ('leftImg8bit','gtFine'):
            (city/name/split).mkdir(parents=True)
    return city


def test_k2_notebook_fresh_setup_uses_combined_k1_k2_upload_without_network(tmp_path):
    bundle=package()[1]
    inputs=tmp_path/'input';city=make_city(inputs)
    code=setup_code(tmp_path/'working',inputs)
    code=code.replace('CODE_INPUT = None',f'CODE_INPUT = Path({str(bundle)!r})')
    namespace=execute_setup(code)
    assert namespace['CITYSCAPES_ROOT']==city and namespace['ARM']=='baseline'
    assert namespace['WEATHER_MODE']=='no-fog'
    repo=namespace['REPO']
    assert (repo/'src/models/proposed/weatherseg_b0.py').is_file()
    assert (repo/'src/data/augmentations/weather/generator.py').is_file()
    assert not (repo/'src/data/augmentations/synthetic_conditions.py').exists()


@pytest.mark.parametrize('expanded',[False,True])
def test_resume_setup_restores_original_weather_settings_and_frozen_code(tmp_path,expanded):
    inputs=tmp_path/'input';city=make_city(inputs)
    original=tmp_path/'saved';original.mkdir()
    hashes=fingerprints();snapshot_source(original/'source.zip',hashes)
    plan=dict(protocol='weatherseg_k2_source_v1',arm='k2',seed=42,epochs=20,batch_size=2,
        city_root=str(city),source_hashes=hashes,
        weather_inputs=dict(mode='full',depth_root=str(inputs/'depth'),distance_root=None,invalid_policy='error'))
    (original/'k2_plan.json').write_text(json.dumps(plan))
    archive=tmp_path/'saved.zip';export_workspace(original,archive)
    code=setup_code(tmp_path/'working',inputs)
    code=code.replace("ARM = 'baseline'", "ARM = 'k2'")
    code=code.replace('RESUME_INPUT = None',f'RESUME_INPUT = Path({str(original if expanded else archive)!r})')
    namespace=execute_setup(code)
    assert namespace['WEATHER_MODE']=='full' and namespace['INVALID_POLICY']=='error'
    assert namespace['DEPTH_ROOT']==str(inputs/'depth')
    assert namespace['CITYSCAPES_ROOT']==city
    assert (namespace['REPO']/'scripts/run_k2.py').is_file()
