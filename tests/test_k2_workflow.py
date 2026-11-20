"""Candidate-recipe isolation, frozen target evaluation, figures and notebook cells."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
import yaml

from scripts.run_k2 import compose
from scripts.k2_results import compare
from scripts.build_k2_workflow import build_all
from scripts.build_kaggle_notebooks import build as package
from scripts.kaggle_bootstrap import extract_safe
import scripts.evaluate_k2 as evaluator
from scripts.plot_k2 import plot_result
from src.experiments.run_support import fingerprints, snapshot_source
from src.models.proposed.weatherseg_b0 import WeatherSegB0
from test_k2_results import make_result


def dump(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


def test_candidate_recipe_is_opt_in_and_preserves_matched_controls(tmp_path):
    paths=dict(source_train_manifest='train.csv',source_monitor_manifest='val.csv')
    old=compose('k2',paths,tmp_path/'old')
    controls={arm:compose(arm,paths,tmp_path/arm,epochs=40,recipe='balanced_v2')
              for arm in ('baseline','film_clean','aug','film','k2')}
    assert old['data']['image_size']==[512,1024] and old['train']['warmup_steps']==20
    assert old['data']['augment']['weather']['severities']==[1,2,3,4,5]
    assert all(c['data']['image_size']==[640,1280] and c['train']['warmup_steps']==500 for c in controls.values())
    assert all(c['train']==controls['baseline']['train'] for c in controls.values())
    assert controls['baseline']['data']==controls['film_clean']['data']
    assert all(controls[a]['data']==controls['k2']['data'] for a in ('aug','film'))
    policy=controls['k2']['data']['augment']['weather']
    assert policy['probabilities']==[.5,1/6,1/6,1/6] and policy['severities']==[1,2,3]


def test_results_never_mix_candidate_and_legacy_recipes(tmp_path):
    legacy=make_result(tmp_path/'legacy','baseline')
    candidate=make_result(tmp_path/'candidate','k2',recipe='balanced_v2')
    with pytest.raises(ValueError,match='recipe/seed'):compare([legacy,candidate])
    baseline=make_result(tmp_path/'candidate_baseline','baseline',recipe='balanced_v2')
    assert compare([baseline,candidate])['core_source_and_initialization_verified']


def completed_real_checkpoint(root):
    root=make_result(root)
    cfg=yaml.safe_load((root/'train.yaml').read_text());cfg['data']['image_size']=[64,96]
    (root/'train.yaml').write_text(yaml.safe_dump(cfg))
    torch.manual_seed(42);WeatherSegB0().save_pretrained(root/'run/best')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    checkpoint=json.loads((root/'run/best/checkpoint.json').read_text())
    checkpoint['files']={n:sha(root/'run/best'/n) for n in ('model.safetensors','config.json')}
    dump(root/'run/best/checkpoint.json',checkpoint)
    plan=json.loads((root/'k2_plan.json').read_text());plan['source_hashes']=fingerprints()
    plan['frozen_files']['train.yaml']=sha(root/'train.yaml')
    snapshot_source(root/'source.zip',plan['source_hashes']);dump(root/'k2_plan.json',plan)
    candidate=json.loads((root/'frozen_candidate.json').read_text())
    candidate.update(weights_sha256=sha(root/'run/best/model.safetensors'),
        config_sha256=sha(root/'run/best/config.json'),plan_sha256=sha(root/'k2_plan.json'))
    dump(root/'frozen_candidate.json',candidate)
    return root


def tiny_eval_data(root):
    city=root/'city';acdc=root/'acdc'
    for i in range(2):
        stem=f'aachen_{i:06d}_000019'
        rgb=city/'leftImg8bit/val/aachen'/(stem+'_leftImg8bit.png')
        gt=city/'gtFine/val/aachen'/(stem+'_gtFine_labelTrainIds.png')
        rgb.parent.mkdir(parents=True,exist_ok=True);gt.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(96,64),(70,110,130)).save(rgb)
        mask=np.zeros((64,96),np.uint8);mask[20:50,20:70]=13;mask[:2]=255;Image.fromarray(mask).save(gt)
    for domain in ('fog','night','rain','snow'):
        stem='GOPR0001_frame_000001'
        rgb=acdc/'rgb_anon'/domain/'val/GOPR0001'/(stem+'_rgb_anon.png')
        gt=acdc/'gt'/domain/'val/GOPR0001'/(stem+'_gt_labelTrainIds.png')
        rgb.parent.mkdir(parents=True,exist_ok=True);gt.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(96,64),(60,100,110)).save(rgb);Image.fromarray(mask).save(gt)
    return city,acdc


def test_partial_candidate_is_blocked_before_any_target_access(tmp_path,monkeypatch):
    root=make_result(tmp_path/'partial',complete=False)
    monkeypatch.setattr(evaluator,'_tree',lambda *a,**k:pytest.fail('target accessed before seal'))
    with pytest.raises(ValueError,match='Finish the original'):
        evaluator.evaluate_result(root,tmp_path/'eval')
    assert not (tmp_path/'eval').exists()


def test_frozen_evaluation_and_figures_use_actual_shared_model_on_tiny_qa(tmp_path,monkeypatch):
    """Six real inference passes + native metrics/panels, not a benchmark result."""
    root=completed_real_checkpoint(tmp_path/'result');city,acdc=tiny_eval_data(tmp_path/'data')
    monkeypatch.setattr(evaluator,'CITY_COUNT',2)
    monkeypatch.setattr(evaluator,'ACDC_COUNTS',{d:1 for d in ('fog','night','rain','snow')})
    before=(root/'run/best/model.safetensors').read_bytes()
    out=tmp_path/'eval'
    reports=evaluator.evaluate_result(root,out,city,acdc,device='cpu')
    assert reports['cityscapes_val']['n_images']==2 and reports['acdc_val']['n_images']==4
    assert set(reports['acdc_val']['by_domain'])=={'fog','night','rain','snow'}
    assert json.loads((out/'evaluation_status.json').read_text())['status']=='complete'
    assert (root/'run/best/model.safetensors').read_bytes()==before
    assert (out/'checkpoint/model.safetensors').read_bytes()==before
    figures=plot_result(root,tmp_path/'figures',out)
    assert {'training_pipeline.png','learning_curves.png','class_learning.png','best_class_iou.png',
            'weather_miou.png','acdc_val_confusion.png'}.issubset(figures['figures'])
    assert (tmp_path/'figures/training_pipeline.svg').is_file()
    assert (out/'acdc_val/panels/night.png').is_file()
    with pytest.raises(ValueError,match='Preserve previous'):
        evaluator.evaluate_result(root,out,city,acdc,device='cpu')


def test_eval_rejects_incomplete_official_split_and_preserves_seal(tmp_path):
    root=completed_real_checkpoint(tmp_path/'result');city,acdc=tiny_eval_data(tmp_path/'data')
    with pytest.raises(ValueError,match='500'):
        evaluator.evaluate_result(root,tmp_path/'eval',city,acdc,device='cpu')
    assert (tmp_path/'eval/evaluation_seal.json').is_file()
    assert json.loads((tmp_path/'eval/evaluation_status.json').read_text())['status']=='failed_preserve_outputs'


@pytest.mark.parametrize('index',[0,1])
def test_generated_workflow_setup_cells_use_uploaded_source_without_network(tmp_path,index):
    bundle=package()[1];paths=build_all()
    notebook=json.loads(paths[index].read_text(encoding='utf-8'))
    code=''.join(notebook['cells'][1]['source'])
    inputs=tmp_path/'input';city=inputs/'city'
    for split in ('train','val'):
        for folder in ('leftImg8bit','gtFine'):(city/folder/split).mkdir(parents=True,exist_ok=True)
    code=code.replace(repr('/kaggle/input'),repr(str(inputs))).replace(repr('/kaggle/working'),repr(str(tmp_path/'working')))
    code=code.replace('CODE_INPUT = None',f'CODE_INPUT = {str(bundle)!r}')
    previous=Path.cwd();paths_before=list(sys.path)
    try:
        namespace={};exec(compile(code,'<actual-new-workflow-setup>','exec'),namespace)
    finally:
        os.chdir(previous);sys.path[:]=paths_before
    assert (namespace['REPO']/'scripts/evaluate_k2.py').is_file()
    if index==0:
        assert namespace['RECIPE']=='balanced_v2' and namespace['EPOCHS']==40
        assert namespace['EVAL_AFTER_TRAIN'] and namespace['ARM']=='baseline'
    else:
        assert namespace['RESULT_INPUT'] is None and namespace['EVAL_ROOT'].name=='k2_post_training_evaluation'
