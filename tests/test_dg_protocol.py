"""Prevent target-data selection and supervised checkpoints from entering DG."""
from copy import deepcopy
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.preparation import write_manifest
from src.experiments.protocol import DG_PROTOCOL, require_training_permission


def source_config(tmp_path):
    manifests={}
    for split,count in [('train',2975),('val',500)]:
        rows=[dict(image=str(tmp_path/'leftImg8bit'/split/'aachen'/f'aachen_{i:06d}_000019_leftImg8bit.png'),
                   mask=str(tmp_path/'gtFine'/split/'aachen'/f'aachen_{i:06d}_000019_gtFine_labelTrainIds.png'),
                   domain='cityscapes',split=split) for i in range(count)]
        path=tmp_path/f'{split}.csv';write_manifest(rows,path)
        manifests['train_manifest' if split=='train' else 'val_manifest']=str(path)
    return dict(protocol='source_only',experiment_protocol=DG_PROTOCOL,data=manifests,
                model=dict(initialization='imagenet_encoder'),train=dict(epochs=20))


def test_dg_uses_city_train_and_official_val_without_reading_image_pixels(tmp_path,monkeypatch):
    from PIL import Image
    monkeypatch.setattr(Image,'open',lambda *a,**k:pytest.fail('DG protocol checks must not open target/image pixels'))
    require_training_permission(source_config(tmp_path))


@pytest.mark.parametrize('role',['train_manifest','val_manifest'])
def test_reject_acdc_training_and_target_selection_even_with_override(tmp_path,role):
    cfg=source_config(tmp_path)
    target=tmp_path/'target.csv'
    write_manifest([dict(image='rain/val/GOPR1/x_rgb_anon.png',mask='rain/val/GOPR1/x_gt_labelTrainIds.png',
                         domain='rain',split='val')],target)
    cfg['data'][role]=str(target)
    with pytest.raises(ValueError,match='cannot use ACDC'):
        require_training_permission(cfg,allow_supervised_acdc=True)


def test_reject_adapted_initialization_wrong_split_and_extra_replay(tmp_path):
    cfg=source_config(tmp_path)
    for change,match in [({'model':dict(initialization='checkpoint',checkpoint='old/adapt/best')},'never an ACDC'),
                         ({'train':dict(epochs=30)},'20 epochs')]:
        bad=deepcopy(cfg);bad.update(change)
        with pytest.raises(ValueError,match=match):require_training_permission(bad)
    bad=deepcopy(cfg);bad['data']['source_train_manifest']=cfg['data']['train_manifest']
    with pytest.raises(ValueError,match='only Cityscapes'):require_training_permission(bad)
    bad=deepcopy(cfg);bad['data']['val_manifest']=cfg['data']['train_manifest']
    with pytest.raises(ValueError,match='val: 500'):require_training_permission(bad)


def test_historical_supervised_requires_explicit_override_and_cannot_claim_dg():
    cfg=dict(protocol='supervised_acdc_replay')
    with pytest.raises(ValueError,match='outside DG'):require_training_permission(cfg)
    require_training_permission(cfg,allow_supervised_acdc=True)
    cfg['experiment_protocol']=DG_PROTOCOL
    with pytest.raises(ValueError,match='outside DG'):require_training_permission(cfg,True)


def test_retired_model_cannot_start_a_new_dg_run(tmp_path):
    from scripts.run_dg import prepare
    from scripts.build_dg_notebook import build
    with pytest.raises(ValueError,match='supported DG model'):
        prepare(tmp_path/'city',tmp_path/'work',model='segformer_b0')
    with pytest.raises(ValueError,match='Unknown DG model'):
        build('segformer_b0')
    assert not (tmp_path/'work').exists()


def test_active_notebook_has_no_target_input_or_adapt_command():
    from scripts.build_dg_notebook import build
    nb=json.loads(build().read_text(encoding='utf-8'))
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    assert 'scripts.run_dg_suite' in code and 'CITYSCAPES_ROOT' in code
    assert 'ARM=' not in code and 'PROFILE_ONLY=' not in code
    assert 'ACDC_ROOT' not in code and 'scripts.prepare_research_data' not in code
    assert 'scripts.run_research_campaign' not in code
    assert 'source_hashes' in code and 'RESUME_INPUT=None' in code


def test_dg_preparation_freezes_manifests_and_rejects_drift(tmp_path,monkeypatch):
    from scripts import run_dg
    def data(city,output):
        output.mkdir()
        cfg=source_config(output)
        # Keep filenames identical to the production preparer.
        result={}
        for split,role in [('train','source_train_manifest'),('val','source_monitor_manifest')]:
            old=output/f'{split}.csv';new=output/f'cityscapes_{split}.csv'
            old.rename(new);result[role]=str(new)
        return result
    monkeypatch.setattr(run_dg,'prepare_cityscapes',data)
    monkeypatch.setattr(run_dg,'validate_manifests',lambda *a,**k:None)
    monkeypatch.setattr(run_dg,'snapshot_source',lambda path,hashes:path.write_bytes(b'source snapshot fixture'))
    root=tmp_path/'work'
    first=run_dg.prepare(tmp_path/'city',root)
    assert run_dg.prepare(tmp_path/'city',root)==first
    assert 'ACDC' not in json.loads((root/'dg_plan.json').read_text())['city_root']
    path=root/'data/cityscapes_val.csv'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='Frozen DG config'):
        run_dg.prepare(tmp_path/'city',root)


def test_dg_queue_exports_then_resumes_only_unfinished_epochs(tmp_path,monkeypatch):
    from scripts import run_dg
    import torch
    import src.training.checkpoints as checkpoints
    root=tmp_path/'work';run=root/'run';run.mkdir(parents=True)
    monkeypatch.setattr(run_dg,'prepare',lambda *args:dict(output_dir=str(run)))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(checkpoints,'recover_run',lambda *args:0)
    monkeypatch.setattr(run_dg,'committed_state',lambda *args:None)
    calls=[];exports=[]
    def trainer(command,check):
        epoch=int(command[command.index('--stop-after-epoch')+1]);calls.append(epoch)
        if epoch>1:assert '--resume' in command
        for slot in ['last','best']:
            (run/slot).mkdir(exist_ok=True)
            (run/slot/'checkpoint.json').write_text(json.dumps(dict(epoch=epoch)))
            (run/slot/'config.json').write_text('{}')
            (run/slot/'model.safetensors').write_bytes(b'unit orchestration fixture, not model weights')
        (run/'config.json').write_text('{}')
        (run/'initialization.json').write_text('{}')
        with (run/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,epoch_seconds=1))+'\n')
    (root/'dg_plan.json').write_text('{}')
    # Keep Matplotlib's font discovery on the real subprocess module.
    monkeypatch.setattr(run_dg,'subprocess',SimpleNamespace(run=trainer))
    monkeypatch.setattr(run_dg,'export_workspace',lambda *args:exports.append(args))
    monkeypatch.setattr(run_dg,'fits_session',lambda *args:len(calls)<2)
    run_dg.run(tmp_path/'city',root,tmp_path/'result.zip')
    assert calls==[1,2] and len(exports)==3
    assert json.loads((root/'status.json').read_text())['status']=='session_checkpoint_ready'
    monkeypatch.setattr(run_dg,'fits_session',lambda *args:True)
    run_dg.run(tmp_path/'city',root,tmp_path/'result.zip')
    assert calls==list(range(1,21))
    frozen=json.loads((root/'frozen_candidate.json').read_text())
    assert frozen['target_training'] is False and frozen['selected_by']=='Cityscapes val500 only'


@pytest.mark.parametrize('model',['segformer_b2','deeplabv3plus_resnet50'])
def test_dg_notebook_restores_kaggle_autoexpanded_source_zip(tmp_path,model):
    import os
    import sys
    import shutil
    from scripts.build_dg_notebook import build
    from src.experiments.run_support import fingerprints, snapshot_source
    from scripts.kaggle_bootstrap import extract_safe
    from src.experiments.packaging import export_workspace
    inputs=tmp_path/'inputs';inputs.mkdir()
    city=inputs/'city'
    for name in ['leftImg8bit/train','leftImg8bit/val','gtFine/train','gtFine/val']:(city/name).mkdir(parents=True)
    previous=inputs/'previous';previous.mkdir()
    hashes=fingerprints();snapshot_source(previous/'source.zip',hashes)
    extract_safe(previous/'source.zip',previous/'source')
    (previous/'source.zip').unlink()  # Simulate Kaggle expanding the nested source archive.
    (previous/'suite_plan.json').write_text(json.dumps(dict(schema='automatic_dg_model_suite_v1',protocol=DG_PROTOCOL,
        source_hashes=hashes,city_root=str(city),model=model,seed=42)))
    work=tmp_path/f'working/dg_{model}_suite_seed42'
    repo=tmp_path/f'working/project_{model}_suite'
    setup=''.join(json.loads(build(model).read_text(encoding='utf-8'))['cells'][1]['source'])
    for old,new in [('/kaggle/input',inputs),('/kaggle/working',tmp_path/'working')]:
        setup=setup.replace(f'Path({old!r})',f'Path({str(new)!r})')
    setup=setup.replace('RESUME_INPUT=None',f'RESUME_INPUT=Path({str(previous)!r})')
    cwd=Path.cwd();paths=list(sys.path);environment=dict(os.environ)
    try:exec(compile(setup,'<DG notebook restore>','exec'),{})
    finally:
        os.chdir(cwd);sys.path[:]=paths;os.environ.clear();os.environ.update(environment)
    assert (work/'source.zip').is_file() and (repo/'scripts/run_dg.py').is_file()
