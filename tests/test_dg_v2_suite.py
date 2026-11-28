import json
from pathlib import Path

import pytest
import torch

from scripts import run_dg_v2_suite as suite


@pytest.mark.parametrize('interrupt',[False,True])
@pytest.mark.parametrize('group',['all','baseline_weights','weather_weights'])
def test_v2_queue_runs_selected_arms_exports_and_skips_committed_epochs(tmp_path,monkeypatch,interrupt,group):
    root=tmp_path/'suite';calls=[];exports=[]
    def prepare(*args):
        plan=dict(arms=list(suite.GROUPS[group]),group=group)
        root.mkdir(exist_ok=True)
        (root/'suite_plan.json').write_text(json.dumps(plan))
        for arm in plan['arms']:
            p=root/'jobs'/arm;p.mkdir(parents=True,exist_ok=True)
            (p/'train.yaml').write_text('fixture')
        return plan
    def run_command(argv,check):
        cfg=Path(argv[argv.index('--config')+1]);arm=cfg.parent.name
        epoch=int(argv[argv.index('--stop-after-epoch')+1]);calls.append((arm,epoch))
        if interrupt and arm==suite.GROUPS[group][1] and epoch==2:raise RuntimeError('simulated failure')
        run=cfg.parent/'run';(run/'last').mkdir(parents=True,exist_ok=True)
        (run/'last/checkpoint.json').write_text(json.dumps({'epoch':epoch,'best_epoch':epoch}))
        (run/'best').mkdir(exist_ok=True);(run/'best/checkpoint.json').write_text(json.dumps({'epoch':epoch}))
        row=dict(epoch=epoch,selection_score=.01*epoch,dev={k:{'miou':.01*epoch} for k in ['clean','low_light','noise','haze']})
        with (run/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        with (run/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,epoch_seconds=.01))+'\n')
        (run/'initialization.json').write_text(json.dumps({'weights_sha256':'same'}))
    import src.training.checkpoints as checkpoints
    monkeypatch.setattr(suite,'prepare',prepare);monkeypatch.setattr(suite.subprocess,'run',run_command)
    monkeypatch.setattr(suite,'export_workspace',lambda *args:exports.append(args))
    monkeypatch.setattr(suite,'committed_state',lambda *args:None)
    monkeypatch.setattr(suite,'plot_history',lambda *args:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(checkpoints,'recover_run',lambda p:json.loads((p/'last/checkpoint.json').read_text())['epoch'] if (p/'last/checkpoint.json').exists() else 0)
    if interrupt:
        with pytest.raises(RuntimeError,match='simulated failure'):
            suite.run(tmp_path/'city',root,tmp_path/'result.zip',group=group)
        assert exports and json.loads((root/'suite_status.json').read_text())['status']=='session_checkpoint_ready'
        interrupt=False
        result=suite.run(tmp_path/'city',root,tmp_path/'result.zip',group=group)
        assert result['status']=='suite_complete' and len(calls)==len(suite.GROUPS[group])*20+1
    else:
        result=suite.run(tmp_path/'city',root,tmp_path/'result.zip',group=group)
        count=len(suite.GROUPS[group])*20
        assert len(calls)==count and result['status']=='suite_complete'
        suite.run(tmp_path/'city',root,tmp_path/'result.zip',group=group)
        assert len(calls)==count and len(exports)==count+2
    assert {arm for arm,epoch in calls}==set(suite.GROUPS[group])


def test_v2_status_never_counts_uncommitted_history_as_completed(tmp_path):
    root=tmp_path/'suite'
    for arm in suite.ARMS:
        run=root/'jobs'/arm/'run';run.mkdir(parents=True)
        (run/'history.jsonl').write_text(''.join(json.dumps({'epoch':i})+'\n' for i in range(1,21)))
    (root/'suite_plan.json').write_text(json.dumps(dict(arms=list(suite.ARMS),group='all')))
    status=suite.report(root)
    assert status['status']=='session_checkpoint_ready' and all(r['epochs']==0 for r in status['jobs'])


@pytest.mark.parametrize('group',['baseline_weights','weather_weights'])
def test_generated_v2_setup_accepts_extracted_new_code_and_train_only_city(tmp_path,monkeypatch,group):
    from scripts.build_dg_v2_notebook import build,NOTEBOOKS,ROOT
    from scripts.kaggle_bootstrap import extract_safe
    import os
    import sys
    _,bundle=build();notebook=ROOT/'notebooks'/NOTEBOOKS[group][0]
    inputs=tmp_path/'input';extract_safe(bundle,inputs/'code')
    city=inputs/'city/Cityscape'
    for folder in ['leftImg8bit/train','gtFine/train']:(city/folder).mkdir(parents=True)
    cells=json.loads(notebook.read_text(encoding='utf-8'))['cells']
    text=''.join(cells[1]['source'])+'\n'+''.join(cells[2]['source'])
    text=text.replace("Path('/kaggle/input')",f'Path({str(inputs)!r})').replace("Path('/kaggle/working')",f'Path({str(tmp_path/"working")!r})')
    cwd=Path.cwd();paths=list(sys.path)
    try:
        ns={};exec(text,ns)
        assert ns['CITYSCAPES_ROOT']==city
        assert 'ACDC_ROOT' not in ns and ns['MODEL']=='segformer_b2'
        assert ns['GROUP']==group
        assert (ns['REPO']/'src/training/dg_v2.py').is_file()
        # Rerunning the setup alone keeps existing output files intact.
        ns['WORK'].mkdir();(ns['WORK']/'keep.txt').write_text('checkpoint sentinel')
        original_started=ns['STARTED']
        exec(''.join(cells[2]['source']).replace("Path('/kaggle/input')",f'Path({str(inputs)!r})')
            .replace("Path('/kaggle/working')",f'Path({str(tmp_path/"working")!r})'),ns)
        assert (ns['WORK']/'keep.txt').read_text()=='checkpoint sentinel'
        # Run All repeats the path-settings cell too; auto-find None is unchanged input.
        exec(text,ns)
        assert ns['STARTED']==original_started and (ns['WORK']/'keep.txt').read_text()=='checkpoint sentinel'
    finally:os.chdir(cwd);sys.path[:]=paths


@pytest.mark.parametrize('expanded',[False,True])
@pytest.mark.parametrize('wrong_group',[False,True])
def test_generated_resume_restores_frozen_source_or_rejects_wrong_pair(tmp_path,expanded,wrong_group):
    import os
    import sys
    from scripts.build_dg_v2_notebook import build
    from scripts.kaggle_bootstrap import extract_safe
    from src.experiments.dg_v2 import source_fingerprints
    from src.experiments.run_support import snapshot_source
    notebook,_=build();previous=tmp_path/'previous';previous.mkdir()
    city=tmp_path/'input/city'
    for folder in ['leftImg8bit/train','gtFine/train']:(city/folder).mkdir(parents=True)
    plan=dict(protocol='cityscapes_groupdev_dg_v2',model='segformer_b2',seed=42,
        group='weather_weights' if wrong_group else 'baseline_weights',city_root=str(city),source_hashes=source_fingerprints())
    (previous/'suite_plan.json').write_text(json.dumps(plan))
    snapshot_source(previous/'source.zip',plan['source_hashes'])
    if expanded:
        extract_safe(previous/'source.zip',previous/'source');(previous/'source.zip').unlink()
    cells=json.loads(notebook.read_text(encoding='utf-8'))['cells']
    text=''.join(cells[1]['source'])+'\n'+''.join(cells[2]['source'])
    text=text.replace('RESUME_INPUT=None',f'RESUME_INPUT=Path({str(previous)!r})')
    text=text.replace("Path('/kaggle/input')",f'Path({str(tmp_path/"input")!r})').replace("Path('/kaggle/working')",f'Path({str(tmp_path/"working")!r})')
    cwd=Path.cwd();paths=list(sys.path)
    try:
        ns={}
        if wrong_group:
            with pytest.raises(AssertionError,match='Sai nhóm'):exec(text,ns)
        else:
            exec(text,ns)
            assert ns['CITYSCAPES_ROOT']==city
            assert (ns['REPO']/'configs/dg_v2/splits/fit.csv').is_file()
            assert (ns['WORK']/'suite_plan.json').read_text()==(previous/'suite_plan.json').read_text()
    finally:os.chdir(cwd);sys.path[:]=paths


def test_group_report_exports_readable_table_and_committed_curves(tmp_path):
    root=tmp_path/'suite';root.mkdir()
    (root/'suite_plan.json').write_text(json.dumps(dict(group='baseline_weights',arms=list(suite.GROUPS['baseline_weights']))))
    for arm in suite.GROUPS['baseline_weights']:
        run=root/'jobs'/arm/'run';(run/'last').mkdir(parents=True);(run/'best').mkdir()
        (run/'last/checkpoint.json').write_text(json.dumps(dict(epoch=1,best_epoch=1)))
        (run/'best/checkpoint.json').write_text(json.dumps(dict(epoch=1)))
        row=dict(epoch=1,selection_score=.4,train_loss=1.5,dev={k:dict(miou=.4) for k in ['clean','low_light','noise','haze']})
        (run/'history.jsonl').write_text(json.dumps(row)+'\n'+json.dumps(dict(epoch=2))+'\n')
        (run/'resources.jsonl').write_text(json.dumps(dict(epoch=1))+'\n')
    status=suite.report(root)
    assert all(r['epochs']==1 for r in status['jobs'])
    assert (root/'learning_curves.png').stat().st_size>1000
    assert 'class_weights' in (root/'suite_summary.csv').read_text()
