import json
from pathlib import Path

import pytest

from scripts import run_dg_suite as suite
from src.utils.artifacts import write_json


@pytest.fixture
def source(monkeypatch):
    def prepare_data(city,out):
        out.mkdir()
        for split in ['train','val']:(out/f'cityscapes_{split}.csv').write_text('source fixture')
    def prepare_arm(city,out,seed,model,arm,p,w,prepared_paths):
        out.mkdir(parents=True,exist_ok=True)
        assert Path(prepared_paths['source_train_manifest']).name=='cityscapes_train.csv'
    monkeypatch.setattr(suite,'prepare_cityscapes',prepare_data)
    monkeypatch.setattr(suite,'prepare',prepare_arm)


def profile(model):
    def call(workspace,out):
        write_json(out,dict(status='PASS',model=model,target_data_access='none',pilot_weights_discarded=True,
            pair_probability_used_for_forecast=.5,consistency_weight_used_for_paired_pilot=.1,
            estimated_20epoch_hours_excluding_checkpoint_zip_setup=dict(baseline=1.,aug=1.5,consistency=1.5)))
    return call


@pytest.mark.parametrize('model',['segformer_b2','deeplabv3plus_resnet50'])
def test_suite_profiles_once_runs_all_arms_and_resumes_without_retraining(tmp_path,monkeypatch,source,model):
    root=tmp_path/'work';trained=[];profiles=[];exports=[];paused=[False]
    def pilot(workspace,out):profiles.append(True);profile(model)(workspace,out)
    def runner(city,output,archive,seed,hours,only,actual_model,arm,p,w,prepared_paths,on_checkpoint):
        assert actual_model==model and 0<hours<=10 and not only and p==.5 and w==.1
        output=Path(output);run=output/'run'
        last=run/'last/checkpoint.json'
        saved=json.loads(last.read_text())['epoch'] if last.exists() else 0
        if saved==20:
            write_json(output/'status.json',dict(status='training_complete'));return
        trained.append((arm,saved))
        epoch=20
        if arm=='aug' and not paused[0]:epoch=4;paused[0]=True
        for slot in ['last','best']:
            (run/slot).mkdir(parents=True,exist_ok=True)
            write_json(run/slot/'checkpoint.json',dict(epoch=epoch))
        history=[dict(epoch=i,val=dict(miou=.2)) for i in range(1,epoch+1)]
        usage=[dict(epoch=i) for i in range(1,epoch+1)]
        for name,rows in [('history.jsonl',history),('resources.jsonl',usage)]:
            (run/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        if epoch==20:write_json(output/'frozen_candidate.json',dict(model=model,arm=arm))
        write_json(output/'status.json',dict(status='training_complete' if epoch==20 else 'session_checkpoint_ready'))
        on_checkpoint()
    kwargs=dict(model=model,runner=runner,profiler=pilot,clock=lambda:0,exporter=lambda *args:exports.append(args))
    first=suite.run_suite(tmp_path/'city',root,tmp_path/'result.zip',**kwargs)
    assert first['status']=='session_checkpoint_ready' and first['completed_arms']==1
    second=suite.run_suite(tmp_path/'city',root,tmp_path/'result.zip',**kwargs)
    assert second['status']=='suite_complete' and second['completed_arms']==3
    assert trained==[('baseline',0),('aug',0),('aug',4),('consistency',0)]
    suite.run_suite(tmp_path/'city',root,tmp_path/'result.zip',**kwargs)
    assert len(trained)==4 and len(profiles)==1 and len(exports)>=6
    assert [j['arm'] for j in second['jobs']]==['baseline','aug','consistency']


def test_suite_refuses_changed_source_manifest_before_training(tmp_path,source):
    root=tmp_path/'work';suite.prepare_suite(tmp_path/'city',root,'segformer_b2')
    (root/'data/cityscapes_val.csv').write_text('changed')
    with pytest.raises(ValueError,match='manifest changed'):
        suite.prepare_suite(tmp_path/'city',root,'segformer_b2')
    with pytest.raises(ValueError,match='original suite'):
        suite.prepare_suite(tmp_path/'city',root,'deeplabv3plus_resnet50')


def test_invalid_profile_prevents_training_and_exports_diagnostics(tmp_path,source):
    exports=[]
    def invalid(workspace,out):write_json(out,dict(status='FAIL'))
    with pytest.raises(ValueError,match='GPU profile'):
        suite.run_suite(tmp_path/'city',tmp_path/'work',tmp_path/'result.zip','segformer_b2',
            runner=lambda *a,**k:pytest.fail('Training started after failed profile'),profiler=invalid,
            clock=lambda:0,exporter=lambda *args:exports.append(args))
    assert exports and (tmp_path/'work/failure.json').exists()
    status=json.loads((tmp_path/'work/suite_status.json').read_text())
    assert status['status']=='failed_preserve_outputs'


def test_python_failure_with_partial_logs_still_preserves_suite(tmp_path,source):
    exports=[]
    def failed(city,output,*args,**kwargs):
        run=Path(output)/'run';run.mkdir(parents=True)
        (run/'history.jsonl').write_text('{"epoch":')
        raise RuntimeError('simulated training interruption')
    with pytest.raises(RuntimeError,match='interruption'):
        suite.run_suite(tmp_path/'city',tmp_path/'work',tmp_path/'result.zip','segformer_b2',
            runner=failed,profiler=profile('segformer_b2'),clock=lambda:0,
            exporter=lambda *args:exports.append(args))
    assert exports and 'recovery_required' in json.loads((tmp_path/'work/suite_status.json').read_text())


def test_short_session_preserves_setup_without_starting_profile_or_training(tmp_path,source):
    exports=[]
    status=suite.run_suite(tmp_path/'city',tmp_path/'work',tmp_path/'result.zip','segformer_b2',
        session_hours=.1,clock=lambda:0,exporter=lambda *args:exports.append(args),
        runner=lambda *a,**k:pytest.fail('Training started without sufficient session time'),
        profiler=lambda *a:pytest.fail('Profile started without sufficient session time'))
    assert status['status']=='session_checkpoint_ready' and status['completed_arms']==0 and exports


def test_sealed_profile_cannot_be_silently_changed_or_recomputed(tmp_path):
    plan=dict(model='segformer_b2',pair_probability=.5,consistency_weight=.1)
    suite.profile_once(tmp_path,plan,profile('segformer_b2'))
    path=tmp_path/'gpu_profile.json';report=json.loads(path.read_text())
    report['estimated_20epoch_hours_excluding_checkpoint_zip_setup']['aug']=9.
    write_json(path,report)
    with pytest.raises(ValueError,match='profile changed'):suite.profile_once(tmp_path,plan)
    path.unlink()
    with pytest.raises(ValueError,match='Missing sealed'):suite.profile_once(tmp_path,plan)
