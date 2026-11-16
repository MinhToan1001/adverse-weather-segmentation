import ast
import json
from pathlib import Path

import pytest

from scripts import run_research_campaign as campaign
from scripts import build_research_notebooks as builder


def test_six_fresh_jobs_use_each_model_source_then_adaptation_and_twenty_epochs():
    jobs=campaign.jobs()
    assert [j['model'] for j in jobs]==['segformer_b0']*2+['segformer_b2']*2+['deeplabv3plus_resnet50']*2
    assert [j['stage'] for j in jobs]==['source','adapt']*3
    assert all(j['epochs']==20 and j['loss']=='ce' for j in jobs)


def test_campaign_rejects_old_jobs_and_changed_data(tmp_path):
    data=tmp_path/'data.json';data.write_text('{}')
    old=tmp_path/'old/jobs';old.mkdir(parents=True);(old/'old_model').mkdir()
    with pytest.raises(ValueError,match='empty jobs'):campaign.prepare_campaign(old.parent,data)
    root=tmp_path/'fresh'
    plan=campaign.prepare_campaign(root,data)
    assert plan['old_project_checkpoints']=='excluded' and plan['initialization']=='imagenet_encoder'
    data.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='original campaign'):campaign.prepare_campaign(root,data)


def test_campaign_pauses_then_continues_without_retraining_completed_jobs(tmp_path,monkeypatch):
    data=tmp_path/'data.json';data.write_text('{}')
    root=tmp_path/'work';archive=tmp_path/'results.zip'
    monkeypatch.setattr(campaign,'plot',lambda path:None)
    calls=[];snapshots=[]
    paused=[False]
    def exporter(work,destination):snapshots.append(str(destination))
    def runner(args,on_epoch,deadline):
        output=Path(args.output);run=output/'run'
        previous=run/'last/checkpoint.json'
        if previous.is_file() and json.loads(previous.read_text())['epoch']==20:return
        calls.append((args.model,args.stage,args.initialization,deadline))
        epoch=20
        if not paused[0] and args.model=='segformer_b0' and args.stage=='adapt':epoch=3;paused[0]=True
        if args.stage=='source':assert args.checkpoint is None
        else:assert Path(args.checkpoint)==root/'jobs'/(args.model+'_source')/'run/best'
        for kind in ('last','best'):
            (run/kind).mkdir(parents=True,exist_ok=True)
            (run/kind/'checkpoint.json').write_text(json.dumps(dict(epoch=epoch)))
        history=[dict(epoch=e,val=dict(miou=.2,per_class_iou=[.2]*19),train_loss=2.) for e in range(1,epoch+1)]
        resources=[dict(epoch=e,optimizer_updates=1) for e in range(1,epoch+1)]
        for name,rows in [('history.jsonl',history),('resources.jsonl',resources)]:
            (run/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        on_epoch(epoch)
        (output/'status.json').write_text(json.dumps(dict(status='training_complete' if epoch==20 else 'session_checkpoint_ready')))
    first=campaign.run_campaign(root,data,archive,runner=runner,clock=lambda:0,exporter=exporter)
    assert first['status']=='session_checkpoint_ready' and first['completed_jobs']==1
    second=campaign.run_campaign(root,data,archive,runner=runner,clock=lambda:0,exporter=exporter)
    assert second['status']=='training_complete' and second['completed_jobs']==6
    assert sum(model=='segformer_b0' and stage=='source' for model,stage,_,_ in calls)==1
    assert all(initial=='imagenet_encoder' and deadline==36000 for _,_,initial,deadline in calls)
    assert len(snapshots)>=8


def test_failed_job_with_truncated_logs_still_exports_workspace(tmp_path,monkeypatch):
    data=tmp_path/'data.json';data.write_text('{}')
    monkeypatch.setattr(campaign,'plot',lambda path:None)
    snapshots=[]
    def runner(args,**kwargs):
        run=Path(args.output)/'run';run.mkdir(parents=True)
        (run/'history.jsonl').write_text('{"epoch":')
        raise RuntimeError('training interrupted')
    with pytest.raises(RuntimeError,match='training interrupted'):
        campaign.run_campaign(tmp_path/'work',data,tmp_path/'result.zip',runner=runner,
            exporter=lambda work,path:snapshots.append(path))
    assert snapshots


@pytest.mark.parametrize('model',list(builder.NOTEBOOKS))
def test_each_notebook_parses_and_restores_only_its_own_model(model):
    notebook=json.loads(builder.build(model).read_text(encoding='utf-8'))
    text='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    assert f"MODEL_NAME='{model}'" in text and 'SOURCE_RUN_INPUT=' not in text
    assert 'run_research_campaign' in text and 'RESUME_INPUT=None' in text
    assert "plan['source_hashes']" in text and 'export_workspace(WORK,ARCHIVE)' in text


def test_single_model_plan_has_two_jobs_and_refuses_cross_model_resume(tmp_path):
    data=tmp_path/'data.json';data.write_text('{}')
    root=tmp_path/'b2'
    plan=campaign.prepare_campaign(root,data,models=['segformer_b2'])
    assert [j['model'] for j in plan['jobs']]==['segformer_b2']*2
    assert campaign.progress(root,plan)['total_jobs']==2
    with pytest.raises(ValueError,match='original campaign'):
        campaign.prepare_campaign(root,data,models=['segformer_b0'])
    with pytest.raises(ValueError,match='unique supported'):
        campaign.jobs(['segformer_b2','segformer_b2'])
