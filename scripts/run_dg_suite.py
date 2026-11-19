"""One model, automatic profile -> baseline -> aug -> consistency, resumable."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_dg import prepare_cityscapes,prepare,run as run_one
from src.experiments.run_support import fingerprints,snapshot_source
from src.experiments.packaging import export_workspace
from src.experiments.protocol import DG_PROTOCOL
from src.models.registry import MODEL_NAMES
from src.training.dg import ARMS
from src.utils.artifacts import records,sha256_file,write_json

SCHEMA='automatic_dg_model_suite_v1'


def prepare_suite(city,root,model,seed=42):
    if model not in MODEL_NAMES:raise ValueError('Unknown DG model')
    city,root=Path(city).resolve(),Path(root).resolve()
    identity=dict(schema=SCHEMA,protocol=DG_PROTOCOL,model=model,seed=seed,city_root=str(city),
                  source_hashes=fingerprints(),arms=list(ARMS),epochs_per_arm=20,
                  pair_probability=.5,consistency_weight=.1,target_data_access='none')
    file=root/'suite_plan.json'
    if file.exists():
        saved=json.loads(file.read_text())
        if any(saved.get(k)!=v for k,v in identity.items()):
            raise ValueError('Resume the original suite model/code/data mount/seed')
        for name,digest in saved['data_hashes'].items():
            if sha256_file(root/name)!=digest:raise ValueError('Suite source manifest changed')
        if not (root/'source.zip').is_file():snapshot_source(root/'source.zip',identity['source_hashes'])
        return saved
    if root.exists() and any(root.iterdir()):raise ValueError('Preserve partial setup; use a fresh suite directory')
    root.mkdir(parents=True,exist_ok=True)
    prepare_cityscapes(city,root/'data')
    snapshot_source(root/'source.zip',identity['source_hashes'])
    identity['data_hashes']={f'data/cityscapes_{split}.csv':sha256_file(root/f'data/cityscapes_{split}.csv')
                             for split in ['train','val']}
    write_json(file,identity)
    return identity


def progress(root,plan):
    root=Path(root);jobs=[]
    for arm in plan['arms']:
        path=root/'jobs'/arm
        file=path/'run/last/checkpoint.json'
        epoch=json.loads(file.read_text())['epoch'] if file.is_file() else 0
        history=records(path/'run/history.jsonl');usage=records(path/'run/resources.jsonl')
        if ([r['epoch'] for r in history[:epoch]]!=list(range(1,epoch+1)) or
                [r['epoch'] for r in usage[:epoch]]!=list(range(1,epoch+1))):
            raise ValueError('Checkpoint and committed logs disagree; preserve outputs for recovery')
        done=(path/'frozen_candidate.json').is_file() and epoch==20
        row=dict(model=plan['model'],arm=arm,completed_epochs=epoch,planned_epochs=20,
                 status='complete' if done else 'in_progress' if epoch else 'pending')
        if epoch:
            best=json.loads((path/'run/best/checkpoint.json').read_text())['epoch']
            row.update(best_epoch=best,cityscapes_val_miou=history[best-1]['val']['miou'])
        jobs.append(row)
    return dict(model=plan['model'],schema=SCHEMA,jobs=jobs,
                completed_arms=sum(j['status']=='complete' for j in jobs),total_arms=len(jobs),
                selection='Cityscapes val500 only; no target evaluation in this suite')


def profile_once(root,plan,profiler=None):
    root=Path(root);profile=root/'gpu_profile.json';seal=root/'profile_seal.json'
    if seal.exists() and not profile.exists():
        raise ValueError('Missing sealed GPU profile; restore the complete suite ZIP')
    if profile.exists():
        report=json.loads(profile.read_text())
    else:
        if profiler is None:
            subprocess.run([sys.executable,'-u','-m','scripts.profile_dg','--workspace',str(root/'jobs/baseline'),
                            '--out',str(profile),'--pair-probability',str(plan['pair_probability']),
                            '--consistency-weight',str(plan['consistency_weight'])],check=True)
        else:profiler(root/'jobs/baseline',profile)
        report=json.loads(profile.read_text())
    estimates=report.get('estimated_20epoch_hours_excluding_checkpoint_zip_setup',{})
    if (report.get('status')!='PASS' or report.get('model')!=plan['model'] or set(estimates)!=set(ARMS)
            or report.get('target_data_access')!='none' or report.get('pilot_weights_discarded') is not True
            or not all(isinstance(x,(float,int)) and math.isfinite(x) and x>0 for x in estimates.values())
            or report.get('pair_probability_used_for_forecast')!=plan['pair_probability']
            or report.get('consistency_weight_used_for_paired_pilot')!=plan['consistency_weight']):
        raise ValueError('Invalid/incompatible GPU profile; training has not been admitted')
    expected=dict(profile_sha256=sha256_file(profile),model=plan['model'])
    if seal.exists() and json.loads(seal.read_text())!=expected:raise ValueError('Sealed GPU profile changed')
    write_json(seal,expected)
    return report


def run_suite(city,root,archive,model,seed=42,session_hours=10.,runner=run_one,profiler=None,
              clock=time.monotonic,exporter=export_workspace):
    if not 0<session_hours<=10:raise ValueError('Session window must be in(0,10]hours')
    root,archive=Path(root).resolve(),Path(archive).resolve()
    if archive.is_relative_to(root):raise ValueError('Save the suite ZIP outside the workspace')
    started=clock();deadline=started+session_hours*3600;status='running';plan=None
    def snapshot():
        try:report=progress(root,plan) if plan else dict(completed_arms=0,total_arms=3)
        except Exception as error:report=dict(completed_arms=0,total_arms=3,recovery_required=str(error))
        report.update(status=status,elapsed_seconds_this_invocation_before_export=clock()-started)
        write_json(root/'suite_status.json',report)
        exporter(root,archive)
        print(f'SUITE ZIP ready: {archive}; {report["completed_arms"]}/3 arms complete',flush=True)
    try:
        plan=prepare_suite(city,root,model,seed)
        paths=dict(source_train_manifest=str(root/'data/cityscapes_train.csv'),
                   source_monitor_manifest=str(root/'data/cityscapes_val.csv'))
        # Freeze all arm configs at setup. They all point to the shared converted source masks.
        for arm in ARMS:
            prepare(city,root/'jobs'/arm,seed,model,arm,plan['pair_probability'],
                    plan['consistency_weight'],prepared_paths=paths)
        if not (root/'gpu_profile.json').is_file() and deadline-clock()<20*60:
            status='session_checkpoint_ready'
        else:
            report=profile_once(root,plan,profiler)
            print('Automatic profile PASS. Estimated train+val hours (excludes ZIP/setup):',
                  report['estimated_20epoch_hours_excluding_checkpoint_zip_setup'],flush=True)
            print('All3 arms remain scheduled; this model-only suite cannot read the shared Kaggle quota.',flush=True)
            for arm in ARMS:
                remaining=(deadline-clock())/3600
                if remaining<=0:status='session_checkpoint_ready';break
                runner(city,root/'jobs'/arm,archive,seed,remaining,False,model,arm,
                       plan['pair_probability'],plan['consistency_weight'],
                       prepared_paths=paths,on_checkpoint=snapshot)
                job=json.loads((root/'jobs'/arm/'status.json').read_text())
                if job['status']!='training_complete':status='session_checkpoint_ready';break
            else:status='suite_complete'
    except Exception as error:
        status='failed_preserve_outputs';write_json(root/'failure.json',dict(error=str(error)));raise
    finally:
        if root.exists():snapshot()
    return json.loads((root/'suite_status.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['city','root','archive']:p.add_argument('--'+name,required=True)
    p.add_argument('--model',choices=MODEL_NAMES,required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--session-hours',type=float,default=10.)
    a=p.parse_args();run_suite(a.city,a.root,a.archive,a.model,a.seed,a.session_hours)
