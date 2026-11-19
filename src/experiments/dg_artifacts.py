"""Freeze the experiment cohort before any target evaluation can be requested."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from src.experiments.protocol import DG_PROTOCOL, require_training_permission
from src.utils.artifacts import sha256_file, records, write_json

PRIMARY_IDS = {'segformer_b2:baseline', 'segformer_b2:aug', 'segformer_b2:consistency',
               'deeplabv3plus_resnet50:baseline', 'deeplabv3plus_resnet50:aug',
               'deeplabv3plus_resnet50:consistency'}


def inspect_candidate(workspace):
    root=Path(workspace).resolve()
    frozen=json.loads((root/'frozen_candidate.json').read_text())
    plan=json.loads((root/'dg_plan.json').read_text())
    if (frozen.get('protocol')!=DG_PROTOCOL or frozen.get('target_training') is not False
            or frozen.get('completed_epochs')!=20 or plan.get('experiment_protocol')!=DG_PROTOCOL
            or plan.get('target_data_access')!='none'):
        raise ValueError('Only complete source-only DG candidates can enter the cohort')
    checks={'dg_plan.json':frozen['plan_sha256'], 'run/config.json':frozen['training_config_sha256'],
            'run/best/config.json':frozen['model_config_sha256'],
            'run/best/model.safetensors':frozen['weights_sha256'],
            'run/initial/model.safetensors':frozen['initialization']['weights_sha256'],
            **plan['frozen_files']}
    for name,digest in checks.items():
        if sha256_file(root/name)!=digest:raise ValueError('Frozen candidate artifact changed: '+name)
    cfg=json.loads((root/'run/config.json').read_text())
    if (cfg['model']['name']!=frozen['model'] or cfg.get('dg',{}).get('arm')!=frozen['arm']
            or plan['model']!=frozen['model'] or plan['arm']!=frozen['arm']):
        raise ValueError('Model/arm identity mismatch')
    history=records(root/'run/history.jsonl');usage=records(root/'run/resources.jsonl')
    if [r['epoch'] for r in history]!=list(range(1,21)) or [r['epoch'] for r in usage]!=list(range(1,21)):
        raise ValueError('A DG candidate must contain all20 committed epochs')
    last=json.loads((root/'run/last/checkpoint.json').read_text())
    best=json.loads((root/'run/best/checkpoint.json').read_text())
    if last['epoch']!=20 or last['best_epoch']!=best['epoch']:
        raise ValueError('Checkpoint selection metadata mismatch')
    # Validate archived manifests even if this evaluator runs under a new mount.
    bound=deepcopy(cfg)
    bound['data']['train_manifest']=str(root/'data/cityscapes_train.csv')
    bound['data']['val_manifest']=str(root/'data/cityscapes_val.csv')
    require_training_permission(bound)
    from src.data.validation import read_manifest
    ids={}
    for role in ['train','val']:
        rows=read_manifest(root/f'data/cityscapes_{role}.csv')
        logical=sorted((r['image'].replace('\\','/').split('/')[-1],
                        r['mask'].replace('\\','/').split('/')[-1],r['domain'],r['split']) for r in rows)
        ids[role]=hashlib.sha256(json.dumps(logical).encode()).hexdigest()
    return dict(id=f"{frozen['model']}:{frozen['arm']}",workspace=str(root),
                candidate_sha256=sha256_file(root/'frozen_candidate.json'),
                weights_sha256=frozen['weights_sha256'],initial_sha256=frozen['initialization']['weights_sha256'],
                source_ids=ids,seed=cfg['seed'],source_hashes=plan['source_hashes'],
                train=cfg['train'],processing={k:v for k,v in cfg['data'].items() if not k.endswith('_manifest')},
                dg=cfg['dg'],model=frozen['model'],arm=frozen['arm'])


def seal_cohort(workspaces,destination):
    destination=Path(destination)
    if destination.exists():raise ValueError('Preserve the existing frozen cohort')
    candidates=[inspect_candidate(p) for p in workspaces]
    ids=[c['id'] for c in candidates]
    if len(set(ids))!=len(ids) or not PRIMARY_IDS.issubset(ids):
        raise ValueError('Freeze all6 primary model/arm candidates before target evaluation')
    if set(ids)-PRIMARY_IDS:
        raise ValueError('Unexpected candidate outside the predeclared matrix')
    first=candidates[0]
    for c in candidates[1:]:
        if any(c[k]!=first[k] for k in ['source_ids','seed','source_hashes','processing']):
            raise ValueError('Cohort must share source splits, processing, seed and code')
    for model in ['segformer_b2','deeplabv3plus_resnet50']:
        group=[c for c in candidates if c['model']==model]
        if len({c['initial_sha256'] for c in group})!=1 or any(c['train']!=group[0]['train'] for c in group):
            raise ValueError('Within a model all arms must share initialization and training recipe')
    styled=[c for c in candidates if c['arm']!='baseline']
    for c in styled[1:]:
        if any(c['dg'][k]!=styled[0]['dg'][k] for k in ['pair_probability','source_style']):
            raise ValueError('Styled arms must share source style and pair schedule')
    full=[c for c in candidates if c['arm']=='consistency']
    if full[0]['dg']['consistency_weight']!=full[1]['dg']['consistency_weight']:
        raise ValueError('Full DG must share the declared consistency weight across models')
    payload=dict(protocol=DG_PROTOCOL,status='cohort_frozen_before_target_evaluation',candidates=candidates,
                 target_feedback_limit='Prior ACDC development feedback exists; not a pristine independent test')
    write_json(destination,payload)
    return payload


def resolve_frozen(cohort,run_id):
    sealed=json.loads(Path(cohort).read_text())
    if sealed.get('status')!='cohort_frozen_before_target_evaluation' or not PRIMARY_IDS.issubset(
            c['id'] for c in sealed.get('candidates',[])):
        raise ValueError('An entire frozen DG cohort is required before evaluation')
    saved=next((c for c in sealed['candidates'] if c['id']==run_id),None)
    if saved is None:raise ValueError('Candidate is outside the frozen cohort')
    current=inspect_candidate(saved['workspace'])
    if current!=saved:raise ValueError('Candidate changed after cohort freeze')
    return current


def seal_suites(suites,destination):
    roots=[Path(p).resolve() for p in suites]
    models=[];jobs=[]
    for root in roots:
        plan=json.loads((root/'suite_plan.json').read_text())
        status=json.loads((root/'suite_status.json').read_text())
        if (plan.get('schema')!='automatic_dg_model_suite_v1' or status.get('status')!='suite_complete'
                or set(plan.get('arms',[]))!={'baseline','aug','consistency'}):
            raise ValueError('Both automatic model suites must complete all3 arms before cohort freeze')
        models.append(plan['model']);jobs.extend(root/'jobs'/arm for arm in plan['arms'])
    if sorted(models)!=['deeplabv3plus_resnet50','segformer_b2']:
        raise ValueError('Provide one complete B2 suite and one complete DeepLab suite')
    return seal_cohort(jobs,destination)
