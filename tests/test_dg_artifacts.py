from copy import deepcopy
import json

import pytest

from src.experiments import dg_artifacts as artifacts
from src.experiments.dg import compose_dg
from src.experiments.protocol import DG_PROTOCOL
from src.data.preparation import write_manifest
from src.utils.artifacts import sha256_file,write_json


def artifact(tmp_path):
    root=tmp_path/'work';(root/'data').mkdir(parents=True)
    for split,count in [('train',2975),('val',500)]:
        rows=[dict(image=str(tmp_path/'images'/split/f'{i}.png'),mask=str(tmp_path/'masks'/split/f'{i}.png'),
                   domain='cityscapes',split=split) for i in range(count)]
        write_manifest(rows,root/f'data/cityscapes_{split}.csv')
    cfg=compose_dg('segformer_b2','baseline',dict(source_train_manifest=str(root/'data/cityscapes_train.csv'),
        source_monitor_manifest=str(root/'data/cityscapes_val.csv')),root/'run')
    for slot in ['initial','best','last']:(root/'run'/slot).mkdir(parents=True)
    (root/'run/initial/model.safetensors').write_bytes(b'fixture initial, not real weights')
    (root/'run/best/model.safetensors').write_bytes(b'fixture selected, not real weights')
    write_json(root/'run/best/config.json',{'fixture':'not a model'})
    write_json(root/'run/config.json',cfg)
    write_json(root/'run/last/checkpoint.json',dict(epoch=20,best_epoch=20))
    write_json(root/'run/best/checkpoint.json',dict(epoch=20))
    for name in ['history.jsonl','resources.jsonl']:
        (root/'run'/name).write_text(''.join(json.dumps(dict(epoch=e))+'\n' for e in range(1,21)))
    files=['data/cityscapes_train.csv','data/cityscapes_val.csv']
    write_json(root/'dg_plan.json',dict(experiment_protocol=DG_PROTOCOL,target_data_access='none',model='segformer_b2',
        arm='baseline',source_hashes={'fixture.py':'hash'},frozen_files={n:sha256_file(root/n) for n in files}))
    write_json(root/'frozen_candidate.json',dict(protocol=DG_PROTOCOL,target_training=False,completed_epochs=20,
        plan_sha256=sha256_file(root/'dg_plan.json'),training_config_sha256=sha256_file(root/'run/config.json'),
        model_config_sha256=sha256_file(root/'run/best/config.json'),weights_sha256=sha256_file(root/'run/best/model.safetensors'),
        initialization=dict(weights_sha256=sha256_file(root/'run/initial/model.safetensors')),model='segformer_b2',arm='baseline'))
    return root


def test_inspection_binds_checkpoint_config_source_splits_and_rejects_changes(tmp_path):
    root=artifact(tmp_path)
    info=artifacts.inspect_candidate(root)
    assert info['id']=='segformer_b2:baseline'
    (root/'run/best/model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):artifacts.inspect_candidate(root)


def cohort_candidates():
    result={}
    for name in artifacts.PRIMARY_IDS:
        model,arm=name.split(':')
        result[name]=dict(id=name,workspace=name,candidate_sha256='candidate',weights_sha256='weights',
            initial_sha256=model,source_ids={'train':'same','val':'same'},seed=42,source_hashes={'code':'same'},
            train={'epochs':20},processing={'image_size':[512,1024]},model=model,arm=arm,
            dg=dict(pair_probability=0. if arm=='baseline' else .5,source_style=None if arm=='baseline' else {'same':'style'},
                    consistency_weight=.1 if arm=='consistency' else 0.))
    return result


def test_cohort_requires_five_runs_and_rejects_initialization_or_recipe_mismatch(tmp_path,monkeypatch):
    candidates=cohort_candidates()
    monkeypatch.setattr(artifacts,'inspect_candidate',lambda key:deepcopy(candidates[key]))
    with pytest.raises(ValueError,match='all5'):
        artifacts.seal_cohort(['segformer_b2:baseline'],tmp_path/'missing.json')
    candidates['segformer_b2:consistency']['initial_sha256']='different'
    with pytest.raises(ValueError,match='initialization'):
        artifacts.seal_cohort(list(candidates),tmp_path/'different.json')
    candidates['segformer_b2:consistency']['initial_sha256']='segformer_b2'
    cohort=tmp_path/'cohort.json';artifacts.seal_cohort(list(candidates),cohort)
    assert artifacts.resolve_frozen(cohort,'segformer_b2:baseline')['arm']=='baseline'
    candidates['segformer_b2:baseline']['candidate_sha256']='changed'
    with pytest.raises(ValueError,match='after cohort freeze'):artifacts.resolve_frozen(cohort,'segformer_b2:baseline')
    with pytest.raises(ValueError,match='Preserve'):artifacts.seal_cohort(list(candidates),cohort)


def test_target_evaluation_cannot_start_without_frozen_cohort(tmp_path,monkeypatch):
    from scripts.evaluate_dg import evaluate_frozen
    from PIL import Image
    monkeypatch.setattr(Image,'open',lambda *a,**k:pytest.fail('Target images opened before freeze'))
    with pytest.raises(FileNotFoundError):
        evaluate_frozen(tmp_path/'missing.json','segformer_b2:baseline',tmp_path/'ACDC.csv',tmp_path/'out','official-acdc')
    assert not (tmp_path/'out').exists()
