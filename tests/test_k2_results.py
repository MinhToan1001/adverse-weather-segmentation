"""Artifact audits reject tampering and confounded controls without loading pickle."""
import hashlib
import io
import json
import subprocess
import sys
import zipfile

import pytest
import torch
from safetensors.torch import save
import yaml

from scripts.k2_results import ARMS, compare, export_share, summarize
from scripts.run_k2 import compose, train_epoch, publish_review
from src.experiments.packaging import export_workspace


def digest(content):
    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def make_result(root, arm='k2', changed_core=False, changed_init=False, lr=None, complete=True,
                foreign_output=False, crlf=False):
    root.mkdir(parents=True)
    paths = {key: str(root/'data'/name) for key, name in [
        ('source_train_manifest','cityscapes_train.csv'),('source_monitor_manifest','cityscapes_val.csv')]}
    cfg = compose(arm, paths, root/'run', epochs=1)
    if foreign_output:
        cfg['output_dir']='/kaggle/working/k2_'+arm+'_seed42/run'
    if lr is not None:
        cfg['train']['lr'] = lr
    (root/'train.yaml').write_text(yaml.safe_dump(cfg))
    for split, count in [('train',2975),('val',500)]:
        p = root/'data'/f'cityscapes_{split}.csv'
        p.parent.mkdir(exist_ok=True)
        p.write_text('image,mask,domain,split\n' + ''.join(
            f'/input/leftImg8bit/{split}/city/{i}_leftImg8bit.png,'
            f'/work/masks/{split}/city/{i}_gtFine_labelTrainIds.png,cityscapes,{split}\n' for i in range(count)))
    clean = arm in ('baseline','film_clean')
    write_json(root/'weather_preflight.json', dict(conditions=['clean'] if clean else ['clean','night','rain','snow'],fog_enabled=False))
    core = {n: b'fixed algorithm\n' for n in [
        'src/training/train.py','src/models/registry.py','src/models/proposed/weatherseg_b0.py',
        'src/data/augmentations/weather/generator.py','configs/proposed/k2_film_b0.yaml']}
    if changed_core:
        core['src/training/train.py'] = b'changed algorithm'
    if crlf:
        core = {n: b.replace(b'\n', b'\r\n') for n,b in core.items()}
    source = io.BytesIO()
    with zipfile.ZipFile(source,'w') as z:
        for n, b in core.items(): z.writestr(n,b)
    (root/'source.zip').write_bytes(source.getvalue())
    frozen_files = ['train.yaml','data/cityscapes_train.csv','data/cityscapes_val.csv','weather_preflight.json']
    plan = dict(protocol='weatherseg_k2_source_v1',arm=arm,seed=42,epochs=1,batch_size=2,
        source_hashes={n:digest(b) for n,b in core.items()},
        frozen_files={n:digest((root/n).read_bytes()) for n in frozen_files})
    write_json(root/'k2_plan.json',plan)
    weights = save({'segformer.weight':torch.ones(2),'decode_head.weight':torch.zeros(2)})
    initial = {'segformer.weight':torch.ones(2)*(2 if changed_init else 1),'decode_head.weight':torch.zeros(2)}
    if arm not in ('baseline','aug'):
        initial['segformer.encoder.film.0.weight'] = torch.zeros(2)
        initial['condition_head.weight'] = torch.zeros(2)
    (root/'run/initial').mkdir(parents=True)
    initial_bytes = save(initial)
    (root/'run/initial/model.safetensors').write_bytes(initial_bytes)
    write_json(root/'run/initialization.json',dict(weights_sha256=digest(initial_bytes),target_data_access='none'))
    write_json(root/'run/model_report.json',dict(parameters=dict(total=3817720)))
    best = root/'run/best';best.mkdir()
    (best/'model.safetensors').write_bytes(weights)
    write_json(best/'config.json',dict(model_type='segformer'))
    write_json(best/'checkpoint.json',dict(epoch=1,files={
        'model.safetensors':digest(weights),'config.json':digest((best/'config.json').read_bytes())}))
    (root/'run/history.jsonl').write_text(json.dumps(dict(epoch=1,train_loss=.4,val=dict(miou=.6,per_class_iou=[.6]*19)))+'\n')
    (root/'run/resources.jsonl').write_text(json.dumps(dict(epoch=1,optimizer_updates=2,epoch_seconds=10.,max_memory_allocated_bytes=100))+'\n')
    write_json(root/'status.json',dict(status='training_complete' if complete else 'session_checkpoint_ready',completed_epochs=1))
    if complete:
        write_json(root/'frozen_candidate.json',dict(arm=arm,seed=42,completed_epochs=1,target_training=False,
            weights_sha256=digest(weights),config_sha256=digest((best/'config.json').read_bytes()),
            plan_sha256=digest((root/'k2_plan.json').read_bytes())))
    return root


def test_light_export_preserves_full_result_and_checks_seals(tmp_path):
    root = make_result(tmp_path/'run')
    full = tmp_path/'full.zip';export_workspace(root,full)
    before = full.read_bytes()
    report = export_share(root,tmp_path/'share.zip')
    assert report['complete'] and report['best_epoch']==1
    assert summarize(full)['checkpoint_sha256']==report['checkpoint_sha256']
    assert full.read_bytes()==before
    with zipfile.ZipFile(tmp_path/'share.zip') as z:
        assert 'run/best/model.safetensors' in z.namelist()
        assert 'result_summary.json' in z.namelist()
        assert 'run/initial/model.safetensors' not in z.namelist()
        assert not any(n.endswith('training_state.pt') for n in z.namelist())
    assert summarize(tmp_path/'share.zip')['complete']
    with pytest.raises(ValueError,match='Missing result artifact'):
        compare([full,tmp_path/'share.zip'])
    (root/'run/best/model.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checkpoint'):
        summarize(root)


def test_five_matched_arms_compare_from_zip_and_expanded_folders(tmp_path):
    runs = [make_result(tmp_path/arm,arm) for arm in ARMS]
    zipped = tmp_path/'baseline.zip';export_workspace(runs[0],zipped)
    report = compare([zipped,*runs[1:]],require_all=True)
    assert report['full_ablation'] and len(report['comparisons'])==4
    assert report['core_source_and_initialization_verified']
    assert all(r['cityscapes_val_delta_pp']==0 for r in report['comparisons'])


@pytest.mark.parametrize('change,match',[
    (dict(changed_core=True),'core code'),(dict(changed_init=True),'initialization'),
    (dict(lr=1e-3),'declared K2 arm'),(dict(complete=False),'completed'),
])
def test_comparison_rejects_confounding_or_partial_runs(tmp_path,change,match):
    a = make_result(tmp_path/'baseline','baseline')
    b = make_result(tmp_path/'k2','k2',**change)
    with pytest.raises(ValueError,match=match):compare([a,b])


def test_compare_requires_declared_full_matrix_and_unique_arms(tmp_path):
    a = make_result(tmp_path/'baseline','baseline')
    b = make_result(tmp_path/'aug','aug')
    assert not compare([a,b])['full_ablation']
    with pytest.raises(ValueError,match='Full K2 ablation'):compare([a,b],require_all=True)
    with pytest.raises(ValueError,match='duplicate arms'):compare([a,a])


def test_kaggle_output_paths_can_be_audited_on_windows(tmp_path):
    a=make_result(tmp_path/'baseline','baseline',foreign_output=True)
    b=make_result(tmp_path/'k2','k2',foreign_output=True)
    assert compare([a,b])['core_source_and_initialization_verified']


def test_clone_and_uploaded_source_allow_only_line_ending_differences(tmp_path):
    a=make_result(tmp_path/'baseline','baseline')
    b=make_result(tmp_path/'k2','k2',crlf=True)
    assert json.loads((a/'k2_plan.json').read_text())['source_hashes'] != json.loads((b/'k2_plan.json').read_text())['source_hashes']
    assert compare([a,b])['core_source_and_initialization_verified']


def test_epoch_heartbeat_and_child_failure_are_visible(capsys):
    train_epoch([sys.executable,'-c','import time; time.sleep(.15)'],1,20,heartbeat_seconds=.03)
    assert 'still running' in capsys.readouterr().out
    with pytest.raises(subprocess.CalledProcessError):
        train_epoch([sys.executable,'-c','raise SystemExit(7)'],1,20)


def test_partial_epoch_can_be_shared_without_claiming_completion(tmp_path):
    root=make_result(tmp_path/'run',complete=False)
    report=export_share(root,tmp_path/'partial_share.zip')
    assert not report['complete']


def test_review_failure_does_not_block_full_archive_preservation(tmp_path):
    root=make_result(tmp_path/'run')
    (root/'run/best/model.safetensors').write_bytes(b'interrupted generation')
    assert publish_review(root,tmp_path/'full.zip') is None
    assert (root/'share_export_error.json').is_file()
    export_workspace(root,tmp_path/'full.zip')
    assert (tmp_path/'full.zip').is_file()
