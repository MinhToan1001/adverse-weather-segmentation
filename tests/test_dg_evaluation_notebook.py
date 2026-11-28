"""Exercise the actual standalone evaluation notebook, without a GPU or weights."""
import json
from pathlib import Path
import sys
from types import ModuleType
import zipfile

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1]/'notebooks/04_evaluate_dg_cohort.ipynb'


def code(index):
    return ''.join(json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells'][index]['source'])


@pytest.fixture
def helpers():
    namespace = {}
    exec(compile(code(1), str(NOTEBOOK), 'exec'), namespace)
    return namespace


def suite(path, model='segformer_b2', complete=True, zip_output=False):
    files = {'suite_plan.json':dict(schema='automatic_dg_model_suite_v1',
        protocol='cityscapes_only_dg_v1',target_data_access='none',model=model,seed=42,
        arms=['baseline','aug','consistency'],source_hashes={}),
        'suite_status.json':dict(status='suite_complete' if complete else 'session_checkpoint_ready',completed_arms=3 if complete else 2)}
    if zip_output:
        path.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(path,'w') as z:
            for n,v in files.items():z.writestr(n,json.dumps(v))
    else:
        path.mkdir(parents=True)
        for n,v in files.items():(path/n).write_text(json.dumps(v))
    return path


@pytest.mark.parametrize('zipped',[False,True])
def test_finds_complete_correct_model_inside_dataset(tmp_path,helpers,zipped):
    expected=suite(tmp_path/'user/input'/('result.zip' if zipped else 'result'),zip_output=zipped)
    suite(tmp_path/'old_partial',complete=False)
    suite(tmp_path/'deeplab',model='deeplabv3plus_resnet50')
    assert helpers['find_suite'](None,'segformer_b2',tmp_path)==expected


def test_multiple_inputs_require_explicit_path(tmp_path,helpers):
    first=suite(tmp_path/'first');suite(tmp_path/'second')
    with pytest.raises(ValueError,match='exactly one'):
        helpers['find_suite'](None,'segformer_b2',tmp_path)
    assert helpers['find_suite'](first,'segformer_b2',tmp_path)==first


def test_partial_epoch4_release_is_not_an_evaluation_input(tmp_path,helpers):
    suite(tmp_path/'epoch4',model='deeplabv3plus_resnet50',complete=False)
    with pytest.raises(ValueError,match='COMPLETE'):
        helpers['find_suite'](None,'deeplabv3plus_resnet50',tmp_path)


@pytest.mark.parametrize('name',['../bad','/bad','x/../../bad','C:/bad',r'x\..\..\bad'])
def test_rejects_unsafe_archive_paths(helpers,name):
    with pytest.raises(ValueError,match='Unsafe'):
        helpers['safe_name'](name)


def test_source_hash_mismatch_rejected_before_import_or_write(tmp_path,helpers):
    archive=tmp_path/'suite.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('source/safe.py','changed')
    with pytest.raises(ValueError,match='source hash mismatch'):
        helpers['restore_source'](archive,tmp_path/'repo',{'safe.py':'wrong'})
    assert not (tmp_path/'repo').exists()


def test_auto_finds_acdc_nested_rgb_gt_and_city(tmp_path,helpers):
    root=tmp_path/'ACDC'
    for d in ['fog','night','rain','snow']:
        (root/f'rgb_anon_trainvaltest/rgb_anon/{d}/val').mkdir(parents=True)
        (root/f'gt_trainval/gt/{d}/val').mkdir(parents=True)
    found=helpers['find_tree'](root,tmp_path,'rgb_anon',[f'rgb_anon/{d}/val' for d in ['fog','night','rain','snow']])
    assert found==root/'rgb_anon_trainvaltest'
    assert helpers['find_tree'](None,tmp_path,'gt',['gt/fog/val'])==root/'gt_trainval'


def test_completed_cache_checks_manifest_weights_and_confusions(tmp_path,helpers):
    out=tmp_path/'eval';out.mkdir();manifest=tmp_path/'val.csv';manifest.write_text('fixture')
    (out/'confusions').mkdir();cache=out/'confusions/0000.npy';cache.write_bytes(b'fixture')
    metrics=out/'metrics.json';metrics.write_text(json.dumps(dict(n_images=1,by_domain={'fog':{'n_images':1}})))
    state=dict(status='complete',checkpoint_sha256='weights',manifest_sha256=helpers['sha'](manifest),
        metrics_sha256=helpers['sha'](metrics),image_size=[512,1024],dtype='float32',tta=False,
        cache_sha256={'confusions/0000.npy':helpers['sha'](cache)})
    (out/'run-state.json').write_text(json.dumps(state))
    assert helpers['completed_evaluation'](out,'weights',manifest,{'fog':1})
    cache.write_bytes(b'changed')
    with pytest.raises(ValueError,match='confusion matrix changed'):
        helpers['completed_evaluation'](out,'weights',manifest,{'fog':1})
    with pytest.raises(ValueError,match='identity changed'):
        helpers['completed_evaluation'](out,'other weights',manifest,{'fog':1})


def mock_module(monkeypatch,name,**attributes):
    module=ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules,name,module)


def test_actual_manifest_cell_respects_target_validator_guard_and_checks_geometry(tmp_path,helpers):
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    city=tmp_path/'city';rgb=tmp_path/'acdc_rgb';gt=tmp_path/'acdc_gt'
    for i in range(500):
        name=f'city_{i:06d}_000000'
        image=city/'leftImg8bit/val/city'/(name+'_leftImg8bit.png')
        mask=city/'gtFine/val/city'/(name+'_gtFine_labelTrainIds.png')
        image.parent.mkdir(parents=True,exist_ok=True);mask.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(8,4),(80,80,80)).save(image);Image.new('L',(8,4),0).save(mask)
    for domain,count in [('fog',100),('night',106),('rain',100),('snow',100)]:
        for i in range(count):
            name=f'{domain}_{i:06d}'
            image=rgb/'rgb_anon'/domain/'val'/(name+'_rgb_anon.png')
            mask=gt/'gt'/domain/'val'/(name+'_gt_labelTrainIds.png')
            image.parent.mkdir(parents=True,exist_ok=True);mask.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(8,4),(20,20,20)).save(image);Image.new('L',(8,4),0).save(mask)
    out=tmp_path/'out';out.mkdir()
    helpers.update(CITYSCAPES_ROOT=city,ACDC_RGB_ROOT=rgb,ACDC_GT_ROOT=gt,OUT=out,STAGING=tmp_path/'staging',torch=torch,
        sealed={'status':'cohort_frozen_before_target_evaluation','candidates':[{}]*6})
    # Use the real generic validator; it must continue blocking target-label scans.
    exec(code(6),helpers)
    assert helpers['checked']=={'cityscapes','fog','night','rain','snow'}


@pytest.mark.parametrize('fail',[False,True])
def test_actual_execute_cell_runs_twelve_jobs_skips_completed_and_exports_on_error(tmp_path,helpers,monkeypatch,fail):
    calls=[];exports=[]
    out=tmp_path/'out';out.mkdir()
    candidates=[dict(id=f'{model}:{arm}',weights_sha256='weights')
        for model in helpers['MODELS'] for arm in helpers['ARMS']]
    def evaluate(cohort,run_id,manifest,destination,scope,device):
        assert device=='cuda'
        calls.append((run_id,scope))
        if fail:raise RuntimeError('simulated interrupt')
        destination.mkdir(parents=True);(destination/'done').write_text('fixture')
        return {'miou':.5}
    mock_module(monkeypatch,'scripts.evaluate_dg',evaluate_frozen=evaluate)
    mock_module(monkeypatch,'src.experiments.packaging',export_workspace=lambda *args:exports.append(args))
    mock_module(monkeypatch,'IPython.display',display=lambda *args:None,FileLink=str)
    helpers.update(OUT=out,ARCHIVE=tmp_path/'result.zip',sealed={'candidates':candidates},cohort_path=tmp_path/'cohort.json',
        plan={'scopes':['official-city','official-acdc']},manifests={'official-city':'city','official-acdc':'acdc'},
        completed_evaluation=lambda p,*args:(p/'done').is_file())
    if fail:
        with pytest.raises(RuntimeError,match='interrupt'):exec(code(8),helpers)
        assert len(exports)==1
    else:
        exec(code(8),helpers);assert len(calls)==12 and len(set(calls))==12
        exec(code(8),helpers);assert len(calls)==12
        assert len(exports)==14
