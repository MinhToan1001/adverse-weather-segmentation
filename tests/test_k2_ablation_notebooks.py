"""Controls preserve legacy settings and gate target evaluation on a matched source audit."""
import ast
import json
from pathlib import Path
import zipfile

import pytest

from scripts.build_k2_ablation_notebooks import build_all, REFERENCE, POST
from scripts.build_kaggle_notebooks import build as package
from test_k2_notebook import execute_setup, make_city
from test_k2_results import make_result


@pytest.mark.parametrize('index,arm',[(0,'aug'),(1,'film')])
def test_actual_control_setup_uses_shared_code_bundle_and_original_recipe(tmp_path,index,arm):
    bundle=package()[1]
    notebook=json.loads(build_all()[index].read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']))
            assert cell['outputs']==[] and cell['execution_count'] is None
    inputs=tmp_path/'inputs';city=make_city(inputs)
    setup=''.join(notebook['cells'][1]['source'])
    setup=setup.replace(repr('/kaggle/input'),repr(str(inputs))).replace(repr('/kaggle/working'),repr(str(tmp_path/'working')))
    setup=setup.replace('CODE_INPUT = None',f'CODE_INPUT = Path({str(bundle)!r})')
    ns=execute_setup(setup)
    assert (ns['ARM'],ns['RECIPE'],ns['SEED'],ns['EPOCHS'],ns['BATCH_SIZE'])==(arm,'legacy_v1',42,20,2)
    assert ns['CITYSCAPES_ROOT']==city and ns['WEATHER_MODE']=='no-fog'
    assert ns['EVAL_AFTER_TRAIN'] and ns['EVAL_ROOT'].name==f'k2_{arm}_seed42_evaluation'
    assert ns['ARCHIVE'].name==f'k2_{arm}_seed42.zip'


def test_reference_identity_rejects_an_unmatched_seed_before_training(tmp_path):
    reference=make_result(tmp_path/'baseline','baseline')
    ns=dict(REFERENCE_INPUT=reference,RECIPE='legacy_v1',SEED=42,EPOCHS=1,
            BATCH_SIZE=2,ARM='aug',WEATHER_MODE='no-fog')
    exec(compile(REFERENCE,'<actual-control-reference>','exec'),ns)
    ns['SEED']=7
    with pytest.raises(AssertionError,match='Reference must'):
        exec(compile(REFERENCE,'<actual-control-reference>','exec'),ns)


@pytest.mark.parametrize('status,audit_fails',[
    ('training_complete',False),('training_complete',True),('session_checkpoint_ready',False)])
def test_post_training_audits_before_target_and_preserves_archives_on_failure(tmp_path,monkeypatch,status,audit_fails):
    import scripts.k2_results as results
    work=tmp_path/'work';(work/'run').mkdir(parents=True)
    (work/'run/initial').mkdir()
    (work/'run/initial/model.safetensors').write_bytes(b'original initialization')
    (work/'status.json').write_text(json.dumps(dict(status=status,completed_epochs=1)))
    (work/'run/history.jsonl').write_text('{}\n')
    evaluation=tmp_path/'eval';figures=work/'figures';events=[]
    def compare(*args):
        events.append('audit')
        if audit_fails:raise ValueError('Unmatched initialization')
        return dict(core_source_and_initialization_verified=True)
    def run(*args):
        if args[1]=='scripts.evaluate_k2':
            events.append('target');evaluation.mkdir()
        elif args[1]=='scripts.plot_k2':figures.mkdir(exist_ok=True)
        elif args[1]=='scripts.k2_results':
            with zipfile.ZipFile(tmp_path/'share.zip','w') as archive:
                archive.writestr('run/best/model.safetensors',b'trained weights')
    monkeypatch.setattr(results,'compare',compare)
    ns=dict(WORK=work,REFERENCE_INPUT=tmp_path/'reference.zip',EVAL_AFTER_TRAIN=True,
            EVAL_ROOT=evaluation,FIGURES=figures,ACDC_ROOT=None,ACDC_MASK_ROOT=None,
            CITYSCAPES_ROOT=tmp_path/'city',ARCHIVE=tmp_path/'full.zip',
            SHARE_ARCHIVE=tmp_path/'share.zip',EVAL_ARCHIVE=tmp_path/'evaluation.zip',
            CONTROL_AUDIT_ARCHIVE=tmp_path/'comparison_bundle.zip',
            run=run,export_workspace=lambda *args:events.append('archive'),
            json=json,zipfile=zipfile,shutil=__import__('shutil'),FileLink=str,display=lambda *args:None)
    if audit_fails:
        with pytest.raises(ValueError,match='Unmatched initialization'):
            exec(compile(POST,'<actual-control-post>','exec'),ns)
    else:exec(compile(POST,'<actual-control-post>','exec'),ns)
    assert 'archive' in events
    if status=='training_complete' and not audit_fails:
        assert events[:2]==['audit','target']
        with zipfile.ZipFile(tmp_path/'comparison_bundle.zip') as bundle:
            assert bundle.read('run/initial/model.safetensors')==b'original initialization'
            assert bundle.read('run/best/model.safetensors')==b'trained weights'
            assert not any(n.endswith('.pt') for n in bundle.namelist())
    else:
        assert 'target' not in events and not evaluation.exists()
        assert not (tmp_path/'comparison_bundle.zip').exists()
