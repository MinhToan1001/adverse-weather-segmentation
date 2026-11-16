import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

from scripts import build_research_notebooks
from scripts.run_research import committed_state, fits_session, prepare
from src.experiments.packaging import export_workspace
from src.utils.artifacts import sha256_file


def test_session_reserve_accounts_for_observed_long_epochs():
    assert fits_session(0, 36000, [])
    assert not fits_session(32000, 36000, [2500])
    assert not fits_session(0, 5000, [4000, 2000])
    assert fits_session(0, 36000, [4000, 2000])


def test_archive_omits_redundant_extracted_source_but_keeps_canonical_zip_and_checkpoint(tmp_path):
    work = tmp_path/'work'
    (work/'experiment/source/src').mkdir(parents=True)
    (work/'experiment/source/src/a.py').write_text('original code')
    with zipfile.ZipFile(work/'experiment/source.zip', 'w') as archive:
        archive.writestr('src/a.py', 'original code')
    (work/'experiment/weights').write_bytes(b'checkpoint')
    result = export_workspace(work, tmp_path/'result.zip')
    with zipfile.ZipFile(result) as archive:
        assert set(archive.namelist()) == {'experiment/source.zip', 'experiment/weights'}
        assert archive.read('experiment/weights') == b'checkpoint'
    with pytest.raises(ValueError, match='outside'):
        export_workspace(work, work/'bad.zip')


def test_prepare_freezes_fit_weights_and_resume_refuses_loss_or_data_drift(tmp_path):
    fit, dev = tmp_path/'fit.csv', tmp_path/'dev.csv'
    fit.write_text('fit'); dev.write_text('dev')
    data = tmp_path/'data.json'
    data.write_text(json.dumps(dict(paths=dict(source_train_manifest=str(fit), source_monitor_manifest=str(dev)))))
    args = SimpleNamespace(output=str(tmp_path/'experiment'), model='segformer_b2', stage='source',
        loss='ce', class_crop=False, seed=42, initialization='random', data=str(data), checkpoint=None)
    initial = prepare(args)
    assert prepare(args) == initial
    changed = deepcopy(args); changed.loss='focal'
    with pytest.raises(ValueError, match='Resume'):
        prepare(changed)
    fit.write_text('different fit')
    with pytest.raises(ValueError, match='manifests changed'):
        prepare(args)


def test_completion_requires_committed_checkpoint_config_and_all_epochs(tmp_path):
    run = tmp_path/'run'; (run/'last').mkdir(parents=True)
    cfg = dict(train=dict(epochs=20))
    (run/'config.json').write_text(json.dumps(cfg))
    state = dict(epoch=20, global_step=40, config=cfg)
    torch.save(state, run/'last/training_state.pt')
    (run/'history.jsonl').write_text(''.join(json.dumps(dict(epoch=e))+'\n' for e in range(1,21)))
    resources = [dict(epoch=e, optimizer_updates=2) for e in range(1,21)]
    (run/'resources.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in resources))
    assert committed_state(run, 20) == cfg
    resources.pop()
    (run/'resources.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in resources))
    with pytest.raises(ValueError, match='committed run'):
        committed_state(run, 20)


def test_current_notebook_has_parsable_cells_model_selection_and_export_in_finally():
    path=build_research_notebooks.build('segformer_b2')
    notebook = json.loads(path.read_text(encoding='utf-8'))
    code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    assert "MODEL_NAME='segformer_b2'" in code
    assert "'--models',MODEL_NAME" in code
    assert 'RESUME_INPUT' in code
    assert "finally:\n    export_workspace" in code
    assert "'source_hashes'" in code and 'NOTEBOOK_STARTED' in code
