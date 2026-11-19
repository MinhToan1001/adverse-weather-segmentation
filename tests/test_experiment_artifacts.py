import json
import zipfile

import pytest
import torch

from src.experiments.run_support import committed_state, fits_session
from src.experiments.packaging import export_workspace


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
