import json
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file
import torch

from src.training.checkpoints import save_transactional, recover_run, verified_state


class TinyModel:
    def __init__(self, value, fail=False): self.value, self.fail = value, fail
    def save_pretrained(self, directory):
        Path(directory, 'config.json').write_text('{"num_labels":19}')
        save_file({'weight':torch.tensor([float(self.value)])},str(Path(directory,'model.safetensors')))
        if self.fail: raise RuntimeError('simulated interruption during checkpoint write')


def state(epoch):
    return dict(epoch=epoch,global_step=epoch,best_epoch=epoch,best_miou=epoch/100,
        config=dict(train=dict(epochs=20),model=dict(name='segformer_b0')),
        optimizer=dict(moment=torch.tensor([float(epoch)])),scheduler=dict(last_epoch=epoch),
        scaler=dict(scale=65536),rng=dict(torch=torch.get_rng_state()))


def logs(root, history_epochs, resource_epochs):
    root.mkdir(exist_ok=True)
    for name, epochs in [('history.jsonl',history_epochs),('resources.jsonl',resource_epochs)]:
        Path(root,name).write_text(''.join(json.dumps(dict(epoch=e,optimizer_updates=1))+'\n' for e in epochs))


def checkpoint(root, epoch):
    for kind in ('last','best'):
        save_transactional(TinyModel(epoch),root/kind,state(epoch))


def test_interrupted_write_keeps_old_generation_and_recovers_optimizer_scheduler(tmp_path):
    checkpoint(tmp_path,1);logs(tmp_path,[1,2],[1])
    with pytest.raises(RuntimeError,match='simulated interruption'):
        save_transactional(TinyModel(2,fail=True),tmp_path/'last',state(2))
    assert recover_run(tmp_path)==1
    restored=verified_state(tmp_path/'last')
    assert restored['optimizer']['moment'].item()==1 and restored['scheduler']['last_epoch']==1
    assert restored['scaler']['scale']==65536 and restored['config']['train']['epochs']==20
    assert len((tmp_path/'history.jsonl').read_text().splitlines())==1


def test_uncommitted_epoch_rolls_back_both_last_and_selected_best(tmp_path):
    checkpoint(tmp_path,1);checkpoint(tmp_path,2);logs(tmp_path,[1,2],[1])
    assert recover_run(tmp_path)==1
    for kind in ('last','best'):
        assert load_file(str(tmp_path/kind/'model.safetensors'))['weight'].item()==1
    assert verified_state(tmp_path/'best')['epoch']==1


def test_checksum_corruption_uses_previous_generation_and_truncates_partial_logs(tmp_path):
    checkpoint(tmp_path,1);checkpoint(tmp_path,2);logs(tmp_path,[1,2],[1,2])
    (tmp_path/'last/model.safetensors').write_bytes(b'corrupted by interrupted write')
    with (tmp_path/'history.jsonl').open('a') as handle:handle.write('{"epoch":')
    assert recover_run(tmp_path)==1
    assert verified_state(tmp_path/'last')['epoch']==1
    assert len((tmp_path/'resources.jsonl').read_text().splitlines())==1


def test_recovery_handles_rename_gap_with_complete_pending_generation(tmp_path):
    checkpoint(tmp_path,1);checkpoint(tmp_path,2);logs(tmp_path,[1,2],[1,2])
    (tmp_path/'last').rename(tmp_path/'.last.pending')
    assert recover_run(tmp_path)==2
    assert verified_state(tmp_path/'last')['epoch']==2


def test_partial_first_epoch_restarts_from_seed_and_irrecoverable_commits_are_rejected(tmp_path):
    checkpoint(tmp_path,1);logs(tmp_path,[1],[])
    assert recover_run(tmp_path)==0 and not (tmp_path/'last').exists()
    checkpoint(tmp_path,1);logs(tmp_path,[1],[1])
    (tmp_path/'last/training_state.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='No verified checkpoint'):
        recover_run(tmp_path)


def test_pending_checkpoint_failure_does_not_replace_last_complete_zip(tmp_path,monkeypatch):
    from src.experiments.packaging import export_workspace
    import zipfile
    work=tmp_path/'work';work.mkdir();(work/'value').write_text('epoch1')
    destination=tmp_path/'snapshot.zip'
    export_workspace(work,destination);original=destination.read_bytes()
    def interrupted(*args,**kwargs):raise OSError('simulated export interruption')
    monkeypatch.setattr(zipfile.ZipFile,'write',interrupted)
    with pytest.raises(OSError,match='export interruption'):export_workspace(work,destination)
    assert destination.read_bytes()==original
