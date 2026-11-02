"""Audit epoch-boundary resume with synthetic data only; never opens raw datasets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
import yaml
from src.training import train


def verify(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for split, count in [('train', 4), ('val', 2)]:
        rows = []
        for i in range(count):
            labels = np.tile(np.arange(64, dtype=np.uint8) // 22, (64, 1))
            colors = np.array([[220,30,30], [30,220,30], [30,30,220]], dtype=np.uint8)
            image, mask = output / f'{split}_{i}_rgb.png', output / f'{split}_{i}_mask.png'
            Image.fromarray(colors[labels]).save(image)
            Image.fromarray(labels).save(mask)
            rows.append(dict(image=str(image), mask=str(mask), domain='synthetic', split=split))
        path = output / f'{split}.csv'
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['image','mask','domain','split'])
            writer.writeheader()
            writer.writerows(rows)
        manifests[split] = str(path)
    cfg = yaml.safe_load(Path('configs/segformer_b0_smoke.yaml').read_text())
    cfg['data'].update(train_manifest=manifests['train'], val_manifest=manifests['val'],
        augment=dict(min_scale=1, max_scale=1, horizontal_flip=0))
    cfg['train'].update(epochs=3, lr=0.001, amp=True, gradient_accumulation_steps=3)
    def run(name, resume=False):
        cfg['output_dir'] = str(output / name)
        config = output / f'{name}.yaml'
        config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        args = ['train', '--config', str(config)]
        if resume:
            args += ['--resume', str(output / name / 'last')]
        with patch.object(sys, 'argv', args):
            train.main()
    started = time.perf_counter()
    run('full')
    original = train._save_checkpoint
    class Interrupted(Exception):
        pass
    def interrupt(model, directory, state):
        original(model, directory, state)
        if directory.name == 'best' and state['epoch'] == 1:
            raise Interrupted('Simulated interruption after epoch 1 checkpoint')
    with patch.object(train, '_save_checkpoint', interrupt):
        try:
            run('resumed')
        except Interrupted:
            pass
        else:
            raise AssertionError('Interruption did not occur')
    run('resumed', resume=True)
    full = train.SegformerForSemanticSegmentation.from_pretrained(output / 'full/last')
    resumed = train.SegformerForSemanticSegmentation.from_pretrained(output / 'resumed/last')
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in full.state_dict().items())
    history = [json.loads(line) for line in (output / 'full/history.jsonl').read_text().splitlines()]
    other = [json.loads(line) for line in (output / 'resumed/history.jsonl').read_text().splitlines()]
    assert history == other
    assert history[-1]['train_loss'] < history[0]['train_loss']
    assert all(not row['amp_enabled'] for row in history) if not torch.cuda.is_available() else True
    states = [torch.load(output / name / 'last/training_state.pt', weights_only=False) for name in ['full','resumed']]
    assert states[0]['scheduler'] == states[1]['scheduler']
    assert states[0]['global_step'] == states[1]['global_step'] == 3
    # Changed training horizon must fail before training continues.
    cfg['train']['epochs'] = 4
    changed = output / 'changed.yaml'
    changed.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with patch.object(sys, 'argv', ['train','--config',str(changed),'--resume',str(output / 'resumed/last')]):
        try:
            train.main()
        except ValueError as error:
            assert 'original config' in str(error)
        else:
            raise AssertionError('Changed resume config was accepted')
    result = dict(status='PASS', data='synthetic only', epochs=3,
        exact_weights_equal=True, exact_history_equal=True, scheduler_equal=True,
        changed_resume_config_rejected=True, partial_accumulation_verified=True,
        losses=[row['train_loss'] for row in history], elapsed_seconds=time.perf_counter()-started,
        device=history[0]['device'], amp_enabled=history[0]['amp_enabled'],
        checkpoints={name: hashlib.sha256((output / name / 'best/model.safetensors').read_bytes()).hexdigest() for name in ['full','resumed']})
    (output / 'verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory, must not already exist')
    verify(parser.parse_args().output)
