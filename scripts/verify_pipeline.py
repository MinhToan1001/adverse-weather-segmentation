"""Small real-data CPU QA; random models and tiny geometry are not accuracy results."""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from safetensors.torch import load_file
import torch
import yaml

from src.data.class_weights import fit_class_weights
from src.experiments.config import compose_experiment
from src.models.registry import MODEL_NAMES, build_model, parameter_report, load_checkpoint
from src.evaluation.runner import evaluate
from scripts.plot_research_run import plot
from src.utils.artifacts import write_json


ROOT = Path(__file__).resolve().parents[1]


def run(output, data_path):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prepared=json.loads(Path(data_path).read_text())['paths']
    paths={}
    for key,count in [('source_train_manifest',4),('source_monitor_manifest',2)]:
        with Path(prepared[key]).open() as handle:rows=list(csv.DictReader(handle))[:count]
        if len(rows)!=count:raise ValueError('Prepared data must contain at least4fit/2monitor samples')
        destination=output/(key+'.csv')
        with destination.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        paths[key]=str(destination)
    reports = []
    for name, loss in zip(MODEL_NAMES, ('ce', 'weighted_ce', 'ce_dice')):
        run_dir = output / name
        cfg = compose_experiment(name, 'source', paths, run_dir, loss=loss, initialization='random', class_crop=True)
        cfg['data']['image_size'] = [64, 96]
        cfg['data']['augment'].update(min_scale=1., max_scale=1.)
        cfg['data']['augment']['class_aware_crop']['probability'] = 1.
        cfg['train'].update(num_workers=0, max_steps=2, gradient_accumulation_steps=2, amp=False)
        if loss == 'weighted_ce':
            weights = output / 'fit_weights.json'
            fit_class_weights([paths['source_train_manifest']], weights)
            cfg['loss']['class_weights_file'] = str(weights)
        config = output / (name + '.yaml')
        config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        with (output / (name + '.log')).open('w', encoding='utf-8') as handle:
            for epoch in (1, 2):
                command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(config), '--stop-after-epoch', str(epoch)]
                if epoch == 2:
                    command += ['--resume', str(run_dir / 'last')]
                subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        state = torch.load(run_dir / 'last/training_state.pt', map_location='cpu', weights_only=False)
        assert state['epoch'] == 2 and state['config']['train']['epochs'] == 20
        assert state['global_step'] == 2
        assert len((run_dir / 'history.jsonl').read_text().splitlines()) == 2
        plot(run_dir)
        model = load_checkpoint(run_dir / 'last', name).eval()
        reports.append(dict(model=name, parameters=parameter_report(model), loss=loss,
                            two_epoch_stop_resume=True, scheduler_horizon=20, raw_accuracy_claim=False))
        del model, state
        if name == 'deeplabv3plus_resnet50':
            continuous = deepcopy(cfg); continuous['output_dir'] = str(output / 'deeplab_continuous')
            reference_config = output / 'continuous.yaml'
            reference_config.write_text(yaml.safe_dump(continuous), encoding='utf-8')
            with (output / 'continuous.log').open('w', encoding='utf-8') as handle:
                subprocess.run([sys.executable, '-m', 'src.training.train', '--config', str(reference_config),
                                '--stop-after-epoch', '2'], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
            before = load_file(str(run_dir / 'last/model.safetensors'))
            after = load_file(str(output / 'deeplab_continuous/last/model.safetensors'))
            assert all(torch.equal(value, after[key]) for key, value in before.items())
            reports[-1]['continuous_and_resumed_weights_bitwise_equal'] = True
            del before, after
            # Same native evaluator and ECE/confusion/panels path as SegFormer.
            evaluate(run_dir / 'best', 'deeplabv3plus', paths['source_monitor_manifest'], output / 'native_eval',
                     canvas=(64, 96), device='cpu')
    write_json(output / 'qa.json', dict(status='PASS', probes=reports,
        scope='Random initialization,4real-fit/2monitor images,64x96,2batches/epoch. Plumbing/serialization QA only; not20epoch convergence or GPU quality.'))
    return reports


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--out', required=True); parser.add_argument('--data', required=True)
    args=parser.parse_args();print(json.dumps(run(args.out,args.data),indent=2))
