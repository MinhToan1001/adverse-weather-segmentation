"""Compose shared policies, then bind a model name and run-specific data."""
from pathlib import Path

import yaml

from src.models.registry import COMPATIBLE_MODEL_NAMES
from src.training.losses import LOSS_NAMES
from src.utils.artifacts import sha256_file


CONFIG_ROOT = Path(__file__).resolve().parents[2] / 'configs'


def compose_experiment(name, stage, data, output, loss='ce', source_checkpoint=None,
                       initialization='imagenet_encoder', seed=42, class_crop=False):
    if name not in COMPATIBLE_MODEL_NAMES or stage not in ('source', 'adapt') or loss not in LOSS_NAMES:
        raise ValueError('Unknown model, stage or loss preset')
    recipe = yaml.safe_load((CONFIG_ROOT / 'recipes' / (
        'source_20epoch.yaml' if stage == 'source' else 'adaptation_20epoch.yaml')).read_text())
    processing = yaml.safe_load((CONFIG_ROOT / 'preprocessing/shared_512.yaml').read_text())
    if stage == 'source':
        paths = dict(train_manifest=data['source_train_manifest'], val_manifest=data['source_monitor_manifest'])
    else:
        if source_checkpoint is None:
            raise ValueError('Adaptation requires a complete matching Cityscapes source checkpoint')
        paths = {key: data[key] for key in ('train_manifest', 'val_manifest', 'source_train_manifest', 'source_monitor_manifest')}
    augment = dict(min_scale=.5, max_scale=2., horizontal_flip=.5)
    if class_crop:
        augment['class_aware_crop'] = dict(probability=.5, min_pixels=64, attempts=10)
    cfg = dict(recipe, experiment=f'{name}-{stage}-{loss}-seed{seed}',
               training_policy='supervised_comparison_20e_v1', seed=seed,
               model=dict(name=name, num_labels=19, initialization=initialization),
               data=dict(paths, **processing, augment=augment),
               loss=yaml.safe_load((CONFIG_ROOT / 'losses' / (loss + '.yaml')).read_text()),
               checkpointing=dict(transactional=True, keep_previous=True, granularity='epoch'),
               monitoring=dict(validation_loss=True), selection=dict(max_source_drop_pp=1.),
               output_dir=str(Path(output).resolve()))
    if stage == 'adapt':
        source_checkpoint = Path(source_checkpoint).resolve()
        cfg['model'].update(initialization='checkpoint', checkpoint=str(source_checkpoint),
                            checkpoint_sha256=sha256_file(source_checkpoint / 'model.safetensors'))
    diagnostic_key = 'source_train_eval_manifest' if stage == 'source' else 'replay_train_eval_manifest'
    if diagnostic_key in data:
        cfg['monitoring']['train_eval_manifest'] = data[diagnostic_key]
    return cfg
