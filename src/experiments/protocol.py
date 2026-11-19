"""Enforce data access before constructing datasets or loading checkpoints."""
from src.data.validation import read_manifest

DG_PROTOCOL = 'cityscapes_only_dg_v1'
SUPERVISED = {'supervised_acdc', 'supervised_acdc_replay'}


def require_training_permission(cfg, allow_supervised_acdc=False):
    dg = cfg.get('experiment_protocol') == DG_PROTOCOL
    if cfg.get('protocol') in SUPERVISED:
        if dg or not allow_supervised_acdc:
            raise ValueError('ACDC supervised training is outside DG. Historical experiments require '
                             '--allow-supervised-acdc; never report them as DG.')
        return
    # Source-only must not select checkpoints using a target val manifest either.
    if cfg.get('protocol', 'source_only') != 'source_only':
        raise ValueError('Unknown training protocol')
    manifests = {k: v for k, v in cfg['data'].items() if k.endswith('_manifest')}
    for key, path in manifests.items():
        rows = read_manifest(path)
        allowed = {'cityscapes'} if dg else {'cityscapes', 'synthetic'}
        if any(row['domain'] not in allowed for row in rows):
            raise ValueError('DG/source-only training and checkpoint selection cannot use ACDC')
        if dg:
            if key not in {'train_manifest', 'val_manifest'}:
                raise ValueError('DG v1 accepts only Cityscapes train/val manifests')
            split, count = ('train', 2975) if key == 'train_manifest' else ('val', 500)
            if len(rows) != count or any(row['split'] != split for row in rows):
                raise ValueError(f'DG v1 requires Cityscapes {split}: {count} images')
    if dg:
        if set(manifests) != {'train_manifest', 'val_manifest'}:
            raise ValueError('DG requires both Cityscapes train and val')
        model = cfg['model']
        if model.get('initialization') != 'imagenet_encoder' or model.get('checkpoint'):
            raise ValueError('DG v1 starts from the pinned ImageNet encoder, never an ACDC-adapted checkpoint')
        if cfg['train']['epochs'] != 20 or cfg.get('early_stopping'):
            raise ValueError('DG v1 uses 20 epochs without early stopping')
