"""Enforce data access before constructing datasets or loading checkpoints."""
from src.data.validation import read_manifest

DG_PROTOCOL = 'cityscapes_only_dg_v1'
SUPERVISED = {'supervised_acdc', 'supervised_acdc_replay'}


def require_training_permission(cfg, allow_supervised_acdc=False):
    dg = cfg.get('experiment_protocol') == DG_PROTOCOL
    k2 = cfg.get('experiment_protocol') == 'weatherseg_k2_source_v1'
    if cfg.get('protocol') in SUPERVISED:
        if dg or k2 or not allow_supervised_acdc:
            raise ValueError('ACDC supervised training is outside DG. Historical experiments require '
                             '--allow-supervised-acdc; never report them as DG.')
        return
    # Source-only must not select checkpoints using a target val manifest either.
    if cfg.get('protocol', 'source_only') != 'source_only':
        raise ValueError('Unknown training protocol')
    manifests = {k: v for k, v in cfg['data'].items() if k.endswith('_manifest')}
    for key, path in manifests.items():
        rows = read_manifest(path)
        allowed = {'cityscapes'} if dg or k2 else {'cityscapes', 'synthetic'}
        if any(row['domain'] not in allowed for row in rows):
            raise ValueError('DG/source-only training and checkpoint selection cannot use ACDC')
        if dg or k2:
            if key not in {'train_manifest', 'val_manifest'}:
                raise ValueError('DG v1 accepts only Cityscapes train/val manifests')
            split, count = ('train', 2975) if key == 'train_manifest' else ('val', 500)
            if len(rows) != count or any(row['split'] != split for row in rows):
                raise ValueError(f'DG v1 requires Cityscapes {split}: {count} images')
    if k2:
        if set(manifests) != {'train_manifest', 'val_manifest'} or cfg.get('dg') is not None:
            raise ValueError('K2 requires only source train/val and its own declared weather policy')
        if (cfg['model']['name'] not in {'segformer_b0','weatherseg_b0'}
                or cfg['model'].get('initialization') != 'imagenet_encoder' or cfg['model'].get('checkpoint')):
            raise ValueError('K2 starts from the pinned B0 ImageNet encoder, not an ACDC checkpoint')
    if dg:
        allowed_data={'train_manifest','val_manifest','image_size','preprocessing','augment'}
        if set(cfg['data'])-allowed_data:
            raise ValueError('Unexpected data strategy outside the source-only DG recipe')
        if cfg.get('dg') is not None:
            from src.training.dg import validate_settings
            validate_settings(cfg['dg'])
            if cfg.get('loss') != {'name': 'ce'}:
                raise ValueError('DG arms use the declared CE control')
        if set(manifests) != {'train_manifest', 'val_manifest'}:
            raise ValueError('DG requires both Cityscapes train and val')
        model = cfg['model']
        if model.get('name')=='deeplabv3plus_resnet50' and cfg['train'].get('batch_size',2)<2:
            raise ValueError('DeepLab training BatchNorm requires a real batch of at least2')
        if model.get('initialization') != 'imagenet_encoder' or model.get('checkpoint'):
            raise ValueError('DG v1 starts from the pinned ImageNet encoder, never an ACDC-adapted checkpoint')
        if cfg['train']['epochs'] != 20 or cfg.get('early_stopping'):
            raise ValueError('DG v1 uses 20 epochs without early stopping')
