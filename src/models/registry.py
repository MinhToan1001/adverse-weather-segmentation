"""One training/checkpoint interface; each architecture stays in its own file."""
import json
from pathlib import Path

from src.models import segformer_b0, segformer_b2
from src.models.deeplabv3plus_resnet50 import DeepLabV3PlusResNet50
from src.models.segformer import load_segformer_checkpoint, make_model as build_segformer
from src.data.labels import CLASS_NAMES


MODEL_NAMES = ('segformer_b0', 'segformer_b2', 'deeplabv3plus_resnet50')
ROLES = dict(segformer_b0='baseline', segformer_b2='main', deeplabv3plus_resnet50='matched_size_cnn')
ENCODERS = {
    'b0': ('nvidia/mit-b0', '80983a413c30d36a39c20203974ae7807835e2b4',
           '4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc'),
    'b2': ('nvidia/mit-b2', '3bb39e8739149c3777d0325349b2a6c32c6413db',
           '4500b5665471b593e6757e15bcca5034f433fe3902fe8ec2b7230774a57f264f'),
}


def build_model(settings):
    cfg = settings['model']
    if cfg.get('num_labels', 19) != 19:
        raise ValueError('The research data contract has19 classes')
    name = cfg['name']
    if name not in MODEL_NAMES:
        raise ValueError(f'Unknown training model: {name}; choose {MODEL_NAMES}')
    initialization = cfg.get('initialization', 'random')
    if initialization == 'checkpoint':
        return load_checkpoint(cfg['checkpoint'], name)
    if initialization not in ('random', 'imagenet_encoder'):
        raise ValueError('Use random, imagenet_encoder or checkpoint initialization')
    if name == 'deeplabv3plus_resnet50':
        return DeepLabV3PlusResNet50(imagenet_encoder=initialization == 'imagenet_encoder')
    variant = name.rsplit('_', 1)[1]
    if initialization == 'random':
        model = (segformer_b0 if variant == 'b0' else segformer_b2).build_model()
        model.config.id2label = dict(enumerate(CLASS_NAMES))
        model.config.label2id = {value: key for key, value in model.config.id2label.items()}
        return model
    # Encoder is ImageNet-pretrained; the19-class segmentation head starts new.
    from huggingface_hub import hf_hub_download
    from src.utils.artifacts import sha256_file
    repository, revision, checksum = ENCODERS[variant]
    config_path = hf_hub_download(repository, 'config.json', revision=revision)
    weights = hf_hub_download(repository, 'pytorch_model.bin', revision=revision)
    if sha256_file(weights) != checksum:
        raise ValueError('ImageNet encoder checkpoint hash mismatch')
    model = dict(cfg, variant=variant, num_labels=19, pretrained=True,
                 checkpoint=str(Path(config_path).parent))
    result = build_segformer(dict(model=model))
    result.config.id2label = dict(enumerate(CLASS_NAMES))
    result.config.label2id = {value: key for key, value in result.config.id2label.items()}
    return result


def load_checkpoint(directory, name=None):
    config = json.loads((Path(directory) / 'config.json').read_text())
    if config.get('model_type') == 'project_deeplabv3plus_resnet50':
        if name not in (None, 'deeplabv3plus_resnet50'):
            raise ValueError('Checkpoint is DeepLab, requested a different model')
        return DeepLabV3PlusResNet50.from_pretrained(directory)
    if name == 'deeplabv3plus_resnet50':
        raise ValueError('DeepLab requires its own complete19-class checkpoint')
    variant = name.rsplit('_', 1)[1] if name in MODEL_NAMES[:2] else None
    model, info = load_segformer_checkpoint(directory, variant=variant, local_files_only=True, output_loading_info=True)
    if info['missing_keys'] or info['unexpected_keys'] or info.get('mismatched_keys'):
        raise ValueError('Checkpoint loading mismatch')
    if model.config.num_labels != 19 or model.config.semantic_loss_ignore_index != 255:
        raise ValueError('Require19classes and ignore255')
    return model


def parameter_report(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
