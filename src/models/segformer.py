from transformers import SegformerConfig, SegformerForSemanticSegmentation
from src.models import segformer_b0, segformer_b2


def model_class_for_config(config, variant=None):
    """Dispatch checkpoint loading to its separate B0 or B2 implementation."""
    variants = {"b0": segformer_b0, "b2": segformer_b2}
    if variant is not None:
        if variant not in variants or not variants[variant].matches_config(config):
            raise ValueError(f"Checkpoint architecture does not match requested {variant}")
        return segformer_b0.SegFormerB0 if variant == "b0" else segformer_b2.SegFormerB2
    if segformer_b0.matches_config(config):
        return segformer_b0.SegFormerB0
    if segformer_b2.matches_config(config):
        return segformer_b2.SegFormerB2
    # Reduced custom sizes are CPU-QA fixtures,not full B0/B2.
    return SegformerForSemanticSegmentation


def load_segformer_checkpoint(checkpoint, variant=None, **options):
    config = options.get("config")
    if config is None:
        routing = {key: options[key] for key in ("local_files_only", "revision", "cache_dir", "token", "subfolder") if key in options}
        config = SegformerConfig.from_pretrained(checkpoint, **routing)
    model_class = model_class_for_config(config, variant)
    return model_class.from_pretrained(checkpoint, **options)

def make_model(settings):
    cfg=settings['model']
    num_labels=int(cfg['num_labels'])
    if cfg.get('pretrained',False):
        model = load_segformer_checkpoint(
            cfg['checkpoint'], num_labels=num_labels, ignore_mismatched_sizes=True,
            revision=cfg.get('revision'), semantic_loss_ignore_index=255,variant=cfg.get('variant'))
        model.config.semantic_loss_ignore_index = 255
        return model
    if cfg.get('variant') in ('b0','b2'):
        module=segformer_b0 if cfg['variant']=='b0' else segformer_b2
        model_cfg=module.make_config(num_labels)
        for name in ('hidden_sizes','depths','decoder_hidden_size'):
            if name in cfg:setattr(model_cfg,name,cfg[name])
        return model_class_for_config(model_cfg,cfg['variant'])(model_cfg)
    model_cfg=SegformerConfig(
        num_labels=num_labels,
        semantic_loss_ignore_index=255,
        hidden_sizes=cfg.get('hidden_sizes',[32,64,160,256]),
        depths=cfg.get('depths',[2,2,2,2]),
        decoder_hidden_size=cfg.get('decoder_hidden_size',256),
        num_attention_heads=[1,2,5,8] if not cfg.get('hidden_sizes') else [1,2,4,8],
        sr_ratios=[8,4,2,1],
        patch_sizes=[7,3,3,3],
        strides=[4,2,2,2],
    )
    return model_class_for_config(model_cfg,cfg.get('variant'))(model_cfg)
