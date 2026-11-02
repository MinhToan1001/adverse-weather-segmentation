from transformers import SegformerConfig, SegformerForSemanticSegmentation

def make_model(settings):
    cfg=settings['model']
    num_labels=int(cfg['num_labels'])
    if cfg.get('pretrained',False):
        model = SegformerForSemanticSegmentation.from_pretrained(
            cfg['checkpoint'], num_labels=num_labels, ignore_mismatched_sizes=True,
            revision=cfg.get('revision'), semantic_loss_ignore_index=255)
        model.config.semantic_loss_ignore_index = 255
        return model
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
    return SegformerForSemanticSegmentation(model_cfg)
