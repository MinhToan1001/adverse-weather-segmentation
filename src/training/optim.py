"""Optional decoder LR control; multiplier1 preserves the original optimizer."""
import math

import torch


def make_optimizer(model, settings):
    lr = float(settings["lr"])
    decay = float(settings.get("weight_decay", .01))
    multiplier = float(settings.get("head_lr_multiplier", 1.))
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("head_lr_multiplier must be finite and positive")
    if multiplier == 1.:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    backbone, head = [], []
    for name, parameter in model.named_parameters():
        is_head = model.is_head_parameter(name) if hasattr(model, 'is_head_parameter') else name.startswith('decode_head.')
        (head if is_head else backbone).append(parameter)
    if not backbone or not head:
        raise ValueError("Decoder LR control requires backbone and decode_head parameters")
    return torch.optim.AdamW([
        dict(params=backbone, lr=lr, name="backbone"),
        dict(params=head, lr=lr*multiplier, name="decode_head"),
    ], lr=lr, weight_decay=decay)
