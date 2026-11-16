"""Architecture-independent19-class losses; fit-only weighting and void-safe reductions."""
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from src.utils.artifacts import sha256_file


LOSS_NAMES = ('ce', 'weighted_ce', 'focal', 'ce_dice')


class SegmentationLoss(nn.Module):
    def __init__(self, name='ce', class_weights=None, gamma=2., dice_weight=.5):
        super().__init__()
        if name not in LOSS_NAMES or not 0 <= gamma <= 5 or not 0 <= dice_weight <= 2:
            raise ValueError('Invalid loss name or loss coefficients')
        if name == 'weighted_ce' and class_weights is None:
            raise ValueError('weighted_ce requires verified fit-only class weights')
        weight = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        if weight is not None and (weight.shape != (19,) or not torch.isfinite(weight).all()
                                   or (weight < 0).any() or not weight.sum() > 0):
            raise ValueError('Require19 finite nonnegative class weights with a positive sum')
        self.register_buffer('weight', weight)
        self.name, self.gamma, self.dice_weight = name, float(gamma), float(dice_weight)

    def forward(self, logits, labels):
        if logits.ndim != 4 or logits.shape[1] != 19 or labels.ndim != 3 or logits.shape[0] != labels.shape[0]:
            raise ValueError('Require Bx19xHxW logits and BxHxW labels')
        if not (((labels >= 0) & (labels < 19)) | (labels == 255)).all():
            raise ValueError('Labels must be trainIds0..18 or void255')
        values = F.interpolate(logits.float(), labels.shape[-2:], mode='bilinear', align_corners=False)
        valid = labels != 255
        if not valid.any():
            return values.sum() * 0.
        target = labels.masked_fill(~valid, 0)
        logp = F.log_softmax(values, dim=1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)[valid]
        weights = self.weight[target[valid]] if self.weight is not None else torch.ones_like(logpt)
        if self.name == 'focal':
            # p_t is derived from unweighted log-probability, not weighted CE.
            pixel_loss = -(1. - logpt.exp()).pow(self.gamma) * logpt
        else:
            pixel_loss = -logpt
        ce = (weights * pixel_loss).sum() / weights.sum().clamp_min(1e-12)
        if self.name != 'ce_dice':
            return ce
        probability = logp.exp() * valid[:, None]
        truth = F.one_hot(target, 19).permute(0, 3, 1, 2).to(probability) * valid[:, None]
        axes = (0, 2, 3)
        support = truth.sum(axes)
        overlap = (probability * truth).sum(axes)
        dice = (2 * overlap + 1e-6) / (probability.sum(axes) + support + 1e-6)
        # Macro average over classes present in this minibatch; void contributes nowhere.
        dice_loss = (1. - dice[support > 0]).mean()
        return ce + self.dice_weight * dice_loss


def make_criterion(options, fit_manifests):
    options = dict(options)
    path = options.pop('class_weights_file', None)
    provenance = dict(name=options.get('name', 'ce'), class_weights_sha256=None)
    weights = None
    if path:
        artifact = json.loads(Path(path).read_text())
        expected = sorted(sha256_file(p) for p in fit_manifests)
        if sorted(artifact['fit_manifest_sha256']) != expected:
            raise ValueError('Loss weights do not belong to the actual training fit manifests')
        weights = artifact['weights']
        provenance['class_weights_sha256'] = sha256_file(path)
    criterion = SegmentationLoss(class_weights=weights, **options)
    return criterion, provenance
