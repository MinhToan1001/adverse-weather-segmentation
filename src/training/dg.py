"""Shared source-only paired-view objective for CNNs and Transformers."""
from contextlib import nullcontext
import math
import random

import torch
import torch.nn.functional as F

from src.data.augmentations.source_style import SourceStyle

ARMS = ('baseline', 'aug', 'consistency')


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) != {'arm', 'pair_probability', 'consistency_weight', 'source_style'}:
        raise ValueError('Invalid DG settings keys')
    arm, probability, weight = (settings[k] for k in ('arm', 'pair_probability', 'consistency_weight'))
    if arm not in ARMS or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                  and math.isfinite(x) for x in (probability, weight)):
        raise ValueError('Invalid DG arm/probability/weight')
    if not 0 <= probability <= 1 or not 0 <= weight <= 10:
        raise ValueError('DG probability must be [0,1], consistency weight [0,10]')
    if arm == 'baseline':
        if probability != 0 or weight != 0 or settings['source_style'] is not None:
            raise ValueError('Baseline must disable paired style and consistency')
    else:
        if probability <= 0 or (arm == 'aug' and weight != 0):
            raise ValueError('Aug arm has no consistency; paired arms require positive probability')
        SourceStyle(settings['source_style'])
    return settings


def paired_batch(settings, seed, epoch, step):
    return random.Random(f'dg-pair:{seed}:{epoch}:{step}').random() < settings['pair_probability']


def reference_log_probability(logits, size):
    return F.log_softmax(F.interpolate(logits.detach().float(), size=size,
                                      mode='bilinear', align_corners=False), dim=1).detach()


def consistency_kl(student_logits, teacher_log_probability, labels):
    values = F.interpolate(student_logits.float(), labels.shape[-2:], mode='bilinear', align_corners=False)
    if values.shape != teacher_log_probability.shape or values.shape[1] != 19:
        raise ValueError('Aligned Bx19xHxW teacher/student predictions required')
    if not (((labels >= 0) & (labels < 19)) | (labels == 255)).all():
        raise ValueError('Consistency labels must be trainIds or ignore255')
    valid = labels != 255
    if not valid.any():
        return values.sum() * 0.
    teacher = teacher_log_probability.detach()
    student = F.log_softmax(values, dim=1)
    return (teacher.exp() * (teacher - student)).sum(1)[valid].mean()


def backward_batch(model, batch, criterion, scaler, divisor, device, paired=False,
                   consistency_weight=0., amp_enabled=False, amp_dtype=torch.float16):
    """Two sequential backward calls, one optimizer update owned by the trainer.

    Clean activations are released before the styled forward. Teacher logits are
    detached; clean parameters still receive their own CE gradient. C1/C2 have
    identical forward order/BN updates; KL itself consumes no random numbers.
    """
    if divisor < 1 or consistency_weight < 0:
        raise ValueError('Invalid accumulation divisor or consistency weight')
    labels = batch['labels'].to(device)
    context = lambda: torch.autocast('cuda', dtype=amp_dtype) if amp_enabled else nullcontext()
    with context():
        clean = model(pixel_values=batch['pixel_values'].to(device)).logits
        clean_ce = criterion(clean, labels)
    if not torch.isfinite(clean_ce):
        raise RuntimeError('Non-finite clean CE')
    teacher = reference_log_probability(clean, labels.shape[-2:]) if paired and consistency_weight else None
    clean_value = float(clean_ce.detach())
    scaler.scale(clean_ce * (.5 if paired else 1.) / divisor).backward()
    del clean, clean_ce
    style_value, kl_value = None, 0.
    if paired:
        with context():
            styled = model(pixel_values=batch['styled_pixel_values'].to(device)).logits
            style_ce = criterion(styled, labels)
            kl = consistency_kl(styled, teacher, labels) if teacher is not None else styled.float().sum() * 0.
            objective = .5 * style_ce + consistency_weight * kl
        if not torch.isfinite(objective):
            raise RuntimeError('Non-finite styled CE/consistency')
        style_value, kl_value = float(style_ce.detach()), float(kl.detach())
        scaler.scale(objective / divisor).backward()
    return dict(objective=clean_value if not paired else .5 * (clean_value + style_value) + consistency_weight * kl_value,
                clean_ce=clean_value, styled_ce=style_value, consistency_kl=kl_value, paired=paired)
