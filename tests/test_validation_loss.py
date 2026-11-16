"""Native-mask loss diagnostics must not change predictions, RNG or gradients."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.training.train import evaluate_model


class FixedModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits)
        self.calls = 0

    def forward(self, pixel_values):
        self.calls += 1
        return SimpleNamespace(logits=self.logits)


def test_native_loss_unpads_ignores_255_and_weights_by_valid_pixels():
    logits = torch.tensor([[[[8., 2., -1., 9.]], [[-8., -2., 1., -9.]]]])
    model = FixedModel(logits)
    batch = dict(pixel_values=torch.zeros(1, 3, 1, 4), domain=['night'],
                 fit_box=[(0, 1, 1, 2)], labels=[torch.tensor([[0, 255, 1]])])
    # An all-ignore image must not dilute the mean or produce NaN.
    ignored = dict(batch, domain=['fog'], labels=[torch.full((1, 3), 255)])
    other = dict(batch, domain=['rain'], labels=[torch.tensor([[255, 0, 255]])])
    native = F.interpolate(logits[:, :, :, 1:3], size=(1, 3), mode='bilinear', align_corners=False)
    expected = F.cross_entropy(native, batch['labels'][0].unsqueeze(0), ignore_index=255)
    rain_loss = F.cross_entropy(native, other['labels'][0].unsqueeze(0), ignore_index=255)
    result = evaluate_model(model, [batch, ignored, other], torch.device('cpu'), 2, compute_loss=True)
    assert result['cross_entropy'] == pytest.approx(float((2*expected+rain_loss).detach())/3)
    assert result['loss_valid_pixels'] == 3
    assert result['by_domain']['night']['loss_valid_pixels'] == 2
    assert result['by_domain']['night']['cross_entropy'] == pytest.approx(float(expected.detach()))
    assert result['by_domain']['rain']['loss_valid_pixels'] == 1
    assert result['by_domain']['fog']['cross_entropy'] is None
    assert result['by_domain']['fog']['loss_valid_pixels'] == 0
    assert model.calls == 3
    assert model.logits.grad is None


def test_loss_diagnostics_do_not_change_metric_callback_or_rng():
    model = FixedModel(torch.arange(12, dtype=torch.float).reshape(1, 2, 2, 3))
    batch = dict(pixel_values=torch.zeros(1, 3, 2, 3), domain=['rain'],
                 labels=[torch.tensor([[0, 1, 255], [1, 0, 1]])])
    saved_rng = torch.get_rng_state().clone()
    plain_cache, loss_cache = [], []
    plain = evaluate_model(model, [batch], torch.device('cpu'), 2,
                           prediction_callback=lambda *args: plain_cache.append(args))
    extra = evaluate_model(model, [batch], torch.device('cpu'), 2, compute_loss=True,
                           prediction_callback=lambda *args: loss_cache.append(args))
    assert torch.equal(saved_rng, torch.get_rng_state())
    for value in [extra, *extra['by_domain'].values()]:
        value.pop('cross_entropy')
        value.pop('loss_valid_pixels')
    assert plain == extra
    assert torch.equal(plain_cache[0][1], loss_cache[0][1])
    assert torch.equal(plain_cache[0][3], loss_cache[0][3])
    assert model.logits.grad is None


def test_validation_loss_empty_loader_is_explicitly_unavailable():
    model = FixedModel(torch.zeros(1, 2, 1, 1))
    result = evaluate_model(model, [], torch.device('cpu'), 2, compute_loss=True)
    assert result['cross_entropy'] is None and result['loss_valid_pixels'] == 0
    assert model.calls == 0
