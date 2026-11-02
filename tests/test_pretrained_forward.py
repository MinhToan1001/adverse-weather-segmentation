"""Day 02: production factory, real pretrained encoder and 19-class loss.

RUN_PRETRAINED_TESTS=1 enables checkpoint checks; HF_HUB_OFFLINE=1 uses cache.
The default suite always runs an offline B0 forward check.
"""
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from transformers import SegformerModel

from src.models.segformer import make_model


@pytest.fixture(scope="module")
def baseline_settings():
    path = Path(__file__).resolve().parents[1] / "configs/segformer_b0_cityscapes.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_offline_b0_forward(baseline_settings):
    settings = {"model": {**baseline_settings["model"], "pretrained": False}}
    torch.manual_seed(42)
    model = make_model(settings).eval()
    with torch.inference_mode():
        logits = model(pixel_values=torch.randn(1, 3, 64, 96)).logits
    assert logits.shape == (1, 19, 16, 24)
    assert torch.isfinite(logits).all()
    assert model.config.semantic_loss_ignore_index == 255


@pytest.fixture(scope="module")
def pretrained_model(baseline_settings):
    if os.environ.get("RUN_PRETRAINED_TESTS") != "1":
        pytest.skip("Set RUN_PRETRAINED_TESTS=1 to verify real pretrained weights")
    torch.manual_seed(42)
    return make_model(baseline_settings).eval()


def test_pretrained_encoder_matches_checkpoint(pretrained_model, baseline_settings):
    """Compare every encoder tensor against the independently loaded checkpoint."""
    cfg = baseline_settings["model"]
    encoder, info = SegformerModel.from_pretrained(
        cfg["checkpoint"], revision=cfg["revision"], output_loading_info=True,
    )
    assert info["missing_keys"] == []
    assert info["mismatched_keys"] == []
    expected = encoder.state_dict()
    actual = pretrained_model.segformer.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name
    assert pretrained_model.decode_head.classifier.out_channels == 19


@pytest.mark.parametrize("height,width", [(512, 512), (64, 96)])
def test_pretrained_mit_b0_forward(pretrained_model, height, width):
    torch.manual_seed(42)
    with torch.inference_mode():
        logits = pretrained_model(pixel_values=torch.randn(1, 3, height, width)).logits
        upsampled = F.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False,
        )
    assert logits.shape == (1, 19, height // 4, width // 4)
    assert upsampled.shape == (1, 19, height, width)
    assert torch.isfinite(upsampled).all()
    assert pretrained_model.config.semantic_loss_ignore_index == 255


def test_pretrained_loss_ignores_255(pretrained_model):
    """Check actual model loss against CE over only non-void pixels."""
    torch.manual_seed(42)
    labels = torch.randint(0, 19, (1, 64, 96))
    labels[:, :32, :] = 255
    with torch.inference_mode():
        output = pretrained_model(pixel_values=torch.randn(1, 3, 64, 96), labels=labels)
        logits = F.interpolate(output.logits, labels.shape[-2:], mode="bilinear", align_corners=False)
        valid = labels != 255
        expected = F.cross_entropy(logits.permute(0, 2, 3, 1)[valid], labels[valid])
    assert torch.isfinite(output.loss)
    torch.testing.assert_close(output.loss, expected)
