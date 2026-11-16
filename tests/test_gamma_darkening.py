import random

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import ManifestSegmentationDataset
from src.data.preparation import write_manifest
from src.data.augmentations.gamma import GammaDarkening, darken_rgb
from src.models.segformer import make_model


FORCED_GAMMA = {"probability": 1, "min_gamma": 2, "max_gamma": 2}


def make_manifest(root, domain="cityscapes", split="train"):
    root.mkdir(exist_ok=True)
    mask = np.zeros((48, 96), dtype=np.uint8)
    mask[:, 48:] = 1
    rgb = np.repeat((96 + mask * 80)[:, :, None], 3, axis=2)
    image_path, mask_path = root / "image.png", root / "mask.png"
    Image.fromarray(rgb).save(image_path)
    Image.fromarray(mask).save(mask_path)
    path = root / "manifest.csv"
    write_manifest([
        {"image": str(image_path), "mask": str(mask_path), "domain": domain, "split": split},
    ], path)
    return path


def test_rgb_lookup_preserves_endpoints():
    image = Image.new("RGB", (1, 1), (0, 128, 255))
    assert darken_rgb(image, 2).getpixel((0, 0)) == (0, 64, 255)
    assert darken_rgb(image, 1).tobytes() == image.tobytes()


@pytest.mark.parametrize("options", [
    None,
    {"probability": -1},
    {**FORCED_GAMMA, "unexpected": 1},
    {**FORCED_GAMMA, "probability": float("nan")},
    {**FORCED_GAMMA, "max_gamma": float("inf")},
    {**FORCED_GAMMA, "min_gamma": 0.9},
    {**FORCED_GAMMA, "min_gamma": 2.1},
])
def test_invalid_gamma_options(options):
    with pytest.raises(ValueError, match="gamma"):
        GammaDarkening.from_config(options)


def test_sample_exponent_depends_on_seed_epoch_and_index_without_global_draws():
    gamma = GammaDarkening(0.5, 1, 1.8)
    state = random.getstate()
    first = [gamma.exponent_for(42, 0, i) for i in range(20)]
    assert first == [gamma.exponent_for(42, 0, i) for i in range(20)]
    assert first != [gamma.exponent_for(42, 1, i) for i in range(20)]
    assert first != [gamma.exponent_for(43, 0, i) for i in range(20)]
    assert None in first and any(1 <= x <= 1.8 for x in first if x is not None)
    assert random.getstate() == state


@pytest.mark.parametrize("scale", [0.75, 2.5])
def test_training_mask_geometry_and_rng_parity(tmp_path, scale):
    manifest = make_manifest(tmp_path)
    geometry = {"min_scale": scale, "max_scale": scale, "horizontal_flip": 1}

    def load(options):
        data = ManifestSegmentationDataset(manifest, (64, 96), True, options, seed=42)
        random.seed(17)
        return data[0], random.getstate()

    baseline, original_state = load(geometry)
    disabled, disabled_state = load({
        **geometry, "gamma_darkening": {**FORCED_GAMMA, "probability": 0},
    })
    proposed, proposed_state = load({**geometry, "gamma_darkening": FORCED_GAMMA})
    assert original_state == disabled_state == proposed_state
    assert torch.equal(baseline["pixel_values"], disabled["pixel_values"])
    assert torch.equal(baseline["labels"], disabled["labels"])
    assert torch.equal(baseline["labels"], proposed["labels"])
    padding = baseline["labels"] == 255
    assert torch.equal(baseline["pixel_values"][:, padding], proposed["pixel_values"][:, padding])
    assert torch.all(proposed["pixel_values"][:, ~padding] < baseline["pixel_values"][:, ~padding])
    if scale < 1:
        assert padding.any()


@pytest.mark.parametrize(("domain", "training"), [("cityscapes", False), ("night", False), ("synthetic", True)])
def test_gamma_only_affects_cityscapes_train(tmp_path, domain, training):
    manifest = make_manifest(tmp_path, domain, "train" if training else "val")
    args = dict(image_size=(64, 96), training=training, native_labels=not training)
    random.seed(17)
    baseline = ManifestSegmentationDataset(manifest, **args)[0]
    random.seed(17)
    proposed = ManifestSegmentationDataset(manifest, augment={"gamma_darkening": FORCED_GAMMA}, **args)[0]
    assert torch.equal(baseline["pixel_values"], proposed["pixel_values"])
    assert torch.equal(baseline["labels"], proposed["labels"])
    assert baseline.get("fit_box") == proposed.get("fit_box")


def test_epoch_state_can_be_recreated_at_resume(tmp_path):
    manifest = make_manifest(tmp_path)
    settings = {"probability": 1, "min_gamma": 1, "max_gamma": 3}
    args = dict(image_size=(64, 96), training=True, augment={"gamma_darkening": settings}, seed=42)
    original = ManifestSegmentationDataset(manifest, **args)
    resumed = ManifestSegmentationDataset(manifest, **args)
    random.seed(17)
    epoch_zero = original[0]
    original.set_epoch(7)
    resumed.set_epoch(7)
    random.seed(17)
    expected = original[0]
    random.seed(17)
    actual = resumed[0]
    assert torch.equal(expected["pixel_values"], actual["pixel_values"])
    assert torch.equal(expected["labels"], epoch_zero["labels"])
    assert not torch.equal(expected["pixel_values"], epoch_zero["pixel_values"])


def test_proposed_forward_backward_keeps_model_shape_and_ignore_index(tmp_path):
    manifest = make_manifest(tmp_path)
    data = ManifestSegmentationDataset(
        manifest, (64, 96), training=True, augment={"gamma_darkening": FORCED_GAMMA},
    )
    batch = data[0]
    torch.manual_seed(42)
    model = make_model({"model": {"num_labels": 19}})
    result = model(pixel_values=batch["pixel_values"][None], labels=batch["labels"][None])
    assert model.config.semantic_loss_ignore_index == 255
    assert result.logits.shape == (1, 19, 16, 24)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    for parameter in (next(model.segformer.parameters()), model.decode_head.classifier.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
