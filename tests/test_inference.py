from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.dataset import ManifestSegmentationDataset, evaluation_collate
from src.data.labels import PALETTE
from src.data.preparation import write_manifest
from src.data.preprocessing import ImagePreprocessor
from src.inference import SegmentationPredictor
from src.models.segformer import make_model
from src.training.train import evaluate_model


@pytest.fixture(scope="module")
def predictor(tmp_path_factory):
    root = tmp_path_factory.mktemp("inference")
    config = {"model": {"num_labels": 19}, "data": {"image_size": [64, 96]}}
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    torch.manual_seed(42)
    model = make_model({"model": {
        "num_labels": 19, "hidden_sizes": [8, 16, 32, 64], "depths": [1, 1, 1, 1],
        "decoder_hidden_size": 32,
    }})
    model.save_pretrained(root / "checkpoint")
    return SegmentationPredictor(root / "checkpoint", path, device="cpu")


@pytest.mark.parametrize(("hw", "expected_box"), [
    ((31, 97), (16, 0, 31, 96)), ((95, 25), (0, 39, 64, 17)),
    ((64, 96), (0, 0, 64, 96)), ((1, 1000), (31, 0, 1, 96)),
])
def test_fit_geometry_and_dataset_parity(tmp_path, hw, expected_box):
    height, width = hw
    rgb = np.random.default_rng(42).integers(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(rgb)
    image.save(tmp_path / "image.png")
    Image.fromarray(np.zeros(hw, dtype=np.uint8)).save(tmp_path / "mask.png")
    manifest = tmp_path / "manifest.csv"
    write_manifest([dict(image=str(tmp_path / "image.png"), mask=str(tmp_path / "mask.png"),
                         domain="synthetic", split="val")], manifest)
    prep = ImagePreprocessor((64, 96))
    item = ManifestSegmentationDataset(manifest, (64, 96), native_labels=True)[0]
    assert prep.fit_box(image.size) == item["fit_box"] == expected_box
    assert torch.equal(prep.to_tensor(prep.fit_image(image)), item["pixel_values"])


def test_predict_matches_native_evaluator(predictor, tmp_path):
    rgb = np.random.default_rng(42).integers(0, 256, (95, 25, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "image.png")
    Image.fromarray(np.zeros((95, 25), dtype=np.uint8)).save(tmp_path / "mask.png")
    manifest = tmp_path / "manifest.csv"
    write_manifest([dict(image=str(tmp_path / "image.png"), mask=str(tmp_path / "mask.png"),
                         domain="synthetic", split="val")], manifest)
    data = ManifestSegmentationDataset(manifest, (64, 96), native_labels=True)
    captured = []
    evaluate_model(predictor.model, DataLoader(data, collate_fn=evaluation_collate),
                   torch.device("cpu"), 19,
                   prediction_callback=lambda domain, prediction, target, conf: captured.append(prediction))
    result = predictor.predict(tmp_path / "image.png")
    np.testing.assert_array_equal(np.asarray(result.train_ids), captured[0].numpy())
    assert result.train_ids.mode == "L" and result.train_ids.size == (25, 95)
    assert result.color_mask.size == result.overlay.size == result.image.size
    labels = np.asarray(result.train_ids)
    assert labels.dtype == np.uint8 and labels.max() <= 18
    np.testing.assert_array_equal(np.asarray(result.color_mask), PALETTE[labels])
    np.testing.assert_array_equal(np.asarray(result.overlay), ((rgb.astype(np.float32) + PALETTE[labels]) / 2).astype(np.uint8))
    assert result.latency_ms >= result.forward_ms > 0


def test_pil_modes_and_concurrent_calls(predictor):
    images = [Image.new("L", (31, 97), 128), Image.new("RGBA", (97, 31), (200, 50, 100, 255))]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(predictor.predict, images))
    assert [result.image.size for result in results] == [image.size for image in images]
    assert all(result.image.mode == "RGB" for result in results)


def test_hash_mismatch_rejected_before_loading(tmp_path):
    # An existing weight filename exercises the digest guard before parsing weights.
    (tmp_path / "model.safetensors").write_bytes(b"wrong-checkpoint")
    with pytest.raises(ValueError, match="Hash checkpoint"):
        SegmentationPredictor(tmp_path, tmp_path / "unused.yaml", expected_sha256="0" * 64)


def test_invalid_image_type(predictor):
    with pytest.raises(ValueError, match="PIL Image"):
        predictor.predict(np.zeros((10, 10, 3), dtype=np.uint8))


def test_named_checkpoint_uses_its_training_geometry_and_rejects_wrong_variant(tmp_path):
    import json
    from src.models.registry import build_model
    run=tmp_path/'run';checkpoint=run/'best'
    model=build_model({'model':{'name':'segformer_b0','initialization':'random'}})
    model.save_pretrained(checkpoint)
    (run/'config.json').write_text(json.dumps(dict(model=dict(name='segformer_b0',num_labels=19),data=dict(image_size=[64,96]))))
    predictor=SegmentationPredictor(checkpoint,device='cpu',model_name='segformer_b0')
    assert predictor.preprocessor.size==(64,96)
    assert predictor.predict(Image.new('RGB',(57,31))).train_ids.size==(57,31)
    with pytest.raises(ValueError,match='requested b2'):
        SegmentationPredictor(checkpoint,device='cpu',model_name='segformer_b2')
