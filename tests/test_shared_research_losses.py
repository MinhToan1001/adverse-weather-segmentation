import csv
import json
import random

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from src.data.augmentations.class_aware_crop import choose_class_crop
from src.data.class_weights import fit_class_weights
from src.training.losses import SegmentationLoss, make_criterion
from src.utils.artifacts import sha256_file


@pytest.mark.parametrize('name', ['ce', 'weighted_ce', 'focal', 'ce_dice'])
def test_losses_ignore_void_even_if_void_logits_change_and_return_finite_zero_when_all_void(name):
    torch.manual_seed(4)
    x = torch.randn(2, 19, 5, 6, requires_grad=True)
    y = torch.randint(0, 19, (2, 5, 6)); y[:, :2] = 255
    weights = torch.linspace(.5, 2, 19) if name == 'weighted_ce' else None
    criterion = SegmentationLoss(name, class_weights=weights)
    actual = criterion(x, y)
    altered = x.detach().clone(); altered[:, :, :2] = 500
    assert torch.allclose(actual, criterion(altered, y))
    actual.backward()
    assert torch.isfinite(x.grad).all() and not x.grad[:, :, :2].any()
    x.grad = None
    empty = criterion(x, torch.full_like(y, 255)); empty.backward()
    assert empty.item() == 0 and not x.grad.any()


def test_ce_and_gamma_zero_focal_match_pytorch_weighted_ce_value_and_gradients():
    torch.manual_seed(33)
    x = torch.randn(1, 19, 8, 9, requires_grad=True)
    y = torch.randint(0, 19, (1, 8, 9)); y[:, 0] = 255
    weights = torch.linspace(.1, 2, 19)
    expected = F.cross_entropy(x, y, weight=weights, ignore_index=255)
    grad = torch.autograd.grad(expected, x, retain_graph=True)[0]
    for name in ('weighted_ce', 'focal'):
        loss = SegmentationLoss(name, weights, gamma=0)(x, y)
        assert torch.allclose(loss, expected)
        assert torch.allclose(torch.autograd.grad(loss, x, retain_graph=True)[0], grad, atol=1e-7)


def test_focal_discounts_easy_examples_and_dice_does_not_dilute_small_class_by_pixel_area():
    easy = torch.zeros(1, 19, 2, 2); easy[:, 0] = 12
    hard = torch.zeros_like(easy)
    labels = torch.zeros(1, 2, 2, dtype=torch.long)
    assert SegmentationLoss('focal')(easy, labels) < SegmentationLoss('ce')(easy, labels)
    assert SegmentationLoss('ce_dice')(hard, labels) > SegmentationLoss('ce')(hard, labels)


def test_fit_weights_are_bounded_bound_to_manifest_and_reject_validation(tmp_path):
    mask = tmp_path / 'labels.png'
    values = np.zeros((10, 10), dtype=np.uint8); values[0, 0] = 1
    Image.fromarray(values).save(mask)
    image = tmp_path / 'rgb.png'; Image.new('RGB', (10, 10)).save(image)
    path = tmp_path / 'fit.csv'
    row = dict(image=str(image), mask=str(mask), domain='cityscapes', split='train')
    def write():
        with path.open('w', newline='') as h:
            w = csv.DictWriter(h, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    write()
    report = fit_class_weights([path], tmp_path / 'weights.json')
    weights = report['weights']
    assert 1 < weights[1] / weights[0] <= 5.000001
    criterion, _ = make_criterion(dict(name='weighted_ce', class_weights_file=str(tmp_path/'weights.json')), [path])
    assert criterion.weight[1] > criterion.weight[0]
    changed = tmp_path/'other.csv'; changed.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='fit manifests'):
        make_criterion(dict(name='weighted_ce', class_weights_file=str(tmp_path/'weights.json')), [changed])
    row['split'] = 'val'; write()
    with pytest.raises(ValueError, match='fit labels'):
        fit_class_weights([path], tmp_path/'bad.json')


def test_class_aware_crop_preserves_rare_class_near_border():
    values = np.zeros((40, 60), dtype=np.uint8); values[35:39, 55:59] = 18
    crop = choose_class_crop(values, (10, 10), dict(probability=1., min_pixels=8, attempts=10), random.Random(10))
    assert crop['met'] and crop['class_id'] == 18
    selected = values[crop['top']:crop['top']+10, crop['left']:crop['left']+10]
    assert (selected == crop['class_id']).sum() == crop['pixels']
    assert choose_class_crop(np.full_like(values, 255), (10,10), dict(probability=1.), random.Random(1)) is None


def test_dataset_class_crop_applies_same_coordinates_and_flip_to_image_and_mask(tmp_path):
    from src.data.dataset import ManifestSegmentationDataset
    values = np.zeros((20, 30), dtype=np.uint8); values[4:17, 21:29] = 18
    mask, image = tmp_path/'mask.png', tmp_path/'image.png'
    Image.fromarray(values).save(mask)
    Image.fromarray(np.repeat(values[...,None], 3, axis=2)).save(image)
    manifest = tmp_path/'fit.csv'
    with manifest.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image','mask','domain','split'])
        writer.writeheader(); writer.writerow(dict(image=str(image),mask=str(mask),domain='cityscapes',split='train'))
    # target10x15 and multiplier2 preserve the original pixels before cropping.
    ds = ManifestSegmentationDataset(manifest, (10,15), training=True,
        augment=dict(min_scale=2.,max_scale=2.,horizontal_flip=1.,
                     class_aware_crop=dict(probability=1.,min_pixels=16,attempts=10)),
        preprocessing=dict(mean=[0,0,0],std=[1,1,1]))
    random.seed(6)
    sample = ds[0]
    recovered = (sample['pixel_values'][0] * 255).round().long()
    assert torch.equal(recovered, sample['labels'])
    assert sample['crop_constraint_met']
    assert (sample['labels']==sample['crop_class_id']).sum() == sample['crop_class_pixels']
