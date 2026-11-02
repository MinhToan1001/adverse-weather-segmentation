"""Hand-calculated cases; no dataset or pretrained download required."""
import math

import pytest
import torch

from src.evaluation.metrics import confusion_matrix, scores


def test_ignore_label_and_hand_calculated_iou():
    pred = torch.tensor([[0, 1], [2, 0]])
    true = torch.tensor([[0, 1], [255, 2]])
    matrix = confusion_matrix(pred, true, 3)
    assert torch.equal(matrix, torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]]))
    metrics = scores(matrix)
    assert metrics["per_class_iou"] == [0.5, 1.0, 0.0]
    assert metrics["miou"] == pytest.approx(0.5)
    assert metrics["excluded_classes"] == []


def test_perfect_19_class_prediction():
    labels = torch.arange(19).reshape(1, 19)
    matrix = confusion_matrix(labels, labels)
    assert torch.equal(matrix, torch.eye(19, dtype=torch.long))
    assert scores(matrix)["miou"] == 1.0


def test_completely_missed_class_is_zero_not_excluded():
    metrics = scores(confusion_matrix(torch.tensor([0, 0]), torch.tensor([0, 1]), 3))
    assert metrics["per_class_iou"] == [0.5, 0.0, None]
    assert metrics["excluded_classes"] == [2]
    assert metrics["miou"] == 0.25


def test_false_positive_only_class_counts_in_mean():
    metrics = scores(confusion_matrix(torch.tensor([0, 1]), torch.tensor([0, 0]), 3))
    assert metrics["per_class_iou"] == [0.5, 0.0, None]
    assert metrics["excluded_classes"] == [2]
    assert metrics["miou"] == 0.25


@pytest.mark.parametrize("ignored_prediction", [0, 18, 255, -1])
def test_predictions_on_void_do_not_change_scores(ignored_prediction):
    matrix = confusion_matrix(torch.tensor([0, 1, ignored_prediction]), torch.tensor([0, 1, 255]))
    assert matrix.sum().item() == 2
    assert scores(matrix)["miou"] == 1.0


def test_all_void_has_no_score():
    matrix = confusion_matrix(torch.tensor([0, 18]), torch.tensor([255, 255]))
    metrics = scores(matrix)
    assert matrix.sum().item() == 0
    assert math.isnan(metrics["miou"])
    assert metrics["per_class_iou"] == [None] * 19
    assert metrics["excluded_classes"] == list(range(19))


def test_global_confusion_accumulation_not_mean_of_image_ious():
    first = confusion_matrix(torch.tensor([0]), torch.tensor([0]), 2)
    second = confusion_matrix(torch.tensor([1, 1, 1]), torch.tensor([0, 0, 0]), 2)
    assert scores(first)["miou"] == 1.0
    assert scores(second)["miou"] == 0.0
    assert scores(first + second)["miou"] == 0.125


@pytest.mark.parametrize("invalid_prediction", [-1, 19, 255])
def test_invalid_prediction_on_valid_target_raises(invalid_prediction):
    with pytest.raises(ValueError, match="Predicted class IDs"):
        confusion_matrix(torch.tensor([invalid_prediction]), torch.tensor([0]))
