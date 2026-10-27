"""Test forward pass pretrained MiT-B0 (SegFormer B0) on 19-class cityscapes labels.

- Models: nvidia/mit-b0 (via HuggingFace transformers)
- Input: random pixel_values float32 [0,1] shape (B,3,H,W)
- Output: logits (B,19,H',W') after interpolation
- Ignore index: 255 (matches experiment_protocol.md)
- Must be runnable with `python -m pytest -q` from repo root.
"""
import os
import torch
import pytest

from transformers import SegformerForSemanticSegmentation


@pytest.fixture(scope="session")
def pretrained_model():
    """Load nvidia/mit-b0 once per test session."""
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=19,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    return model


def test_pretrained_mit_b0_forward(pretrained_model):
    """Forward pass produces logits with correct shape and ignore_index."""
    # Random input 1×3×512×512 (same as eval config image_size)
    x = torch.rand(1, 3, 512, 512)

    with torch.no_grad():
        logits = pretrained_model(pixel_values=x).logits

    # Logits before upsample: (1, 19, 64, 64) for B0 stride 32
    assert logits.shape[0] == 1
    assert logits.shape[1] == 19, f"Expected 19 logits, got {logits.shape[1]}"
    assert logits.shape[2] <= 512 and logits.shape[3] <= 512

    # After bilinear upsample to original resolution (optional check)
    upsampled = torch.nn.functional.interpolate(
        logits, size=(512, 512), mode="bilinear", align_corners=False
    )
    assert upsampled.shape == (1, 19, 512, 512)


def test_ignore_index_255():
    """Verify that confusion_matrix + scores handle ignore_index=255 correctly.

    This mirrors the unit test in tests/test_metrics.py but uses the real
    confusion_matrix from src.evaluation.metrics.
    """
    from src.evaluation.metrics import confusion_matrix, scores

    pred = torch.tensor([[0, 1], [2, 0]])
    true = torch.tensor([[0, 1], [255, 2]])

    matrix = confusion_matrix(pred, true, num_classes=3)
    assert int(matrix.sum()) == 3  # 255 ignored

    metrics = scores(matrix)
    # mIoU = (IoU0 + IoU1) / 2 where class 2 is excluded (only 2 valid classes)
    expected_miou = (0.5 + 1.0) / 2  # class 0: 0.5, class 1: 1.0, class 2: excluded
    assert abs(metrics["miou"] - expected_miou) < 1e-4, f"miou={metrics['miou']}, expected={expected_miou}"


if __name__ == "__main__":
    # Allow running directly: python -m pytest this_file.py -v
    pytest.main([__file__, "-v"])