import numpy as np
import pytest

from src.evaluation.calibration import CalibrationBins


def test_pixel_ece_hand_computed_and_void_excluded():
    stats = CalibrationBins(bins=2)
    # Low bin: confidence .3, accuracy .5; high bin: .9, accuracy 1.
    stats.update(np.array([.2, .4, .9, 1.]), np.array([0, 0, 1, 18]), np.array([0, 1, 1, 255]))
    r = stats.report()
    assert r["valid_pixels"] == 3
    assert r["pixel_accuracy"] == pytest.approx(2 / 3)
    assert r["ece"] == pytest.approx((2 * .2 + .1) / 3)
    assert r["selective"][0]["coverage"] == pytest.approx(1 / 3)
    assert r["selective"][0]["accuracy"] == 1


def test_confidence_one_last_bin_and_all_void():
    stats = CalibrationBins()
    stats.update(np.ones(2), np.zeros(2), np.array([0, 255]))
    assert stats.report()["bins"][-1]["count"] == 1
    assert stats.report()["ece"] == 0
    assert CalibrationBins().report()["ece"] is None


def test_invalid_confidence_rejected():
    with pytest.raises(ValueError, match="Invalid confidence"):
        CalibrationBins().update(np.array([np.nan]), np.array([0]), np.array([0]))
