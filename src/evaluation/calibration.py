"""Pixel-weighted reliability diagnostics with explicit void-label exclusion."""
import numpy as np


class CalibrationBins:
    def __init__(self, bins=15):
        self.count = np.zeros(bins, dtype=np.int64)
        self.confidence = np.zeros(bins, dtype=np.float64)
        self.correct = np.zeros(bins, dtype=np.int64)
        self.thresholds = (0.5, 0.7, 0.9, 0.95)
        self.accepted = np.zeros(len(self.thresholds), dtype=np.int64)
        self.accepted_correct = np.zeros(len(self.thresholds), dtype=np.int64)

    def update(self, confidence, prediction, target):
        confidence, prediction, target = map(np.asarray, (confidence, prediction, target))
        if not (confidence.shape == prediction.shape == target.shape):
            raise ValueError("Confidence, prediction and target shapes must agree")
        if not np.isin(target, [*range(19), 255]).all():
            raise ValueError("Invalid trainId target")
        valid = target != 255
        c, p, y = confidence[valid], prediction[valid], target[valid]
        if not np.isfinite(c).all() or ((c < 0) | (c > 1)).any() or not np.isin(p, range(19)).all():
            raise ValueError("Invalid confidence or predicted class")
        correct = p == y
        index = np.minimum((c * len(self.count)).astype(int), len(self.count) - 1)
        self.count += np.bincount(index, minlength=len(self.count))
        self.confidence += np.bincount(index, weights=c, minlength=len(self.count))
        self.correct += np.bincount(index, weights=correct.astype(int), minlength=len(self.count)).astype(np.int64)
        for i, threshold in enumerate(self.thresholds):
            accepted = c >= threshold
            self.accepted[i] += accepted.sum()
            self.accepted_correct[i] += correct[accepted].sum()

    def report(self):
        total = int(self.count.sum())
        average = np.divide(self.confidence, self.count, out=np.zeros_like(self.confidence), where=self.count > 0)
        accuracy = np.divide(self.correct, self.count, out=np.zeros_like(self.confidence), where=self.count > 0)
        return {
            "valid_pixels": total,
            "pixel_accuracy": float(self.correct.sum() / total) if total else None,
            "mean_confidence": float(self.confidence.sum() / total) if total else None,
            "ece": float((np.abs(average - accuracy) * self.count).sum() / total) if total else None,
            "bins": [dict(lower=i / len(self.count), upper=(i + 1) / len(self.count), count=int(n),
                          confidence=float(c) if n else None, accuracy=float(a) if n else None)
                     for i, (n, c, a) in enumerate(zip(self.count, average, accuracy))],
            "selective": [dict(threshold=t, coverage=float(n / total) if total else None,
                               accuracy=float(k / n) if n else None)
                          for t, n, k in zip(self.thresholds, self.accepted, self.accepted_correct)],
        }
