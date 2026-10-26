import torch


def confusion_matrix(pred, target, num_classes=19, ignore_index=255):
    pred = pred.reshape(-1).long()
    target = target.reshape(-1).long()
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    if bool(((pred[valid] < 0) | (pred[valid] >= num_classes)).any()):
        raise ValueError("Predicted class IDs must be in [0, num_classes)")
    indices = target[valid] * num_classes + pred[valid]
    return torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)


def scores(conf):
    conf = conf.to(torch.float64)
    true_positive = conf.diag()
    denominator = conf.sum(0) + conf.sum(1) - true_positive
    present = denominator > 0
    iou = torch.full_like(denominator, torch.nan)
    iou[present] = true_positive[present] / denominator[present]
    return {
        "miou": float(iou[present].mean()) if bool(present.any()) else float("nan"),
        "per_class_iou": [None if torch.isnan(value) else float(value) for value in iou],
        "excluded_classes": [index for index, value in enumerate(iou) if torch.isnan(value)],
    }
