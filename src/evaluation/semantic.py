"""Family-specific outputs to a common native-resolution 19-class score tensor."""
import torch
import torch.nn.functional as F


def native_logits(output, index, canvas_hw, fit_box, target_hw, family):
    if family in ("segformer", "deeplabv3plus"):
        values = output.logits[index:index + 1]
    else:
        raise ValueError("Unsupported architecture")
    top, left, height, width = fit_box
    values = F.interpolate(values, size=canvas_hw, mode="bilinear", align_corners=False)
    values = values[:, :, top:top + height, left:left + width]
    values = F.interpolate(values, size=target_hw, mode="bilinear", align_corners=False)
    return values
