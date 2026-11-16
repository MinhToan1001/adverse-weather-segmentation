"""Shared opt-in data strategies and normalization policy, independent of model size."""
from copy import deepcopy

import torch


def set_training_mode(model, freeze_batch_norm_stats=False):
    """BN eval changes train normalization and freezes buffers; affine still learns."""
    model.train()
    if freeze_batch_norm_stats:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def apply_data_strategy(config, *, rare_sampling, photometric, weights_path=None, rgb_settings=None):
    """Compose an arm without mutating the common recipe or changing model identity."""
    result = deepcopy(config)
    data = result["data"]
    if rare_sampling:
        if weights_path is None:
            raise ValueError("Rare sampling requires fit-only weights")
        data["target_sampling_weights"] = str(weights_path)
    else:
        data.pop("target_sampling_weights", None)
    augment = data.setdefault("augment", {})
    if photometric:
        if not rgb_settings:
            raise ValueError("Photometric strategy requires RGB settings")
        augment["photometric"] = deepcopy(rgb_settings)
    else:
        augment.pop("photometric", None)
    return result
