"""Balanced source replay with deterministic per-epoch membership."""
import math
from pathlib import Path
import random

from torch.utils.data import Dataset


def source_scene(row):
    image = Path(row["image"])
    parts = image.name.split("_")
    return (image.parent.name, parts[1] if len(parts) > 2 else image.stem)


def validate_source_monitor(fit, monitor):
    if not fit or not monitor or any(row["domain"] != "cityscapes" or row["split"] != "train" for row in fit + monitor):
        raise ValueError("Replay source fit/monitor must come from Cityscapes official train")
    if {source_scene(r) for r in fit} & {source_scene(r) for r in monitor}:
        raise ValueError("Source fit/monitor scene groups overlap")


def replay_selection_rank(target_miou, source_delta_pp, max_drop_pp=1.0):
    if not all(math.isfinite(value) for value in (target_miou, source_delta_pp, max_drop_pp)) or max_drop_pp < 0:
        raise ValueError("Replay selection values must be finite with a nonnegative source-drop limit")
    eligible = source_delta_pp >= -max_drop_pp
    # If no feasible epoch exists, retain the least source drop as an explicitly ineligible diagnostic fallback.
    return eligible, target_miou if eligible else source_delta_pp, target_miou


def validate_replay_loader_options(options):
    accumulation = int(options["gradient_accumulation_steps"])
    limit = options.get("max_steps")
    if (int(options["batch_size"]) != 1 or accumulation < 2 or accumulation % 2
            or (limit is not None and (int(limit) < 2 or int(limit) % 2))):
        raise ValueError("Balanced replay requires batch_size=1 and positive even accumulation/max_steps")


class BalancedReplayDataset(Dataset):
    def __init__(self, target, source, seed, target_weights=None):
        if len(source) < len(target):
            raise ValueError("Source pool must cover one source sample per target per epoch")
        self.target, self.source, self.seed = target, source, int(seed)
        if target_weights is not None and (len(target_weights)!=len(target)
                or not all(math.isfinite(w) and w>0 for w in target_weights)):
            raise ValueError('Target sampling weights must be finite positive and match fit rows')
        self.target_weights=target_weights
        self.set_epoch(0)

    def __len__(self):
        return 2 * len(self.target)

    def set_epoch(self, epoch):
        self.target.set_epoch(epoch)
        self.source.set_epoch(epoch)
        rng = random.Random(self.seed + 104729 * int(epoch))
        self.target_order = list(range(len(self.target)))
        rng.shuffle(self.target_order)
        if self.target_weights is not None:
            # Preserve source draws and the historical global RNG stream.
            target_rng=random.Random(self.seed+104729*int(epoch)+77773)
            self.target_order=target_rng.choices(range(len(self.target)),weights=self.target_weights,k=len(self.target))
        self.source_order = rng.sample(range(len(self.source)), len(self.target))

    def __getitem__(self, index):
        pair = index // 2
        return self.target[self.target_order[pair]] if index % 2 == 0 else self.source[self.source_order[pair]]
