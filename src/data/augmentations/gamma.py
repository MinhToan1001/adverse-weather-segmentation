import hashlib
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class GammaDarkening:
    probability: float
    min_gamma: float
    max_gamma: float

    @classmethod
    def from_config(cls, options):
        if not isinstance(options, Mapping) or set(options) != {"probability", "min_gamma", "max_gamma"}:
            raise ValueError("gamma_darkening requires probability, min_gamma and max_gamma")
        try:
            probability, min_gamma, max_gamma = (
                float(options[key]) for key in ("probability", "min_gamma", "max_gamma")
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("gamma_darkening values must be finite numbers") from exc
        if not (
            0 <= probability <= 1 and 1 <= min_gamma <= max_gamma and math.isfinite(max_gamma)
        ):
            raise ValueError("Require gamma probability in [0, 1] and finite 1 <= min_gamma <= max_gamma")
        return cls(probability, min_gamma, max_gamma)

    def exponent_for(self, seed, epoch, index):
        key = f"{seed}:{epoch}:{index}".encode("ascii")
        rng = random.Random(int.from_bytes(hashlib.blake2b(key, digest_size=16).digest(), "big"))
        if rng.random() < self.probability:
            return rng.uniform(self.min_gamma, self.max_gamma)
        return None


def darken_rgb(image, exponent):
    lut = [round(255 * (value / 255) ** exponent) for value in range(256)]
    return image.point(lut * 3)
