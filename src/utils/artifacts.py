"""Content hashes and atomic JSON writes for reproducible experiment artifacts."""
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def records(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def load_training_state(path):
    """Load project training state with only the known NumPy RNG constructors."""
    import numpy as np
    import torch
    expected = {'numpy.ndarray', 'numpy.dtype', 'numpy._core.multiarray._reconstruct',
                'numpy.core.multiarray._reconstruct'}
    if not set(torch.serialization.get_unsafe_globals_in_checkpoint(path)) <= expected:
        raise ValueError('Unexpected checkpoint pickle globals')
    core = np._core if hasattr(np, '_core') else np.core
    reconstruct = core.multiarray._reconstruct
    with torch.serialization.safe_globals([np.ndarray, np.dtype, type(np.dtype(np.uint32)),
            (reconstruct, 'numpy._core.multiarray._reconstruct'),
            (reconstruct, 'numpy.core.multiarray._reconstruct')]):
        return torch.load(path, map_location='cpu', weights_only=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
