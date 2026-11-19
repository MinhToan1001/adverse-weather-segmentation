"""Source snapshots and committed-epoch checks shared by DG runners."""
import json
from pathlib import Path
import zipfile

from src.utils.artifacts import load_training_state, records, sha256_file


REPO = Path(__file__).resolve().parents[2]


def fingerprints():
    return {p.relative_to(REPO).as_posix(): sha256_file(p) for directory in ('src', 'scripts', 'configs')
            for p in sorted((REPO / directory).rglob('*')) if p.is_file() and p.suffix in ('.py', '.yaml', '.json')}


def snapshot_source(destination, hashes=None):
    hashes = hashes or fingerprints()
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(REPO / name, name)
        for name in ('requirements.txt', 'THIRD_PARTY_NOTICES.md', 'third_party/transformers/LICENSE'):
            archive.write(REPO / name, name)


def committed_state(run_dir, expected_epoch=None):
    """Only a checkpoint with matching config and log commits is complete."""
    run_dir = Path(run_dir)
    state = load_training_state(run_dir / 'last/training_state.pt')
    config = json.loads((run_dir / 'config.json').read_text())
    epoch = state['epoch']
    usage, history = records(run_dir / 'resources.jsonl'), records(run_dir / 'history.jsonl')
    if ((expected_epoch is not None and epoch != expected_epoch)
            or [r['epoch'] for r in usage] != list(range(1, epoch + 1))
            or [r['epoch'] for r in history] != list(range(1, epoch + 1))
            or state['config'] != config
            or state['global_step'] != sum(r['optimizer_updates'] for r in usage)):
        raise ValueError('Checkpoint/config/logs do not form a committed run')
    return config


def fits_session(now, deadline, recent_epoch_seconds):
    estimate = max(recent_epoch_seconds[-3:] or [2400])
    return now + estimate * 1.5 + 900 <= deadline
