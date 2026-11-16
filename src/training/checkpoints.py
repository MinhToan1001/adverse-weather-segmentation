"""Transactional epoch checkpoints with verified previous generations and log rollback."""
import json
from pathlib import Path
import shutil

import torch

from src.utils.artifacts import sha256_file, write_json


def _owned(path, root):
    path, root = Path(path), Path(root).resolve()
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError('Checkpoint operation must stay inside its run directory')
    return path


def _remove(path, root):
    path = _owned(path, root)
    if path.exists():
        shutil.rmtree(path) if path.is_dir() else path.unlink()


def save_transactional(model, directory, state):
    directory = Path(directory).resolve()
    if directory.name not in ('last', 'best'):
        raise ValueError('Transactional slots are last/ and best/')
    root = directory.parent
    root.mkdir(parents=True, exist_ok=True)
    pending, previous = root / ('.' + directory.name + '.pending'), root / (directory.name + '.previous')
    _remove(pending, root)
    pending.mkdir()
    model.save_pretrained(pending)
    torch.save(state, pending / 'training_state.pt')
    names = ('config.json', 'model.safetensors', 'training_state.pt')
    write_json(pending / 'checkpoint.json', dict(schema=1, epoch=state['epoch'],
        global_step=state['global_step'], best_epoch=state['best_epoch'],
        files={name: sha256_file(pending / name) for name in names}))
    # In either rename gap a complete generation remains in pending/ or previous/.
    if directory.exists():
        _remove(previous, root)
        _owned(directory, root).rename(_owned(previous, root))
    _owned(pending, root).rename(_owned(directory, root))


def verified_state(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'checkpoint.json').read_text())
    expected = {'config.json', 'model.safetensors', 'training_state.pt'}
    if manifest.get('schema') != 1 or set(manifest['files']) != expected:
        raise ValueError('Invalid checkpoint manifest')
    for name, checksum in manifest['files'].items():
        if sha256_file(directory / name) != checksum:
            raise ValueError('Checkpoint checksum mismatch: ' + str(directory / name))
    state = torch.load(directory / 'training_state.pt', map_location='cpu', weights_only=False)
    if any(state[key] != manifest[key] for key in ('epoch', 'global_step', 'best_epoch')):
        raise ValueError('Checkpoint state and manifest disagree')
    return state


def _rows(path):
    if not path.exists():
        return []
    result = []
    lines = path.read_text(encoding='utf-8').splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError('Corrupt non-final log row: ' + str(path))
            break  # A killed process may have left a partial final JSON line.
    return result


def _promote(selected, primary):
    selected, primary = Path(selected), Path(primary)
    if selected == primary:
        return
    root = primary.parent
    discarded = root / ('.' + primary.name + '.interrupted')
    _remove(discarded, root)
    if primary.exists():
        _owned(primary, root).rename(_owned(discarded, root))
    _owned(selected, root).rename(_owned(primary, root))
    _remove(discarded, root)


def recover_run(run_dir):
    """Recover the newest complete model/state + best + history/resources commit.

    Incomplete work is rolled back to the preceding epoch, never spliced into
    another scheduler. A partial first epoch can safely restart from its seed.
    """
    root = Path(run_dir).resolve()
    if not root.exists():
        return 0
    history, resources = _rows(root / 'history.jsonl'), _rows(root / 'resources.jsonl')
    valid = {'last': [], 'best': []}
    errors = []
    any_slot = False
    for kind in valid:
        for name in (kind, kind + '.previous', '.' + kind + '.pending'):
            path = root / name
            if not path.exists():
                continue
            any_slot = True
            try:
                state = verified_state(path)
                valid[kind].append(dict(path=path, epoch=state['epoch'], best_epoch=state['best_epoch'],
                    global_step=state['global_step'], config=state['config']))
                del state
            except (OSError, ValueError, KeyError, RuntimeError, EOFError) as error:
                errors.append(dict(slot=name, error=str(error)))
    selected = None
    for candidate in sorted(valid['last'], key=lambda item: item['epoch'], reverse=True):
        epoch = candidate['epoch']
        prefix = list(range(1, epoch + 1))
        if ([r['epoch'] for r in history[:epoch]] != prefix
                or [r['epoch'] for r in resources[:epoch]] != prefix
                or candidate['global_step'] != sum(r['optimizer_updates'] for r in resources[:epoch])):
            continue
        best = next((item for item in valid['best'] if item['epoch'] == candidate['best_epoch']
                     and item['config'] == candidate['config']), None)
        if best is not None:
            selected = candidate, best
            break
    if selected is None:
        if resources:
            raise ValueError('No verified checkpoint generation matches committed logs; preserve this workspace and use the previous saved ZIP')
        if any_slot or history:
            for kind in valid:
                for name in (kind, kind + '.previous', '.' + kind + '.pending'):
                    _remove(root / name, root)
            for name in ('history.jsonl', 'resources.jsonl'):
                path = root / name
                if path.exists():
                    path.replace(root / (name + '.interrupted'))
            write_json(root / 'recovery.json', dict(epoch=0, reason='Uncommitted first epoch; restart from original initialization', errors=errors))
        return 0
    current, best = selected
    epoch = current['epoch']
    _promote(current['path'], root / 'last')
    _promote(best['path'], root / 'best')
    for name, rows in (('history.jsonl', history), ('resources.jsonl', resources)):
        path = root / name
        contents = ''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows[:epoch])
        if path.read_text(encoding='utf-8') != contents:
            shutil.copy2(path, root / (name + '.interrupted'))
            temporary = root / (name + '.tmp')
            temporary.write_text(contents, encoding='utf-8')
            temporary.replace(path)
    write_json(root / 'recovery.json', dict(epoch=epoch, last='last', best='best', errors=errors,
        granularity='Completed epochs; an interrupted in-progress epoch is repeated'))
    return epoch
