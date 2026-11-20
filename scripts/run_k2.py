"""Independent K2 source-only experiment; one selected arm per Kaggle session."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml
import numpy as np
from PIL import Image

from scripts.run_dg import prepare_cityscapes
from src.experiments.packaging import export_workspace
from src.experiments.protocol import require_training_permission
from src.experiments.run_support import fingerprints, snapshot_source, committed_state, fits_session
from src.training.checkpoints import recover_run
from src.utils.artifacts import sha256_file, records, write_json
from src.data.validation import read_manifest
from src.data.preparation import resolve_data_path
from src.data.augmentations.weather import CONDITIONS, VERSION


ARMS = ('baseline', 'film_clean', 'aug', 'film', 'k2')
CLEAN_ARMS = ('baseline', 'film_clean')
PROTOCOL = 'weatherseg_k2_source_v1'
RECIPE = Path(__file__).resolve().parents[1] / 'configs/proposed/k2_film_b0.yaml'
RECIPES = ('legacy_v1', 'balanced_v2')


def compose(arm, paths, output, seed=42, epochs=20, batch_size=2, weather=None, recipe='legacy_v1'):
    if arm not in ARMS or recipe not in RECIPES or min(epochs, batch_size) < 1:
        raise ValueError('Choose baseline/film_clean/aug/film/k2 and positive epochs/batch_size')
    cfg = yaml.safe_load(RECIPE.read_text(encoding='utf-8'))
    cfg.update(seed=seed, experiment=f'k2-{arm}-seed{seed}', output_dir=str(Path(output).resolve()))
    cfg['data'].update(train_manifest=paths['source_train_manifest'], val_manifest=paths['source_monitor_manifest'])
    cfg['train'].update(epochs=epochs, batch_size=batch_size)
    cfg['condition_training'] = arm == 'k2'
    if arm in ('baseline', 'aug'):
        cfg['model'] = dict(name='segformer_b0', num_labels=19, initialization='imagenet_encoder')
    elif arm in ('film_clean', 'film'):
        cfg['model']['condition_loss_weight'] = 0.
    if arm in CLEAN_ARMS:
        cfg['data']['augment'].pop('weather')
    elif weather is not None:
        cfg['data']['augment']['weather'] = weather
    if recipe == 'balanced_v2':
        # Opt-in candidate recipe, not a claim of improved accuracy. Keep the
        # old model/trainer and legacy recipe unchanged for historical controls.
        cfg['k2_recipe'] = recipe
        cfg['data']['image_size'] = [640, 1280]
        cfg['train'].update(warmup_steps=500, num_workers=4)
        cfg['data']['augment']['class_aware_crop'] = dict(probability=.5, min_pixels=64, attempts=5)
        if arm not in CLEAN_ARMS:
            policy = dict(cfg['data']['augment']['weather'])
            conditions = policy['conditions']
            if 'clean' not in conditions or len(conditions) < 2:
                raise ValueError('Balanced K2 requires clean and at least one adverse condition')
            policy.update(severities=[1, 2, 3], probabilities=[
                .5 if condition == 'clean' else .5/(len(conditions)-1) for condition in conditions])
            cfg['data']['augment']['weather'] = policy
    return cfg


def prepare_weather(city, train_manifest, mode='auto', depth_root=None, distance_root=None, invalid_policy='far'):
    """Check every source sidecar before GPU training; never invent missing depth."""
    if mode not in {'auto', 'full', 'no-fog'} or invalid_policy not in {'error', 'far'}:
        raise ValueError('Invalid weather mode or stereo invalid policy')
    if depth_root and distance_root:
        raise ValueError('Choose stereo depth_root OR completed metric distance_root')
    city = Path(city).resolve()
    stereo = Path(depth_root).resolve() if depth_root else city
    distances = Path(distance_root).resolve() if distance_root else None
    has_stereo = (stereo/'disparity/train').is_dir() or (stereo/'camera/train').is_dir()
    fog = mode != 'no-fog' and (distances is not None or has_stereo or depth_root is not None)
    if mode == 'full' and not fog:
        raise ValueError('Full five-condition training needs Cityscapes disparity+camera or metric distance NPY files')
    rows = read_manifest(train_manifest)
    if any(row['domain'] != 'cityscapes' or row['split'] != 'train' for row in rows):
        raise ValueError('K2 weather preflight accepts Cityscapes train only')
    conditions = list(CONDITIONS) if fog else ['clean', 'night', 'rain', 'snow']
    weather = dict(conditions=conditions, severities=[1,2,3,4,5], probabilities=[1/len(conditions)]*len(conditions))
    if fog:
        weather.update(invalid_policy=invalid_policy, max_distance_m=200.)
        if distances:
            weather.update(distance_root=str(distances), distance_unit='meters')
        else:
            weather['cityscapes_root'] = str(stereo)
        for row in rows:
            image_path = resolve_data_path(row['image'])
            with Image.open(image_path) as image:
                expected = image.size
            if distances:
                path = distances/(image_path.stem+'.npy')
                values = np.load(path, mmap_mode='r', allow_pickle=False)
                if values.shape != (expected[1], expected[0]) or values.dtype.kind != 'f':
                    raise ValueError('Distance must be aligned floating-point camera-ray meters: '+str(path))
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError('Distance contains invalid/negative values: '+str(path))
                del values
            else:
                stem = image_path.stem.removesuffix('_leftImg8bit')
                location = stem.rsplit('_',2)[0]
                disparity = stereo/'disparity/train'/location/(stem+'_disparity.png')
                camera = stereo/'camera/train'/location/(stem+'_camera.json')
                with Image.open(disparity) as values:
                    if values.size != expected:
                        raise ValueError('Disparity must match native RGB geometry: '+str(disparity))
                # Validate camera and disparity encoding/invalids once, before the first epoch.
                from src.data.augmentations.weather import cityscapes_distance
                cityscapes_distance(disparity, camera, invalid_policy=invalid_policy)
    report = dict(version=VERSION, requested_mode=mode, conditions=conditions,
        condition_ids=[CONDITIONS.index(c) for c in conditions], checked_depth_images=len(rows) if fog else 0,
        fog_enabled=fog, depth_source='completed_metric_npy' if distances and fog else 'cityscapes_stereo' if fog else None,
        invalid_policy=invalid_policy if fog and not distances else None,
        note='Condition IDs identify K1 transforms, not real-weather classification accuracy')
    print('K1 training conditions:', ', '.join(conditions), flush=True)
    if not fog:
        print('Fog is excluded: no metric-depth training examples; the five-logit head has no positive fog labels.', flush=True)
    return weather, report


def prepare(city, root, arm='k2', seed=42, epochs=20, batch_size=2,
            weather_mode='auto', depth_root=None, distance_root=None, invalid_policy='far', recipe='legacy_v1'):
    root, city = Path(root).resolve(), Path(city).resolve()
    compose(arm, dict(source_train_manifest='', source_monitor_manifest=''), root/'run', seed, epochs, batch_size, recipe=recipe)
    identity = dict(protocol=PROTOCOL, arm=arm, seed=seed, epochs=epochs, batch_size=batch_size,
                    city_root=str(city), source_hashes=fingerprints(), target_data_access='none',
                    weather_inputs=dict(mode=weather_mode, depth_root=str(Path(depth_root).resolve()) if depth_root else None,
                        distance_root=str(Path(distance_root).resolve()) if distance_root else None, invalid_policy=invalid_policy))
    if recipe != 'legacy_v1':
        identity['recipe'] = recipe
    if (root/'k2_plan.json').is_file():
        plan = json.loads((root/'k2_plan.json').read_text())
        if any(plan.get(k) != v for k, v in identity.items()):
            raise ValueError('Resume K2 using the original source snapshot, arm, seed, recipe and Cityscapes mount')
        for name, checksum in plan['frozen_files'].items():
            if sha256_file(root/name) != checksum:
                raise ValueError('K2 frozen config/manifest changed: ' + name)
        if not (root/'source.zip').is_file():
            snapshot_source(root/'source.zip', identity['source_hashes'])
        cfg = yaml.safe_load((root/'train.yaml').read_text())
        require_training_permission(cfg)
        return cfg
    if root.exists() and any(root.iterdir()):
        raise ValueError('Preserve existing K2 outputs; choose a fresh workspace')
    root.mkdir(parents=True, exist_ok=True)
    paths = prepare_cityscapes(city, root/'data')
    if arm not in CLEAN_ARMS:
        weather, preflight = prepare_weather(city, paths['source_train_manifest'], weather_mode,
                                             depth_root, distance_root, invalid_policy)
    else:
        weather, preflight = None, dict(conditions=['clean'], fog_enabled=False, checked_depth_images=0)
    cfg = compose(arm, paths, root/'run', seed, epochs, batch_size, weather, recipe)
    if recipe != 'legacy_v1':
        preflight.update(recipe=recipe, actual_weather_policy=cfg['data']['augment'].get('weather'))
    require_training_permission(cfg)
    write_json(root/'weather_preflight.json', preflight)
    (root/'train.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    snapshot_source(root/'source.zip', identity['source_hashes'])
    files = ['train.yaml', 'data/cityscapes_train.csv', 'data/cityscapes_val.csv', 'weather_preflight.json']
    write_json(root/'k2_plan.json', dict(identity,
        frozen_files={name: sha256_file(root/name) for name in files},
        condition_labels='K1 synthetic transform IDs; metric scattering fog only when depth preflight passes',
        selection='Cityscapes val500 mIoU; target evaluation is not run by this launcher'))
    return cfg


def train_epoch(command, epoch, epochs, heartbeat_seconds=60.):
    """Show liveness without touching the trainer, its RNG, or its recipe."""
    if heartbeat_seconds <= 0:
        raise ValueError('Heartbeat interval must be positive')
    started = time.monotonic()
    print(f'[K2] Starting epoch {epoch}/{epochs}; metrics follow training + validation.', flush=True)
    with subprocess.Popen(command) as process:
        while True:
            try:
                code = process.wait(timeout=heartbeat_seconds)
                break
            except subprocess.TimeoutExpired:
                elapsed = (time.monotonic() - started) / 60
                print(f'[K2] Epoch {epoch}/{epochs}: training/validation process is still running '
                      f'({elapsed:.1f} min elapsed).', flush=True)
        if code:
            raise subprocess.CalledProcessError(code, command)


def publish_review(root, archive):
    """Review packaging must never prevent saving a full failure/resume archive."""
    from scripts.k2_results import export_share
    try:
        return export_share(root, archive.with_name(archive.stem + '_share.zip'))
    except Exception as error:
        write_json(root/'share_export_error.json', dict(error=str(error), full_archive=str(archive)))
        print('[K2] Review ZIP could not be updated:', error,
              '| preserving the full resume archive.', flush=True)
        return None


def run(city, root, archive, arm='k2', seed=42, epochs=20, batch_size=2, session_hours=9., prepare_only=False,
        weather_mode='auto', depth_root=None, distance_root=None, invalid_policy='far', recipe='legacy_v1'):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    if not 0 < session_hours <= 10 or archive.is_relative_to(root):
        raise ValueError('Session window must be (0,10] hours and archive outside workspace')
    started = time.monotonic()
    cfg = prepare(city, root, arm, seed, epochs, batch_size, weather_mode, depth_root, distance_root, invalid_policy, recipe)
    if prepare_only:
        export_workspace(root, archive)
        return
    import torch
    if not torch.cuda.is_available():
        raise ValueError('Enable GPU for K2 training, or use --prepare-only')
    run_dir = Path(cfg['output_dir'])
    saved = recover_run(run_dir)
    if saved:
        committed_state(run_dir, saved)
    status = 'running'
    try:
        for epoch in range(saved + 1, epochs + 1):
            usage = records(run_dir/'resources.jsonl')
            if not fits_session(time.monotonic(), started + session_hours * 3600,
                                [r['epoch_seconds'] for r in usage]):
                status = 'session_checkpoint_ready'
                break
            command = [sys.executable, '-u', '-m', 'src.training.train', '--config', str(root/'train.yaml'),
                       '--stop-after-epoch', str(epoch)]
            if saved:
                command += ['--resume', str(run_dir/'last')]
            train_epoch(command, epoch, epochs)
            committed_state(run_dir, epoch)
            saved = epoch
            write_json(root/'status.json', dict(protocol=PROTOCOL, status='running', completed_epochs=saved))
            publish_review(root, archive)
            export_workspace(root, archive)
            print('K2 epoch', saved, 'saved to', archive, flush=True)
        else:
            status = 'training_complete'
            write_json(root/'frozen_candidate.json', dict(protocol=PROTOCOL, arm=arm, seed=seed,
                completed_epochs=epochs, target_training=False, selected_by='Cityscapes val500 only',
                weights_sha256=sha256_file(run_dir/'best/model.safetensors'),
                config_sha256=sha256_file(run_dir/'best/config.json'),
                plan_sha256=sha256_file(root/'k2_plan.json')))
    except Exception:
        status = 'failed_preserve_outputs'
        raise
    finally:
        write_json(root/'status.json', dict(protocol=PROTOCOL, status=status, completed_epochs=saved,
                                         elapsed_seconds=time.monotonic()-started))
        publish_review(root, archive)
        export_workspace(root, archive)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--arm', choices=ARMS, default='k2')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--recipe', choices=RECIPES, default='legacy_v1')
    parser.add_argument('--epochs', type=int, help='Default: legacy20, balanced40; frozen on resume')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--session-hours', type=float, default=9.)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--weather-mode', choices=['auto','full','no-fog'], default='auto')
    parser.add_argument('--depth-root', help='Root containing disparity/train and camera/train')
    parser.add_argument('--distance-root', help='Completed camera-ray distance NPY folder, meters')
    parser.add_argument('--invalid-policy', choices=['error','far'], default='far')
    a = parser.parse_args()
    epochs = a.epochs if a.epochs is not None else (40 if a.recipe == 'balanced_v2' else 20)
    run(a.city, a.root, a.archive, a.arm, a.seed, epochs, a.batch_size, a.session_hours, a.prepare_only,
        a.weather_mode, a.depth_root, a.distance_root, a.invalid_policy, a.recipe)
