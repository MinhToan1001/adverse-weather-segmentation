"""Post-training evaluation of a sealed K2 checkpoint, never target training.

ACDC val has public labels; the official test labels are withheld. Per-run
evaluation requested during development is reported as exploratory validation.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import zipfile

import yaml

from scripts.k2_results import ResultFiles, _summary
from src.data.preparation import create_manifest, write_manifest, convert_label_ids
from src.data.validation import validate_manifests
from src.evaluation.runner import evaluate
from src.utils.artifacts import sha256_file, write_json


CITY_COUNT = 500
ACDC_COUNTS = dict(fog=100, night=106, rain=100, snow=100)
SCOPES = ('cityscapes_val', 'acdc_val')
REPO = Path(__file__).resolve().parents[1]


def resolve_result(hint=None, input_root='/kaggle/input'):
    if hint:
        return Path(hint)
    root = Path(input_root)
    folders = sorted({p.parent for p in root.rglob('k2_plan.json')})
    candidates = list(folders)
    if not folders:
        for path in root.rglob('*.zip'):
            with zipfile.ZipFile(path) as archive:
                if any(Path(n).name == 'k2_plan.json' for n in archive.namelist()):
                    candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f'Set RESULT_INPUT to one completed K2 ZIP/folder; found {len(candidates)} candidates')
    return candidates[0]


def _tree(hint, input_root, folder, required):
    base = Path(hint) if hint else Path(input_root)
    roots = ([base] if all((base/folder/n).is_dir() for n in required) else
             sorted({p.parent for p in base.rglob(folder) if p.is_dir() and all((p/n).is_dir() for n in required)}))
    if len(roots) != 1:
        raise ValueError(f'Attach one {folder} dataset tree or set its root explicitly; found {roots}')
    return roots[0]


def _source_compatible(files, plan):
    # Loading archived Python would execute third-party code. Verify the exact
    # inference implementation and use the checked-in implementation instead.
    required = ['src/models/registry.py', 'src/models/legacy/segformer_b0.py',
                'src/models/proposed/weatherseg_b0.py', 'src/data/preprocessing.py',
                'src/data/dataset.py', 'src/data/labels.py', 'src/evaluation/semantic.py',
                'src/evaluation/runner.py', 'src/evaluation/metrics.py', 'src/evaluation/calibration.py']
    with zipfile.ZipFile(io.BytesIO(files.read('source.zip'))) as snapshot:
        for name in required:
            original = snapshot.read(name)
            if hashlib.sha256(original).hexdigest() != plan['source_hashes'].get(name):
                raise ValueError('Archived inference source is corrupt: ' + name)
            if original.replace(b'\r\n', b'\n') != (REPO/name).read_bytes().replace(b'\r\n', b'\n'):
                raise ValueError('Use the matching inference implementation for this result: ' + name)


def _city_manifest(city, out):
    masks = city
    if len(list((city/'gtFine/val').rglob('*_gtFine_labelTrainIds.png'))) != CITY_COUNT:
        raw = sorted((city/'gtFine/val').rglob('*_gtFine_labelIds.png'))
        if len(raw) != CITY_COUNT:
            raise ValueError(f'Cityscapes val requires {CITY_COUNT} labelIds/trainIds masks')
        masks = out/'converted_city'
        def convert(path):
            destination = masks/path.relative_to(city)
            convert_label_ids(path, destination.with_name(destination.name.replace('_labelIds', '_labelTrainIds')))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(convert, raw))
    rows = create_manifest(city, 'cityscapes', 'val', mask_root=masks, allow_subset=True)
    if len(rows) != CITY_COUNT:
        raise ValueError(f'Cityscapes evaluation requires all {CITY_COUNT} validation images')
    return rows


def evaluate_result(result, out, city=None, acdc=None, acdc_masks=None, input_root='/kaggle/input',
                    scopes=SCOPES, device=None):
    if not scopes or len(set(scopes)) != len(scopes) or set(scopes)-set(SCOPES):
        raise ValueError('Choose unique cityscapes_val/acdc_val scopes; test labels are not public')
    result, out = Path(result).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('Preserve previous evaluation; choose a fresh output directory')
    with ResultFiles(result) as files:
        summary = _summary(files)
        if not summary['complete']:
            raise ValueError('Finish the original training horizon and freeze the candidate before evaluation')
        plan = files.json('k2_plan.json')
        cfg = yaml.safe_load(files.read('train.yaml'))
        _source_compatible(files, plan)
        weights, config = files.read('run/best/model.safetensors'), files.read('run/best/config.json')
    # No target listing, manifest creation or target image/label read precedes
    # the candidate validation above and the immutable evaluation seal below.
    out.mkdir(parents=True)
    checkpoint = out/'checkpoint';checkpoint.mkdir()
    (checkpoint/'model.safetensors').write_bytes(weights)
    (checkpoint/'config.json').write_bytes(config)
    seal = dict(protocol='k2_post_training_validation_v1', arm=summary['arm'], seed=summary['seed'],
        recipe=summary['recipe'], selected_epoch=summary['best_epoch'],
        weights_sha256=sha256_file(checkpoint/'model.safetensors'),
        config_sha256=sha256_file(checkpoint/'config.json'), plan_sha256=summary['plan_sha256'],
        image_size=cfg['data']['image_size'], scopes=list(scopes),
        checkpoint_selection='Cityscapes val only', target_training=False, target_weight_updates=False,
        inference='FP32, native-label geometry, single scale, no adaptation',
        label_scope='ACDC val public labels; not official hidden test',
        development_scope='Exploratory post-training validation; ACDC has historical development feedback')
    write_json(out/'evaluation_seal.json', seal)
    write_json(out/'evaluation_status.json', dict(status='running', completed_scopes=[]))
    reports = {}
    try:
        for scope in scopes:
            if scope == 'cityscapes_val':
                city_root = _tree(city, input_root, 'leftImg8bit', ['val'])
                rows = _city_manifest(city_root, out)
                counts = {'cityscapes':CITY_COUNT}
            else:
                acdc_root = _tree(acdc, input_root, 'rgb_anon', [d+'/val' for d in ACDC_COUNTS])
                mask_root = _tree(acdc_masks or acdc, input_root, 'gt', [d+'/val' for d in ACDC_COUNTS])
                rows = create_manifest(acdc_root, 'acdc', 'val', mask_root=mask_root, allow_subset=True)
                if Counter(r['domain'] for r in rows) != ACDC_COUNTS:
                    raise ValueError('ACDC evaluation requires fog100/night106/rain100/snow100, not a subset or train split')
                counts = ACDC_COUNTS
            manifest = out/(scope+'.csv');write_manifest(rows, manifest)
            validate_manifests([manifest], check_files=True)
            print(f'Evaluating {scope}: {len(rows)} images, frozen epoch {summary["best_epoch"]}', flush=True)
            reports[scope] = evaluate(checkpoint, 'segformer', manifest, out/scope,
                canvas=cfg['data']['image_size'], device=device, expected_counts=counts, expected_split='val')
            print(f'{scope}: mIoU = {100*reports[scope]["miou"]:.2f}%', flush=True)
            for domain, part in reports[scope]['by_domain'].items():
                print(f'  {domain}: {part["n_images"]} images, mIoU = {100*part["miou"]:.2f}%', flush=True)
            if sha256_file(checkpoint/'model.safetensors') != seal['weights_sha256']:
                raise ValueError('Evaluation unexpectedly changed checkpoint weights')
            write_json(out/'evaluation_status.json',dict(status='running',completed_scopes=list(reports)))
        write_json(out/'evaluation_summary.json', dict(seal=seal, metrics=reports))
        write_json(out/'evaluation_status.json',dict(status='complete',completed_scopes=list(reports)))
    except Exception as error:
        write_json(out/'evaluation_status.json',dict(status='failed_preserve_outputs',
            completed_scopes=list(reports), error=str(error)))
        raise
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--city');p.add_argument('--acdc');p.add_argument('--acdc-masks')
    p.add_argument('--input-root',default='/kaggle/input')
    p.add_argument('--scopes',nargs='+',choices=SCOPES,default=list(SCOPES))
    p.add_argument('--device',choices=['cpu','cuda'])
    a=p.parse_args()
    evaluate_result(a.input,a.out,a.city,a.acdc,a.acdc_masks,a.input_root,a.scopes,a.device)


if __name__=='__main__':
    main()
