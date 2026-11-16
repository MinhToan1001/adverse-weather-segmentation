"""Prepare shared Cityscapes/ACDC manifests once for every research model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from src.data.partitions import prepare_source_split as source_split, prepare_target_split as target_split
from src.data.preparation import create_manifest, write_manifest, convert_label_ids
from src.utils.artifacts import sha256_file, write_json


def prepare(city, acdc, output):
    city, acdc, output = Path(city).resolve(), Path(acdc).resolve(), Path(output).resolve()
    intent = dict(city=str(city), acdc=str(acdc))
    ready = output / 'data.json'
    if ready.exists():
        saved = json.loads(ready.read_text())
        if saved['roots'] != intent:
            raise ValueError('Use the same dataset mounts or prepare a separate data directory')
        for name, checksum in saved['manifest_sha256'].items():
            if sha256_file(output / name) != checksum:
                raise ValueError('Shared manifest changed')
        return saved
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve partial preparation; use a fresh directory')
    output.mkdir(parents=True)
    raw = sorted((city / 'gtFine/train').rglob('*_gtFine_labelIds.png'))
    if len(raw) == 2975:
        masks = output / 'city_train'
        def convert(path):
            relative = path.relative_to(city)
            convert_label_ids(path, masks / relative.parent / relative.name.replace('_labelIds.png', '_labelTrainIds.png'))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(convert, raw))
    else:
        masks = city
    manifest = output / 'city_train.csv'
    write_manifest(create_manifest(city, 'cityscapes', 'train', mask_root=masks), manifest)
    source_split(manifest, output / 'source')
    target_split(acdc / 'rgb_anon_trainvaltest/rgb_anon', acdc / 'gt_trainval/gt', output / 'target')
    paths = dict(source_train_manifest=str(output / 'source/cityscapes_fit.csv'),
                 source_monitor_manifest=str(output / 'source/cityscapes_monitor.csv'),
                 train_manifest=str(output / 'target/acdc_train_fit.csv'),
                 val_manifest=str(output / 'target/acdc_train_dev.csv'))
    from src.data.validation import read_manifest
    source_examples = read_manifest(paths['source_train_manifest'])[:16]
    target_examples = read_manifest(paths['train_manifest'], allow_target_train=True)
    target_examples = [next(row for row in target_examples if row['domain'] == domain)
                       for domain in ('fog', 'night', 'rain', 'snow')]
    write_manifest(source_examples, output / 'source_train_eval.csv')
    write_manifest(source_examples[:4] + target_examples, output / 'replay_train_eval.csv')
    paths['source_train_eval_manifest'] = str(output / 'source_train_eval.csv')
    paths['replay_train_eval_manifest'] = str(output / 'replay_train_eval.csv')
    report = dict(roots=intent, paths=paths,
                  manifest_sha256={p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob('*.csv')},
                  scope='Source fit2787/monitor188; target fit1198/dev402. Official-val never selects source or adaptation best.')
    write_json(ready, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', required=True)
    parser.add_argument('--acdc', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.city, args.acdc, args.out), indent=2))
