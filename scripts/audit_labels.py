"""Source pixel QA; target manifests are metadata-only before frozen evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.data.preparation import LABEL_IDS, resolve_data_path
from src.data.validation import read_manifest, validate_manifests


def audit(manifests, check_files=False, check_labels=False, raw_root=None):
    rows = [row for path in manifests for row in read_manifest(path)]
    # Includes cross-manifest checks and refuses target pixel audits before I/O.
    validate_manifests(manifests, check_files=check_files or check_labels)
    if check_labels and any(row['domain'] != 'cityscapes' for row in rows):
        raise ValueError('Pixel audit supports Cityscapes only; ACDC pixels stay unopened')
    totals, hashes = {}, {}
    errors = []
    def check(row):
        image_path, mask_path = [resolve_data_path(row[key]) for key in ['image','mask']]
        counts = None
        digest = None
        if check_labels:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                image.load()
                values = np.asarray(mask)
                if mask.mode not in {'L','I','I;16','I;16L','I;16B'} or values.ndim != 2:
                    raise ValueError(f'Not an integer mask: {mask_path}')
                if image.size != mask.size or not (((values>=0)&(values<19))|(values==255)).all():
                    raise ValueError(f'Invalid dimensions/labels: {mask_path}')
                if raw_root:
                    raw = Path(raw_root)/'gtFine'/row['split']/mask_path.parent.name/mask_path.name.replace('_labelTrainIds.png','_labelIds.png')
                    with Image.open(raw) as original:
                        ids = np.asarray(original)
                        expected = np.full(ids.shape,255,dtype=np.uint8)
                        for train_id, label_id in enumerate(LABEL_IDS):
                            expected[ids == label_id] = train_id
                    if not np.array_equal(expected, values):
                        raise ValueError(f'Mapping mismatch: {mask_path}')
                counts = np.bincount(values.reshape(-1), minlength=256).astype(np.int64)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        return row, counts, digest
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(check, rows), 1):
            row, counts, digest = result
            split = row['domain']+'/'+row['split']
            totals.setdefault(split, dict(images=0, class_pixels=np.zeros(256,dtype=np.int64)))
            totals[split]['images'] += 1
            if counts is not None:
                totals[split]['class_pixels'] += counts
            if digest:
                if digest in hashes:
                    errors.append(f'Duplicate image content: {row["image"]} and {hashes[digest]}')
                hashes[digest] = row['image']
            if i%250 == 0:
                print(f'Audited {i}/{len(rows)}', flush=True)
    return dict(status='PASS' if not errors else 'FAIL', rows=len(rows), errors=errors,
        pixel_checks=check_labels, mapping_compared_to_raw=bool(raw_root and check_labels),
        image_sha256_duplicate_check=check_labels,
        manifests={str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in manifests},
        splits={key:dict(images=value['images'],class_pixels={str(i):int(value['class_pixels'][i]) for i in [*range(19),255]}) for key,value in totals.items()})

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',nargs='+',required=True)
    p.add_argument('--check-files',action='store_true')
    p.add_argument('--check-labels',action='store_true')
    p.add_argument('--raw-root')
    p.add_argument('--out',required=True)
    args=p.parse_args()
    try:
        report=audit(args.manifest,args.check_files,args.check_labels,args.raw_root)
    except (OSError,ValueError) as error:
        report=dict(status='FAIL',errors=[str(error)])
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='splits'},indent=2))
    sys.exit(0 if report['status']=='PASS' else 1)
