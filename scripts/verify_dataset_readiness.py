"""Reproducible manifests, provenance and source/target metadata readiness."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.preparation import create_manifest,write_manifest,resolve_data_path
from src.data.validation import read_manifest,validate_manifests
from src.data.dataset import ManifestSegmentationDataset


def verify(out):
    cases=[('cityscapes','train','data/raw/cityscapes','data/processed/cityscapes'),('cityscapes','val','data/raw/cityscapes','data/processed/cityscapes'),('acdc','val','data/raw/ACDC/rgb_anon_trainvaltest','data/raw/ACDC/gt_trainval')]
    checks={}
    # Any accidental pixel decoding makes this metadata stage fail.
    with patch('PIL.Image.open',side_effect=AssertionError('Metadata stage must not open pixels')):
        with tempfile.TemporaryDirectory() as temp:
            for dataset,split,root,masks in cases:
                rows=create_manifest(root,dataset,split,masks)
                original=Path('data/manifests')/f'{dataset}_{split}.csv'
                generated=Path(temp)/original.name
                write_manifest(rows,generated)
                assert original.read_bytes()==generated.read_bytes(),original
                checks[str(original)]={'rows':len(rows),'sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'regenerated_identically':True,'image_bytes':sum(resolve_data_path(r['image']).stat().st_size for r in rows),'mask_bytes':sum(resolve_data_path(r['mask']).stat().st_size for r in rows)}
        total=validate_manifests(list(checks),check_files=True)
        target=ManifestSegmentationDataset('data/manifests/acdc_val.csv',native_labels=True)
        assert len(target)==406
        assert {r['split'] for r in target.items}=={'val'}
        for path in ['data/manifests/cityscapes_train.csv','data/manifests/cityscapes_val.csv']:
            rows=read_manifest(path)
            assert {r['domain'] for r in rows}=={'cityscapes'}
    provenance={}
    for path in [Path('data/raw/cityscapes/README'),Path('data/raw/cityscapes/license.txt'),*Path('data/raw/ACDC').glob('*/README.md'),*Path('data/raw/ACDC').glob('*/License.pdf')]:
        provenance[str(path)]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    report=dict(status='PASS_PRE_FREEZE_READINESS',combined_pairs=total,manifests=checks,provenance=provenance,acdc_constructor_metadata_only=True,acdc_pixels_opened=False,limits=['Target pixel validation and target iteration deferred to frozen evaluation under DG protocol','Official release identifiers and original provider checksums unavailable; local metadata fingerprints do not certify origin','Gate sign-off is separate from task artifacts'])
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);verify(p.parse_args().out)
