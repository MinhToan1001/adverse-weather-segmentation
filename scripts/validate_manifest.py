import argparse,csv
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--check-files',action='store_true');p.add_argument('--check-labels',action='store_true');args=p.parse_args()
with open(args.manifest,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
assert rows and {'image','mask','domain','split'}.issubset(rows[0]),'Missing rows or CSV columns'
for row in rows:
    if args.check_files or args.check_labels:
        for col in ['image','mask']:assert Path(row[col]).is_file(),f'Not found: {row[col]}'
    if args.check_labels:
        mask=np.asarray(Image.open(row['mask']))
        assert mask.ndim==2,f'Mask must be single channel: {row["mask"]}'
        assert np.isin(mask,[*range(19),255]).all(),f'Unmapped labels: {row["mask"]}'
print(f'PASS manifest rows={len(rows)}; files={args.check_files}; labels={args.check_labels}')
