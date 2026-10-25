from pathlib import Path
import csv
import numpy as np
from PIL import Image
root=Path('data/processed/synthetic');root.mkdir(parents=True,exist_ok=True)
Path('data/manifests').mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(42)
for split,n in [('train',8),('val',4)]:
    rows=[]
    for i in range(n):
        image=root/f'{split}_{i:03d}_rgb.png'; mask=root/f'{split}_{i:03d}_mask.png'
        Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8),'RGB').save(image)
        Image.fromarray(rng.integers(0,19,(64,64),dtype=np.uint8),'L').save(mask)
        rows.append({'image':str(image),'mask':str(mask),'domain':'synthetic','split':split})
    with open(f'data/manifests/synthetic_{split}.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['image','mask','domain','split']);w.writeheader();w.writerows(rows)
print('Created synthetic manifests; NOT suitable for real research.')
