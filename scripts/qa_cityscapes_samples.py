"""Source-only visual sample sheets and actual configured loader checks."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
import torch
import yaml
from src.data.dataset import ManifestSegmentationDataset
from src.data.preparation import resolve_data_path
from src.data.validation import read_manifest

PALETTE=np.array([(128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),(153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),(70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100),(0,80,100),(0,0,230),(119,11,32)],dtype=np.uint8)

def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(Path('configs/segformer_b0_cityscapes.yaml').read_text())
    selected=[];checks=[]
    random.seed(42)
    for split in ['train','val']:
        manifest=cfg['data'][split+'_manifest']
        rows=read_manifest(manifest)
        data=ManifestSegmentationDataset(manifest,cfg['data']['image_size'],training=split=='train',native_labels=split=='val',augment=cfg['data'].get('augment'),preprocessing=cfg['data']['preprocessing'])
        for idx in np.linspace(0,len(rows)-1,10,dtype=int):
            sample=data[int(idx)]
            assert sample['pixel_values'].shape==(3,512,1024)
            assert torch.isfinite(sample['pixel_values']).all()
            assert all(value in [*range(19),255] for value in sample['labels'].unique().tolist())
            checks.append(dict(split=split,index=int(idx),image_shape=list(sample['pixel_values'].shape),mask_shape=list(sample['labels'].shape)))
            selected.append(rows[int(idx)])
    for start in range(0,20,5):
        sheet=Image.new('RGB',(768,5*150),(255,255,255))
        draw=ImageDraw.Draw(sheet)
        for n,row in enumerate(selected[start:start+5]):
            with Image.open(resolve_data_path(row['image'])) as image,Image.open(resolve_data_path(row['mask'])) as mask:
                rgb=image.convert('RGB').resize((256,128))
                labels=np.asarray(mask.resize((256,128),Image.Resampling.NEAREST))
                colors=np.zeros((128,256,3),dtype=np.uint8)
                valid=labels<19;colors[valid]=PALETTE[labels[valid]]
                colored=Image.fromarray(colors)
                overlay=Image.blend(rgb,colored,0.4)
                for col,panel in enumerate([rgb,colored,overlay]):
                    sheet.paste(panel,(col*256,n*150+22))
                draw.text((4,n*150+4),row['split']+' '+Path(row['image']).name,fill='black')
        sheet.save(output/f'samples_{start//5+1}.jpg',quality=90)
    (output/'loader_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    with (output/'selected.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['image','mask','domain','split']);writer.writeheader();writer.writerows(selected)
    print('PASS: 20 real source samples; configured train/val loaders; 4 visual sheets')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);main(p.parse_args().output)
