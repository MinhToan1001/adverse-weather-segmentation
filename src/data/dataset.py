import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class ManifestSegmentationDataset(Dataset):
    """Masks must already be 0..18 / 255 trainId maps, not raw Cityscapes labelId RGB colors."""
    def __init__(self, manifest, image_size=(512, 512)):
        with open(manifest, newline='', encoding='utf-8') as f:
            self.items=list(csv.DictReader(f))
        if not self.items: raise ValueError(f'Empty manifest: {manifest}')
        self.size=tuple(image_size)
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        row=self.items[idx]
        image=Image.open(row['image']).convert('RGB').resize((self.size[1],self.size[0]), Image.Resampling.BILINEAR)
        mask=Image.open(row['mask']).resize((self.size[1],self.size[0]), Image.Resampling.NEAREST)
        pixels=torch.from_numpy(np.array(image, dtype=np.float32).transpose(2,0,1) /255.)
        pixels=(pixels-torch.tensor([0.485,0.456,0.406])[:,None,None])/torch.tensor([0.229,0.224,0.225])[:,None,None]
        labels=torch.from_numpy(np.array(mask,dtype=np.int64))
        valid=((labels>=0)&(labels<19))|(labels==255)
        if not bool(valid.all()): raise ValueError(f'Invalid trainId mask: {row["mask"]}')
        return {'pixel_values':pixels.float(),'labels':labels.long(),'domain':row.get('domain','unknown')}
