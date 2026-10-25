import argparse,json
import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from src.training.train import read_cfg,evaluate_model
from src.data.dataset import ManifestSegmentationDataset

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);args=p.parse_args()
    cfg=read_cfg(args.config);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=SegformerForSemanticSegmentation.from_pretrained(args.checkpoint).to(device)
    ds=ManifestSegmentationDataset(cfg['data']['val_manifest'],cfg['data']['image_size'])
    loader=DataLoader(ds,batch_size=cfg['train']['batch_size'],num_workers=0)
    print(json.dumps(evaluate_model(model,loader,device,cfg['model']['num_labels']),indent=2))
if __name__=='__main__':main()
