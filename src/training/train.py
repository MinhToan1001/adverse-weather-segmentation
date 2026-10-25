import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from src.data.dataset import ManifestSegmentationDataset
from src.models.segformer import make_model
from src.evaluation.metrics import confusion_matrix, scores

def read_cfg(path):
    with open(path,encoding='utf-8') as f:return yaml.safe_load(f)

def evaluate_model(model, loader, device, n):
    model.eval();conf=torch.zeros((n,n),dtype=torch.int64)
    with torch.no_grad():
        for b in loader:
            logits=model(pixel_values=b['pixel_values'].to(device)).logits
            logits=F.interpolate(logits,size=b['labels'].shape[-2:],mode='bilinear',align_corners=False)
            pred=logits.argmax(1).cpu()
            conf+=confusion_matrix(pred,b['labels'],n)
    return scores(conf)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args()
    cfg=read_cfg(args.config);seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    train=ManifestSegmentationDataset(cfg['data']['train_manifest'],cfg['data']['image_size'])
    val=ManifestSegmentationDataset(cfg['data']['val_manifest'],cfg['data']['image_size'])
    trainloader=DataLoader(train,batch_size=cfg['train']['batch_size'],shuffle=True,num_workers=cfg['train']['num_workers'])
    valloader=DataLoader(val,batch_size=cfg['train']['batch_size'],num_workers=cfg['train']['num_workers'])
    model=make_model(cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['train']['lr'])
    best=-1.;out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    for epoch in range(cfg['train']['epochs']):
        model.train();losses=[]
        for step,b in enumerate(trainloader):
            if cfg['train']['max_steps'] is not None and step>=cfg['train']['max_steps']:break
            opt.zero_grad(set_to_none=True)
            result=model(pixel_values=b['pixel_values'].to(device),labels=b['labels'].to(device))
            result.loss.backward();opt.step();losses.append(result.loss.item())
        metrics=evaluate_model(model,valloader,device,cfg['model']['num_labels'])
        record={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'val_miou':metrics['miou'],'seed':seed,'device':device}
        print(json.dumps(record));(out/'history.jsonl').open('a').write(json.dumps(record)+'\n')
        if metrics['miou']>best:
            best=metrics['miou'];model.save_pretrained(out/'best')
    print(f'Best checkpoint: {out / "best"}')
if __name__=='__main__':main()
