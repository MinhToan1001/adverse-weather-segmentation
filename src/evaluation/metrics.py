import torch

def confusion_matrix(pred, target, num_classes=19, ignore_index=255):
    pred=pred.reshape(-1).long();target=target.reshape(-1).long()
    valid=(target!=ignore_index)&(target>=0)&(target<num_classes)
    idx=target[valid]*num_classes+pred[valid].clamp(0,num_classes-1)
    return torch.bincount(idx,minlength=num_classes**2).reshape(num_classes,num_classes)

def scores(conf):
    conf=conf.float();tp=conf.diag();denom=conf.sum(0)+conf.sum(1)-tp
    valid=denom>0
    iou=torch.where(valid,tp/denom.clamp(min=1),torch.nan)
    return {'miou':float(iou[valid].mean()) if bool(valid.any()) else float('nan'),
            'per_class_iou':[None if torch.isnan(x) else float(x) for x in iou]}
