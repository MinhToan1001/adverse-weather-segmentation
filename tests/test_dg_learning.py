from copy import deepcopy
import random

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from types import SimpleNamespace

from src.data.augmentations.source_style import SourceStyle
from src.data.dataset import ManifestSegmentationDataset
from src.data.preparation import write_manifest
from src.experiments.dg import compose_dg
from src.training.dg import backward_batch, consistency_kl, paired_batch, reference_log_probability, validate_settings
from src.training.losses import SegmentationLoss


def style():
    return dict(brightness=[.6,1.4],contrast=[.6,1.4],saturation=[.5,1.5],gamma=[.6,1.8],
                blur_radius=[0,1],noise_std=[0,.015])


def test_source_style_preserves_shape_and_global_rng():
    im=Image.fromarray(np.arange(16*24*3,dtype=np.uint8).reshape(16,24,3))
    random.seed(13);np.random.seed(13);torch.manual_seed(13)
    py=random.getstate();np_state=np.random.get_state();tr=torch.get_rng_state().clone()
    operation=SourceStyle(style());a=operation.apply(im,42,0,3);b=operation.apply(im,42,0,3)
    assert a.size==im.size and a.mode=='RGB' and np.array_equal(a,b)
    assert not np.array_equal(a,operation.apply(im,42,1,3))
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],np_state[1])
    assert torch.equal(torch.get_rng_state(),tr)
    with pytest.raises(ValueError):SourceStyle(dict(style(),target_image='ACDC'))


def test_paired_dataset_uses_identical_mask_geometry_and_clean_view(tmp_path):
    image=tmp_path/'x.png';mask=tmp_path/'y.png'
    Image.fromarray(np.arange(64*96*3,dtype=np.uint8).reshape(64,96,3)).save(image)
    Image.fromarray(np.tile(np.array([0,13,255],np.uint8),(64,32))).save(mask)
    manifest=tmp_path/'train.csv'
    write_manifest([dict(image=str(image),mask=str(mask),domain='cityscapes',split='train')],manifest)
    common=dict(manifest=manifest,image_size=(64,96),training=True,
                augment=dict(min_scale=.5,max_scale=1.5,horizontal_flip=.5),seed=42)
    base=ManifestSegmentationDataset(**common)
    paired=ManifestSegmentationDataset(**common,paired_style=style())
    random.seed(7);a=base[0]
    random.seed(7);b=paired[0]
    assert torch.equal(a['labels'],b['labels']) and torch.equal(a['pixel_values'],b['pixel_values'])
    assert b['styled_pixel_values'].shape==b['pixel_values'].shape
    assert not torch.equal(b['styled_pixel_values'],b['pixel_values'])
    with pytest.raises(ValueError,match='Cityscapes train only'):
        ManifestSegmentationDataset(manifest,image_size=(64,96),paired_style=style())


def test_consistency_detaches_teacher_excludes_void_and_handles_all_ignore():
    torch.manual_seed(8)
    teacher=torch.randn(2,19,3,4,requires_grad=True)
    student=torch.randn(2,19,3,4,requires_grad=True)
    labels=torch.zeros(2,3,4,dtype=torch.long);labels[:,:,0]=255
    logteacher=teacher.log_softmax(1)
    loss=consistency_kl(student,logteacher,labels)
    changed=student.detach().clone();changed[:,:,:,0]+=torch.randn(2,19,3)*100
    assert torch.allclose(loss,consistency_kl(changed,logteacher,labels))
    loss.backward()
    assert teacher.grad is None and student.grad.abs().sum()>0
    assert torch.equal(student.grad[:,:,:,0],torch.zeros_like(student.grad[:,:,:,0]))
    zero=consistency_kl(student,logteacher,torch.full_like(labels,255))
    assert zero.item()==0 and torch.isfinite(zero)
    equal=consistency_kl(teacher,reference_log_probability(teacher,(3,4)),labels)
    assert abs(float(equal.detach()))<1e-6


class TinyBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Conv2d(3,8,1),nn.BatchNorm2d(8),nn.ReLU(),nn.Dropout(.1),nn.Conv2d(8,19,1))
    def forward(self,pixel_values):return SimpleNamespace(logits=self.network(pixel_values))


def batch():
    x=torch.randn(2,3,8,12)
    y=torch.randint(0,19,(2,8,12));y[:,0,:]=255
    return dict(pixel_values=x,styled_pixel_values=x*.6+.2,labels=y)


@pytest.mark.parametrize('weight',[0.,.1])
def test_sequential_backward_matches_declared_joint_loss_bn_and_accumulation(weight):
    torch.manual_seed(9);a=TinyBN().train();b=deepcopy(a);data=batch();criterion=SegmentationLoss()
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    torch.manual_seed(42)
    report=backward_batch(a,data,criterion,scaler,4,'cpu',paired=True,consistency_weight=weight)
    torch.manual_seed(42)
    clean=b(data['pixel_values']).logits
    teacher=reference_log_probability(clean,data['labels'].shape[-2:])
    styled=b(data['styled_pixel_values']).logits
    reference=.5*(criterion(clean,data['labels'])+criterion(styled,data['labels']))
    reference=reference+weight*consistency_kl(styled,teacher,data['labels'])
    (reference/4).backward()
    assert report['objective']==pytest.approx(float(reference.detach()),rel=1e-6)
    for p,q in zip(a.parameters(),b.parameters()):assert torch.allclose(p.grad,q.grad,atol=2e-7,rtol=1e-5)
    assert a.network[1].num_batches_tracked.item()==2
    assert torch.equal(a.network[1].running_mean,b.network[1].running_mean)


def test_aug_and_zero_weight_consistency_have_identical_gradients_and_pair_schedule():
    torch.manual_seed(17);a=TinyBN();b=deepcopy(a);data=batch();criterion=SegmentationLoss()
    for model in [a,b]:
        torch.manual_seed(11)
        backward_batch(model,data,criterion,torch.amp.GradScaler('cuda',enabled=False),1,'cpu',paired=True)
    assert all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()))
    settings=dict(arm='aug',pair_probability=.5,consistency_weight=0.,source_style=style())
    validate_settings(settings)
    one=[paired_batch(settings,42,0,i) for i in range(20)]
    other=dict(settings,arm='consistency',consistency_weight=.1)
    assert one==[paired_batch(other,42,0,i) for i in range(20)]
    assert any(one) and not all(one)


@pytest.mark.parametrize('model',['segformer_b2','deeplabv3plus_resnet50'])
@pytest.mark.parametrize('arm',['baseline','aug','consistency'])
def test_composed_arm_keeps_shared_source_recipe_and_no_target(model,arm,tmp_path):
    cfg=compose_dg(model,arm,dict(source_train_manifest='city_train',source_monitor_manifest='city_val'),tmp_path/model)
    assert cfg['protocol']=='source_only' and cfg['train']['epochs']==20
    assert cfg['model']['initialization']=='imagenet_encoder' and cfg['loss']=={'name':'ce'}
    assert cfg['dg']['arm']==arm and cfg['train']['batch_size']==2
    assert set(k for k in cfg['data'] if k.endswith('_manifest'))=={'train_manifest','val_manifest'}
    assert cfg['dg']['consistency_weight']==(.1 if arm=='consistency' else 0.)
