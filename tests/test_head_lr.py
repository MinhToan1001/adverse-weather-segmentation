from copy import deepcopy

import pytest
import torch

from src.training.optim import make_optimizer


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=torch.nn.Linear(2,2)
        self.decode_head=torch.nn.Linear(2,2)

    def forward(self,x):
        return self.decode_head(self.encoder(x))


def test_disabled_multiplier_matches_legacy_optimizer_bitwise():
    torch.manual_seed(42)
    reference=Network();candidate=deepcopy(reference)
    old=torch.optim.AdamW(reference.parameters(),lr=6e-5,weight_decay=.01)
    new=make_optimizer(candidate,dict(lr=6e-5,weight_decay=.01))
    x=torch.ones(3,2)
    for _ in range(3):
        for model,opt in ((reference,old),(candidate,new)):
            opt.zero_grad();model(x).square().mean().backward();opt.step()
    assert len(new.param_groups)==1
    assert all(torch.equal(a,b) for a,b in zip(reference.parameters(),candidate.parameters()))


def test_group_membership_lr_schedule_and_resume():
    model=Network();settings=dict(lr=6e-5,head_lr_multiplier=10,weight_decay=.01)
    opt=make_optimizer(model,settings)
    assert [g['name'] for g in opt.param_groups]==['backbone','decode_head']
    assert {id(p) for p in opt.param_groups[1]['params']}=={id(p) for p in model.decode_head.parameters()}
    ids=[id(p) for g in opt.param_groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(model.parameters()))
    schedule=torch.optim.lr_scheduler.LambdaLR(opt,lambda step: max(0,1-step/4))
    opt.zero_grad();model(torch.ones(3,2)).sum().backward();opt.step();schedule.step()
    restored=make_optimizer(model,settings);restored.load_state_dict(opt.state_dict())
    assert [g['lr'] for g in restored.param_groups]==[g['lr'] for g in opt.param_groups]
    for _ in range(4):
        assert opt.param_groups[1]['lr']==pytest.approx(10*opt.param_groups[0]['lr'])
        opt.step();schedule.step()


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf')])
def test_invalid_head_multiplier_rejected(value):
    with pytest.raises(ValueError,match='finite and positive'):
        make_optimizer(Network(),dict(lr=6e-5,head_lr_multiplier=value))


def test_missing_decoder_rejected():
    with pytest.raises(ValueError,match='decode_head'):
        make_optimizer(torch.nn.Linear(2,2),dict(lr=6e-5,head_lr_multiplier=10))


def test_source_recipe_uses_head_multiplier_ten():
    from src.experiments.config import compose_experiment
    cfg=compose_experiment('segformer_b0','source',{'source_train_manifest':'fit.csv','source_monitor_manifest':'monitor.csv'},'outputs/unused')
    assert cfg['train']['head_lr_multiplier']==10 and cfg['train']['epochs']==20
