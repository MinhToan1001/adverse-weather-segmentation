import pytest
import torch

from src.models.registry import build_model, load_checkpoint, parameter_report, MODEL_NAMES
from src.training.losses import SegmentationLoss
from src.training.strategies import set_training_mode
from src.training.optim import make_optimizer
from src.experiments.config import compose_experiment


@pytest.mark.parametrize('name', MODEL_NAMES)
def test_active_architectures_share_logits_loss_and_gradient_contract(name):
    torch.manual_seed(9)
    model = build_model(dict(model=dict(name=name, initialization='random')))
    set_training_mode(model, True)
    pixels = torch.randn(1, 3, 64, 96)
    labels = torch.randint(0, 19, (1, 64, 96)); labels[:, :4] = 255
    result = model(pixel_values=pixels)
    assert result.logits.shape == (1, 19, 16, 24)
    loss = SegmentationLoss('ce_dice')(result.logits, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)


def test_deeplab_is_matched_size_and_checkpoint_roundtrip_is_exact(tmp_path):
    model = build_model(dict(model=dict(name='deeplabv3plus_resnet50'))).eval()
    count = parameter_report(model)['total']
    assert count == 26695619 and abs(count / 27361235 - 1) < .03
    pixels = torch.randn(1, 3, 64, 96)
    with torch.inference_mode(): expected = model(pixels).logits
    model.save_pretrained(tmp_path)
    restored = load_checkpoint(tmp_path, 'deeplabv3plus_resnet50').eval()
    with torch.inference_mode(): actual = restored(pixels).logits
    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match='different model'):
        load_checkpoint(tmp_path, 'segformer_b2')
    optimizer = make_optimizer(model, dict(lr=1e-5, head_lr_multiplier=10))
    head = {id(p) for p in optimizer.param_groups[1]['params']}
    assert id(model.aspp.fuse[0][0].weight) in head
    assert id(model.low_projection[0].weight) in head


def test_source_recipe_only_changes_model_identity_and_uses_real_batch_two(tmp_path):
    data = dict(source_train_manifest='fit.csv', source_monitor_manifest='monitor.csv')
    configs = [compose_experiment(name, 'source', data, tmp_path/'run') for name in MODEL_NAMES]
    assert all(c['train']['epochs'] == 20 and c['train']['batch_size'] == 2 and c['train']['drop_last'] for c in configs)
    assert all(c['train']['head_lr_multiplier'] == 10 for c in configs)
    assert all(c['train'] == configs[0]['train'] and c['data'] == configs[0]['data'] for c in configs)
    assert all('early_stopping' not in c and c['loss'] == dict(name='ce') for c in configs)


def test_complete_checkpoint_loader_refuses_missing_parameters(tmp_path):
    from safetensors.torch import load_file,save_file
    model=build_model(dict(model=dict(name='segformer_b0',initialization='random')))
    model.save_pretrained(tmp_path)
    state={key:value.clone() for key,value in load_file(str(tmp_path/'model.safetensors')).items()}
    state.pop(next(iter(state)))
    save_file(state,str(tmp_path/'model.safetensors'))
    with pytest.raises(ValueError,match='loading mismatch'):
        load_checkpoint(tmp_path,'segformer_b0')
