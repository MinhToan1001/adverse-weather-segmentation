"""Separate model files must preserve architecture, loss, gradients and loading behavior."""
import pytest
import torch
from transformers import SegformerForSemanticSegmentation

from src.models.segformer_b0 import SegFormerB0, make_config as b0_config
from src.models.segformer_b2 import SegFormerB2, make_config as b2_config
from src.models.segformer import make_model, model_class_for_config, load_segformer_checkpoint


@pytest.mark.parametrize('cls,config_fn,channels,decoder,parameters',[
    (SegFormerB0,b0_config,[32,64,160,256],256,3719027),
    (SegFormerB2,b2_config,[64,128,320,512],768,27361235),
])
@pytest.mark.parametrize('training',[False,True])
def test_separate_variants_keep_official_forward_loss_and_gradients(cls,config_fn,channels,decoder,parameters,training):
    torch.manual_seed(42);model=cls(config_fn()).eval()
    torch.manual_seed(42);reference=SegformerForSemanticSegmentation(config_fn()).eval()
    if training:
        from src.training.strategies import set_training_mode
        set_training_mode(model,True);set_training_mode(reference,True)
    assert sum(p.numel() for p in model.parameters())==parameters
    assert model.decode_head.linear_fuse.in_channels==4*decoder
    assert model.decode_head.classifier.in_channels==decoder
    assert model.state_dict().keys()==reference.state_dict().keys()
    assert all(torch.equal(v,reference.state_dict()[k]) for k,v in model.state_dict().items())
    pixels=torch.randn(1,3,64,96);labels=torch.randint(0,19,(1,64,96));labels[:,:13,:]=255
    torch.manual_seed(914);result=model(pixels,labels=labels,output_hidden_states=True)
    torch.manual_seed(914);expected=reference(pixels,labels=labels,output_hidden_states=True)
    assert result.logits.shape==(1,19,16,24) and torch.equal(result.logits,expected.logits)
    assert [t.shape[1] for t in result.hidden_states]==channels
    assert [tuple(t.shape[-2:]) for t in result.hidden_states]==[(16,24),(8,12),(4,6),(2,3)]
    assert torch.equal(result.loss,expected.loss) and torch.isfinite(result.loss)
    result.loss.backward();expected.loss.backward()
    gradients=dict(reference.named_parameters())
    for name,param in model.named_parameters():
        assert param.grad is not None and torch.equal(param.grad,gradients[name].grad),name
    with torch.inference_mode():
        torch.manual_seed(835);actual_tuple=model(pixels,return_dict=False,output_hidden_states=True)
        torch.manual_seed(835);expected_tuple=reference(pixels,return_dict=False,output_hidden_states=True)
    assert torch.equal(actual_tuple[0],expected_tuple[0]) and len(actual_tuple[1])==4


@pytest.mark.parametrize('cls,wrong_config',[(SegFormerB0,b2_config),(SegFormerB2,b0_config)])
def test_explicit_variant_rejects_other_variant_checkpoint_config(cls,wrong_config):
    with pytest.raises(ValueError,match='configuration'):cls(wrong_config())


def test_factory_dispatches_real_variants_and_keeps_custom_smoke_size():
    assert isinstance(make_model({'model':{'num_labels':19}}),SegFormerB0)
    assert isinstance(make_model({'model':{'num_labels':19,'variant':'b2'}}),SegFormerB2)
    custom=make_model({'model':{'num_labels':19,'hidden_sizes':[8,16,32,64],'depths':[1]*4,'decoder_hidden_size':32}})
    assert type(custom) is SegformerForSemanticSegmentation
    assert model_class_for_config(b0_config()) is SegFormerB0
    assert model_class_for_config(b2_config()) is SegFormerB2
    with pytest.raises(ValueError,match='requested b2'):model_class_for_config(b0_config(),variant='b2')


def test_checkpoint_round_trip_keeps_model_type_and_exact_weights(tmp_path):
    model=SegFormerB0().eval();model.save_pretrained(tmp_path)
    loaded,info=load_segformer_checkpoint(tmp_path,local_files_only=True,output_loading_info=True)
    assert type(loaded) is SegFormerB0
    assert not info['missing_keys'] and not info['unexpected_keys'] and not info.get('mismatched_keys')
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
