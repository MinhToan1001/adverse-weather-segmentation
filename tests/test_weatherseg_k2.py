"""K1->K2 gradient, model compatibility and real shared-trainer resume checks."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F
import yaml

from scripts.run_k2 import compose, prepare_weather
from src.data.preparation import write_manifest
from src.models.legacy.segformer_b0 import SegFormerB0
from src.models.proposed.weatherseg_b0 import WeatherSegB0, StatisticsFiLM
from src.models.registry import build_model, load_checkpoint
from src.training.losses import SegmentationLoss


def test_identity_copies_all_stages_and_preserves_global_rng():
    torch.manual_seed(23)
    base=SegFormerB0().eval()
    rng=torch.get_rng_state().clone()
    model=WeatherSegB0.from_b0(base).eval()
    assert torch.equal(rng,torch.get_rng_state())
    pixels=torch.randn(1,3,64,96)
    with torch.inference_mode():
        expected=base(pixels,output_hidden_states=True)
        actual=model(pixels,output_hidden_states=True)
    assert torch.equal(actual.logits,expected.logits)
    assert actual.condition_logits.shape==(1,5)
    assert len(actual.hidden_states)==4
    assert all(torch.equal(a,b) for a,b in zip(actual.hidden_states,expected.hidden_states))
    assert actual.logits.shape==(1,19,16,24)
    with torch.no_grad():
        model.segformer.encoder.film[0].mlp[-1].bias[32]=.5
    with torch.inference_mode(): changed=model(pixels,output_hidden_states=True)
    # The modified first stage must reach the NEXT encoder stage, not just its decoder skip.
    assert not torch.equal(changed.hidden_states[1],actual.hidden_states[1])


def test_film_and_condition_head_learn_with_correct_weighted_loss():
    torch.manual_seed(42)
    model=WeatherSegB0().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    pixels=torch.randn(2,3,64,96)
    labels=torch.randint(0,19,(2,64,96));labels[:,:4]=255
    conditions=torch.tensor([2,4])
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        result=model(pixels,labels=labels,condition_labels=conditions)
        ce=SegmentationLoss()(result.logits,labels)
        auxiliary=F.cross_entropy(result.condition_logits,conditions)
        assert torch.allclose(result.loss,ce+.1*auxiliary)
        result.loss.backward()
        for film in model.segformer.encoder.film:
            grad=film.mlp[-1].weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
            if step:
                assert film.mlp[0].weight.grad.abs().sum()>0
        assert model.condition_head.weight.grad.abs().sum()>0
        optimizer.step()
    ignored=model(pixels,labels=torch.full_like(labels,255),condition_labels=conditions)
    assert ignored.segmentation_loss==0 and torch.isfinite(ignored.loss)
    with pytest.raises(ValueError,match='condition_labels'):
        model(pixels,condition_labels=torch.tensor([5,0]))


def test_film_statistics_are_finite_on_single_pixel_and_checkpoint_is_strict(tmp_path):
    film=StatisticsFiLM(8)
    features=torch.ones(2,8,1,1)
    assert torch.equal(film(features),features)
    model=WeatherSegB0().eval()
    model.save_pretrained(tmp_path)
    restored=load_checkpoint(tmp_path).eval()
    assert isinstance(restored,WeatherSegB0)
    pixels=torch.randn(1,3,64,96)
    with torch.inference_mode():
        expected=model(pixels);actual=restored(pixels)
        assert torch.equal(expected.logits,actual.logits)
        assert torch.equal(expected.condition_logits,actual.condition_logits)
    from src.inference import SegmentationPredictor
    prediction=SegmentationPredictor(tmp_path,device='cpu',model_name='weatherseg_b0').predict(Image.new('RGB',(96,64)))
    assert prediction.train_ids.size==(96,64)
    with pytest.raises(ValueError,match='different model'):
        load_checkpoint(tmp_path,'segformer_b0')
    from safetensors.torch import load_file,save_file
    state={k:v.clone() for k,v in load_file(str(tmp_path/'model.safetensors')).items()}
    state.pop('condition_head.weight')
    save_file(state,str(tmp_path/'model.safetensors'))
    with pytest.raises(ValueError,match='loading mismatch'):
        load_checkpoint(tmp_path,'weatherseg_b0')


def test_registry_initialization_matches_plain_b0_and_new_parameters_use_head_lr():
    from src.training.optim import make_optimizer
    torch.manual_seed(19)
    base=build_model(dict(model=dict(name='segformer_b0',initialization='random')))
    rng=torch.get_rng_state().clone()
    torch.manual_seed(19)
    model=build_model(dict(model=dict(name='weatherseg_b0',initialization='random')))
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in base.state_dict().items())
    optimizer=make_optimizer(model,dict(lr=1e-5,head_lr_multiplier=10))
    head={id(p) for p in optimizer.param_groups[1]['params']}
    assert id(model.condition_head.weight) in head
    assert all(id(p) in head for p in model.segformer.encoder.film.parameters())


def make_source(tmp_path,train_count=4):
    paths={}
    for split,count in [('train',train_count),('val',2)]:
        rows=[]
        for i in range(count):
            stem=f'aachen_{i:06d}_000019'
            image=tmp_path/'leftImg8bit'/split/'aachen'/(stem+'_leftImg8bit.png')
            mask=tmp_path/'gtFine'/split/'aachen'/(stem+'_gtFine_labelTrainIds.png')
            image.parent.mkdir(parents=True,exist_ok=True);mask.parent.mkdir(parents=True,exist_ok=True)
            values=np.zeros((64,96,3),np.uint8);values[:]=[80+i*10,140,180];values[20:50,20:70]=[180,50,40]
            Image.fromarray(values).save(image)
            labels=np.zeros((64,96),np.uint8);labels[20:50,20:70]=13;labels[:2]=255
            Image.fromarray(labels).save(mask)
            rows.append(dict(image=str(image),mask=str(mask),domain='cityscapes',split=split))
        path=tmp_path/(split+'.csv');write_manifest(rows,path)
        paths['source_train_manifest' if split=='train' else 'source_monitor_manifest']=str(path)
    return paths


def test_weather_preflight_auto_full_and_missing_sidecars(tmp_path):
    paths=make_source(tmp_path)
    weather,report=prepare_weather(tmp_path,paths['source_train_manifest'])
    assert weather['conditions']==['clean','night','rain','snow'] and not report['fog_enabled']
    with pytest.raises(ValueError,match='five-condition'):
        prepare_weather(tmp_path,paths['source_train_manifest'],'full')
    distances=tmp_path/'meters';distances.mkdir()
    for image in (tmp_path/'leftImg8bit/train').rglob('*.png'):
        np.save(distances/(image.stem+'.npy'),np.full((64,96),20.,np.float32))
    weather,report=prepare_weather(tmp_path,paths['source_train_manifest'],'full',distance_root=distances)
    assert 'fog' in weather['conditions'] and report['checked_depth_images']==4
    np.save(next(distances.glob('*.npy')),np.zeros((2,3),np.float32))
    with pytest.raises(ValueError,match='aligned'):
        prepare_weather(tmp_path,paths['source_train_manifest'],'full',distance_root=distances)


def test_ablation_configs_share_k1_and_keep_baseline_model_unchanged(tmp_path):
    paths=make_source(tmp_path)
    configs={arm:compose(arm,paths,tmp_path/arm) for arm in ('baseline','film_clean','aug','film','k2')}
    assert 'weather' not in configs['baseline']['data']['augment']
    assert configs['aug']['model']['name']=='segformer_b0'
    assert all(configs[arm]['data']==configs['k2']['data'] for arm in ('aug','film'))
    assert configs['film']['model']['condition_loss_weight']==0 and not configs['film']['condition_training']
    assert configs['k2']['condition_training'] and configs['k2']['model']['name']=='weatherseg_b0'
    assert configs['film_clean']['data'] == configs['baseline']['data']
    assert configs['film_clean']['model']['condition_loss_weight'] == 0
    assert not configs['film_clean']['condition_training']


def test_k2_protocol_blocks_supervised_override_and_target_selection(tmp_path):
    from src.experiments.protocol import require_training_permission
    cfg=compose('k2',make_source(tmp_path),tmp_path/'run')
    bad=deepcopy(cfg);bad['protocol']='supervised_acdc'
    with pytest.raises(ValueError,match='outside DG'):
        require_training_permission(bad,True)
    with pytest.raises(ValueError,match='train: 2975'):
        require_training_permission(cfg)


def test_shared_trainer_k1_k2_resume_matches_uninterrupted(tmp_path):
    """Actual CLI, DataLoader, auxiliary loss, optimizer/RNG and checkpoint integration."""
    from safetensors.torch import load_file
    from src.training.checkpoints import verified_state
    paths=make_source(tmp_path/'source')
    config=compose('k2',paths,tmp_path/'continuous',epochs=2)
    config['experiment_protocol']='k2_tiny_source_qa'  # Deliberately not official-size benchmark evidence.
    config['model']['initialization']='random'
    config['data']['image_size']=[64,96]
    config['data']['augment'].update(min_scale=1.,max_scale=1.)
    config['train'].update(num_workers=0,max_steps=2,amp=False,gradient_accumulation_steps=1)
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,USE_TF='0',USE_TORCH='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def execute(cfg,path,*extra):
        path.write_text(yaml.safe_dump(cfg),encoding='utf-8')
        result=subprocess.run([sys.executable,'-m','src.training.train','--config',str(path),*map(str,extra)],
            cwd=root,env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
    execute(config,tmp_path/'continuous.yaml')
    interrupted=deepcopy(config);interrupted['output_dir']=str(tmp_path/'resumed')
    execute(interrupted,tmp_path/'resumed.yaml','--stop-after-epoch','1')
    execute(interrupted,tmp_path/'resumed.yaml','--resume',tmp_path/'resumed/last')
    continuous=load_file(str(tmp_path/'continuous/last/model.safetensors'))
    resumed=load_file(str(tmp_path/'resumed/last/model.safetensors'))
    assert all(torch.equal(continuous[name],resumed[name]) for name in continuous)
    state=verified_state(tmp_path/'resumed/last')
    assert state['epoch']==2 and state['global_step']==4
    history=[json.loads(line) for line in (tmp_path/'resumed/history.jsonl').read_text().splitlines()]
    assert len(history)==2 and sum(history[-1]['condition_train']['counts'])==4
    assert np.isfinite(history[-1]['condition_train']['cross_entropy'])
    assert (tmp_path/'resumed/best/model.safetensors').is_file()
