from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from safetensors.torch import save_file,load_file

from src.data.preparation import write_manifest
from src.data.augmentations.weather_v2 import WeatherStyle
from src.experiments.dg_v2 import (split_cities,fit_statistics,compose,validate,selection,
    ROOT,ARMS,GROUPS,portable_rows,verify_reference_split,verify_reference_weights,source_fingerprints,validate_source_pairs)


@pytest.fixture
def source(tmp_path):
    rows=[]
    for city in ['a','b','c','d']:
        for i in range(2):
            image=tmp_path/'leftImg8bit/train'/city/f'{city}_{i}_leftImg8bit.png'
            mask=tmp_path/'gtFine/train'/city/f'{city}_{i}_gtFine_labelTrainIds.png'
            image.parent.mkdir(parents=True,exist_ok=True);mask.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(32,24),(120,80,50)).save(image)
            labels=np.arange(32*24,dtype=np.uint16).reshape(24,32)%19
            labels=labels.astype(np.uint8);labels[0]=255;Image.fromarray(labels).save(mask)
            rows.append(dict(image=str(image),mask=str(mask),domain='cityscapes',split='train'))
    fit,dev=split_cities(rows,seed=42)
    f=tmp_path/'fit.csv';d=tmp_path/'dev.csv';w=tmp_path/'weights.json'
    write_manifest(fit,f);write_manifest(dev,d);w.write_text(json.dumps(fit_statistics(f)))
    return rows,f,d,w


def test_city_split_is_deterministic_disjoint_complete_and_rejects_target(source):
    rows,*_=source
    a,b=split_cities(rows);assert (a,b)==split_cities(rows)
    assert len(a)+len(b)==len(rows)
    assert not {Path(r['image']).parent.name for r in a}&{Path(r['image']).parent.name for r in b}
    with pytest.raises(ValueError):split_cities([dict(r,domain='night') for r in rows])


def test_class_weights_are_fit_bound_capped_and_ignore_void(source):
    rows,f,d,w=source;stats=fit_statistics(f)
    assert len(stats['weights'])==19 and all(.75<=v<=2 for v in stats['weights'])
    assert sum(stats['pixel_counts'])==6*32*23
    cfg=compose('segformer_b2','class_weights',f,d,w,f.parent/'run')
    cfg['loss']['class_weights_file']=str(f.parent/'wrong.json')
    wrong=deepcopy(stats);wrong['fit_manifest_sha256']=['changed'];Path(cfg['loss']['class_weights_file']).write_text(json.dumps(wrong))
    with pytest.raises(ValueError,match='bound'):validate(cfg,qa=True)


@pytest.mark.parametrize('arm',['baseline','class_weights','weather_aug','weather_aug_class_weights'])
def test_matrix_composition_and_no_production_qa_escape(source,arm):
    _,f,d,w=source;cfg=compose('segformer_b2',arm,f,d,w,f.parent/arm)
    validate(cfg,qa=True)
    with pytest.raises(ValueError,match='Production'):validate(cfg)
    assert (cfg['loss']['name']=='weighted_ce')==(arm in ['class_weights','weather_aug_class_weights'])
    assert cfg['v2']['expanded_style']==(arm in ['weather_aug','weather_aug_class_weights'])
    changed=deepcopy(cfg);changed['v2']['pair_probability']=.9
    with pytest.raises(ValueError,match='matrix'):validate(changed,qa=True)


def test_dev_overlap_or_target_cannot_reach_v2_trainer(source):
    rows,f,d,w=source;cfg=compose('segformer_b2','baseline',f,d,w,f.parent/'run')
    cfg['data']['val_manifest']=str(f)
    with pytest.raises(ValueError,match='overlap'):validate(cfg,qa=True)
    write_manifest([dict(rows[0],domain='night')],d)
    cfg['data']['val_manifest']=str(d)
    with pytest.raises(ValueError):validate(cfg,qa=True)


def test_dev_score_is_equal_weight_macro_corruptions():
    assert selection(.8,dict(low_light=.4,noise=.6,haze=.2))==pytest.approx(.6)
    with pytest.raises(ValueError):selection(.8,dict(low_light=.4))


def test_shared_handoff_has_exact_portable_split_and_fit_only_weights():
    from src.data.validation import read_manifest
    from src.utils.artifacts import sha256_file
    fit=read_manifest(ROOT/'configs/dg_v2/splits/fit.csv')
    dev=read_manifest(ROOT/'configs/dg_v2/splits/dev.csv')
    assert len(fit)==2380 and len(dev)==595
    assert set(GROUPS['baseline_weights']).isdisjoint(GROUPS['weather_weights'])
    assert GROUPS['baseline_weights']+GROUPS['weather_weights']==ARMS
    for rows in [fit,dev]:
        assert all(not Path(r['image']).is_absolute() and not Path(r['mask']).is_absolute() for r in rows)
    relocated=lambda rows:[dict(r,image=str(Path('/another/mount')/r['image']),mask=str(Path('/converted')/r['mask'])) for r in rows]
    verify_reference_split(relocated(fit),relocated(dev))
    with pytest.raises(ValueError,match='shared portable'):verify_reference_split(dev,fit)
    stats=json.loads((ROOT/'configs/dg_v2/class_weights_reference.json').read_text())
    assert stats['portable_fit_manifest_sha256']==sha256_file(ROOT/'configs/dg_v2/splits/fit.csv')
    assert stats['target_data_access']=='none'
    verify_reference_weights(stats)
    stats['pixel_counts'][0]+=1
    with pytest.raises(ValueError,match='statistics'):verify_reference_weights(stats)
    hashes=source_fingerprints()
    assert all('configs/dg_v2/splits/'+name+'.csv' in hashes for name in ['fit','dev'])


def test_source_preflight_checks_geometry_and_dev_ids_without_decoding_rgb(source,monkeypatch):
    from src.data.validation import read_manifest
    _,f,d,_=source;fit=read_manifest(f);dev=read_manifest(d)
    # Decompressing every source RGB image is not required to check mask/data contracts.
    monkeypatch.setattr(Image.Image,'convert',lambda *a,**k:pytest.fail('Unexpected RGB conversion'))
    validate_source_pairs(fit,dev)
    Image.new('L',(32,24),19).save(dev[0]['mask'])
    with pytest.raises(ValueError,match='dev trainIds'):validate_source_pairs(fit,dev)
    Image.new('L',(8,8),0).save(dev[0]['mask'])
    with pytest.raises(ValueError,match='geometry'):validate_source_pairs(fit,dev)


def test_weather_is_reproducible_rgb_only_and_dev_is_epoch_invariant():
    import yaml
    root=Path(__file__).resolve().parents[1]
    original=yaml.safe_load((root/'configs/dg/paired_source.yaml').read_text())['source_style']
    image=Image.new('RGB',(32,24),(120,80,50));before=np.asarray(image).copy()
    for kind in ['low_light','noise','haze']:
        style=WeatherStyle(original,corruption=kind)
        a=style.apply(image,42,0,1);b=style.apply(image,42,12,1)
        assert a.size==image.size and a.mode=='RGB' and np.array_equal(a,b)
        assert not np.array_equal(a,before)
        if kind=='low_light':assert np.asarray(a).mean()>8, 'Do not turn a midtone scene almost black'
    assert np.array_equal(np.asarray(image),before)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Conv2d(3,19,3,stride=4,padding=1);self.drop=torch.nn.Dropout2d(.2)
    def forward(self,pixel_values):
        from types import SimpleNamespace
        return SimpleNamespace(logits=self.layer(self.drop(pixel_values)))
    def save_pretrained(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=True)
        (path/'config.json').write_text('{}')
        save_file({k:v.detach().cpu().contiguous() for k,v in self.state_dict().items()},str(path/'model.safetensors'))


def test_actual_epoch_trainer_resume_matches_continuous_with_weighted_paired_views(source,tmp_path,monkeypatch):
    import src.training.dg_v2 as trainer
    torch.set_num_threads(2)
    monkeypatch.setattr(trainer,'build_model',lambda cfg:TinyModel())
    def load(path,name):
        model=TinyModel();model.load_state_dict(load_file(str(path/'model.safetensors')));return model
    monkeypatch.setattr(trainer,'load_checkpoint',load)
    _,f,d,w=source
    def config(out):
        c=compose('segformer_b2','weather_aug_class_weights',f,d,w,out);c['_qa']=True;c['model']['initialization']='random'
        c['train'].update(epochs=2,batch_size=2,num_workers=0,max_steps=2,gradient_accumulation_steps=2,amp=False,head_lr_multiplier=1.)
        c['data']['image_size']=[64,96];return c
    continuous=config(tmp_path/'continuous');resumed=config(tmp_path/'resumed')
    trainer.train_epoch_run(continuous,qa=True)
    trainer.train_epoch_run(resumed,stop_after=1,qa=True)
    trainer.train_epoch_run(resumed,qa=True)
    left=load_file(str(tmp_path/'continuous/last/model.safetensors'))
    right=load_file(str(tmp_path/'resumed/last/model.safetensors'))
    assert all(torch.equal(v,right[k]) for k,v in left.items())
    a=[json.loads(r) for r in (tmp_path/'continuous/history.jsonl').read_text().splitlines()]
    b=[json.loads(r) for r in (tmp_path/'resumed/history.jsonl').read_text().splitlines()]
    assert a==b and len(a)==2 and all(set(r['dev'])=={'clean','low_light','noise','haze'} for r in a)
    changed=deepcopy(resumed);changed['train']['epochs']=3
    with pytest.raises(ValueError,match='resume config'):trainer.train_epoch_run(changed,qa=True)
