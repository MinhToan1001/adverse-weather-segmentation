import ast
import json
import os
import random
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.build_k1_notebook import build
from scripts.preview_weather import preview, analytic_scene
from src.data.augmentations.weather import WeatherGenerator, CONDITIONS, cityscapes_distance, weather_seed
from src.data.augmentations.weather.color import srgb_to_linear, linear_to_srgb
from src.data.augmentations.weather.effects import atmospheric_fog
from src.data.augmentations.weather.policy import SourceWeather
from src.data.dataset import ManifestSegmentationDataset
from src.data.preparation import write_manifest


def test_fog_obeys_linear_light_scattering_and_zero_beta_is_identity():
    image=Image.new('RGB',(3,2),(80,120,160))
    distance=np.array([[0,10,50],[0,10,50]],np.float32)
    actual=np.asarray(atmospheric_fog(image,distance,.02))
    transmission=np.exp(-.02*distance)[...,None]
    expected=linear_to_srgb(srgb_to_linear(np.asarray(image)/255)*transmission+.9*(1-transmission))
    np.testing.assert_allclose(actual,np.rint(expected*255),atol=1)
    assert np.array_equal(np.asarray(atmospheric_fog(image,distance,0)),image)
    assert actual[0,0].mean()<actual[0,1].mean()<actual[0,2].mean()
    assert np.asarray(atmospheric_fog(image,distance,.04)).mean()>actual.mean()


@pytest.mark.parametrize('distance',[None,np.ones((2,2)),np.full((2,3),np.nan),-np.ones((2,3))])
def test_fog_rejects_missing_bad_or_misaligned_metric_distance(distance):
    with pytest.raises(ValueError):
        WeatherGenerator().apply(Image.new('RGB',(3,2)), 'fog', distance_m=distance)


def calibration():
    return dict(intrinsic=dict(fx=100.,fy=100.,u0=0.,v0=0.),extrinsic=dict(baseline=.2))


def test_cityscapes_disparity_offset_scale_and_ray_distance():
    # stored=256*2+1 means disparity2px -> Z=10m, not a normalized relative depth.
    stored=np.full((2,3),513,np.uint16)
    result=cityscapes_distance(stored,calibration())
    assert result.meters[0,0]==10.
    assert result.meters[0,2]>result.meters[0,0]
    assert result.metadata['invalid_fraction']==0
    np.testing.assert_array_equal(stored,np.full((2,3),513,np.uint16))


def test_invalid_stereo_policy_is_explicit_and_recorded():
    stored=np.array([[0,1,513]],np.uint16)
    with pytest.raises(ValueError,match='Missing'):
        cityscapes_distance(stored,calibration())
    result=cityscapes_distance(stored,calibration(),invalid_policy='far',max_distance_m=150.)
    np.testing.assert_array_equal(result.meters[0,:2],[150.,150.])
    assert result.metadata['invalid_fraction']==pytest.approx(2/3)
    with pytest.raises(ValueError,match='16-bit'):
        cityscapes_distance(np.ones((2,3),np.uint8),calibration())


@pytest.mark.parametrize('condition',CONDITIONS)
def test_effects_preserve_geometry_input_and_global_rng(condition):
    image,distance=analytic_scene((256,128))
    before=np.asarray(image).copy()
    random.seed(17);np.random.seed(17);torch.manual_seed(17)
    python=random.getstate();numpy=np.random.get_state();tensor=torch.get_rng_state().clone()
    sample=WeatherGenerator().apply(image,condition,3,seed=123,distance_m=distance)
    again=WeatherGenerator().apply(image,condition,3,seed=123,distance_m=distance)
    assert sample.image.size==image.size and sample.image.mode=='RGB'
    assert sample.condition_id==CONDITIONS.index(condition)
    assert np.array_equal(sample.image,again.image)
    np.testing.assert_array_equal(image,before)
    assert random.getstate()==python
    assert np.array_equal(np.random.get_state()[1],numpy[1])
    assert torch.equal(torch.get_rng_state(),tensor)
    json.dumps(sample.metadata,allow_nan=False)


def test_night_darkens_and_strength_increases_in_expectation():
    image=Image.new('RGB',(256,128),(160,160,160))
    generator=WeatherGenerator()
    mild=np.asarray(generator.apply(image,'night',1,seed=9).image).mean()
    strong=np.asarray(generator.apply(image,'night',5,seed=9).image).mean()
    assert 160>mild>strong


@pytest.mark.parametrize('severity',[0,6,True,1.5])
def test_invalid_severity_fails(severity):
    with pytest.raises(ValueError,match='severity'):
        WeatherGenerator().apply(Image.new('RGB',(4,4)), 'rain', severity)


def test_policy_requires_depth_and_rejects_target_before_pixels_are_changed():
    with pytest.raises(ValueError,match='Fog policy'):
        SourceWeather(dict(conditions=['fog']))
    with pytest.raises(ValueError,match='distance_unit'):
        SourceWeather(dict(conditions=['fog'],distance_root='depth'))
    policy=SourceWeather(dict(conditions=['night']))
    with pytest.raises(ValueError,match='source train'):
        policy.apply(Image.new('RGB',(8,8)),dict(domain='night',split='train'),42,0,0)


@pytest.mark.parametrize('condition',CONDITIONS)
def test_source_dataset_weather_keeps_labels_geometry_and_emits_condition_ids(tmp_path,condition):
    image=tmp_path/'frame.png';mask=tmp_path/'labels.png';manifest=tmp_path/'train.csv'
    Image.fromarray(np.arange(64*96*3,dtype=np.uint8).reshape(64,96,3)).save(image)
    Image.fromarray(np.tile(np.array([0,13,255],np.uint8),(64,32))).save(mask)
    write_manifest([dict(image=str(image),mask=str(mask),domain='synthetic',split='train')],manifest)
    np.save(tmp_path/'frame.npy',np.full((64,96),35.,np.float32))
    common=dict(manifest=manifest,image_size=(64,96),training=True,seed=42)
    geometry=dict(min_scale=.7,max_scale=1.5,horizontal_flip=.5)
    baseline=ManifestSegmentationDataset(**common,augment=geometry)
    policy=dict(conditions=[condition],severities=[3],distance_root=str(tmp_path),distance_unit='meters')
    augmented=ManifestSegmentationDataset(**common,augment=dict(geometry,weather=policy))
    random.seed(8);expected=baseline[0]
    random.seed(8);actual=augmented[0]
    assert torch.equal(actual['labels'],expected['labels'])
    assert actual['pixel_values'].shape==expected['pixel_values'].shape
    assert actual['condition_labels'].item()==CONDITIONS.index(condition)
    assert actual['weather_severity'].item()==3
    if condition=='clean':
        assert torch.equal(actual['pixel_values'],expected['pixel_values'])
    assert set(torch.unique(actual['labels']).tolist())<={0,13,255}
    with pytest.raises(ValueError,match='source training'):
        ManifestSegmentationDataset(manifest,image_size=(64,96),augment=dict(weather=policy))


def test_preview_exports_full_grid_provenance_and_does_not_change_mask(tmp_path):
    image,distance=analytic_scene((96,64))
    image.save(tmp_path/'source.png')
    mask=tmp_path/'labels.png';Image.new('L',image.size,13).save(mask)
    original=mask.read_bytes()
    np.save(tmp_path/'distance.npy',distance)
    report=preview(tmp_path/'preview',image=tmp_path/'source.png',distance=tmp_path/'distance.npy',mask=mask)
    assert len(report['samples'])==20 and report['mask_unchanged']
    assert mask.read_bytes()==original and (tmp_path/'preview/preview.png').is_file()
    assert json.loads((tmp_path/'preview/metadata.json').read_text())['distance']['units']=='meters'
    with pytest.raises(ValueError,match='Preserve existing'):
        preview(tmp_path/'preview')
    with pytest.raises(ValueError,match='requires metric distance'):
        preview(tmp_path/'no_depth',image=tmp_path/'source.png')
    report=preview(tmp_path/'skip_fog',image=tmp_path/'source.png',skip_fog=True)
    assert len(report['samples'])==15 and report['skipped_conditions']==['fog']


def test_k1_notebook_builds_executable_empty_cells_and_uses_no_training():
    notebook=json.loads(build().read_text(encoding='utf-8'))
    code=''
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            text=''.join(cell['source']);ast.parse(text);code+=text
            assert cell['outputs']==[] and cell['execution_count'] is None
    assert 'scripts.preview_weather' in code and 'src.training.train' not in code
    assert '--skip-fog' in code and 'feat/k1-weather-augmentation' in code
    assert weather_seed(42,0,1)==weather_seed(42,0,1)!=weather_seed(42,1,1)


def test_actual_kaggle_cells_bootstrap_upload_and_preview_source_without_fake_depth(tmp_path,monkeypatch):
    from scripts.build_kaggle_notebooks import build as package
    bundle=package()[1]
    inputs=tmp_path/'input';working=tmp_path/'working';working.mkdir()
    city=inputs/'city/leftImg8bit/train/aachen';city.mkdir(parents=True)
    Image.new('RGB',(96,64),(120,150,180)).save(city/'aachen_000000_000019_leftImg8bit.png')
    notebook=json.loads(build().read_text(encoding='utf-8'))
    setup=''.join(notebook['cells'][1]['source'])
    setup=setup.replace('CODE_INPUT=None',f'CODE_INPUT=Path({str(bundle)!r})')
    for old,new in [('/kaggle/working/k1_code',working/'code'),('/kaggle/working/k1_preview',working/'preview'),('/kaggle/input',inputs)]:
        setup=setup.replace(repr(old),repr(str(new)))
    invoked=[]
    original_run=subprocess.run
    monkeypatch.setattr(subprocess,'run',lambda command,**kwargs:invoked.append(command))
    previous=os.getcwd()
    old_paths=list(sys.path)
    try:
        namespace={}
        exec(compile(setup,'<k1-setup>','exec'),namespace)
        assert invoked and 'pip' in invoked[0]
        assert (namespace['REPO']/'scripts/preview_weather.py').is_file()
        select=''.join(notebook['cells'][3]['source']).replace(repr('/kaggle/input'),repr(str(inputs)))
        exec(compile(select,'<k1-select>','exec'),namespace)
        assert len(namespace['images'])==1
        def run(*args):
            original_run([sys.executable,*map(str,args)],cwd=namespace['REPO'],check=True,capture_output=True)
        namespace.update(run=run,display=lambda value:None,Image=Image)
        exec(compile(''.join(notebook['cells'][4]['source']),'<k1-source>','exec'),namespace)
        report=json.loads((namespace['OUT']/'source_00/metadata.json').read_text())
        assert report['skipped_conditions']==['fog'] and report['distance'] is None
        assert len(report['samples'])==15
    finally:
        os.chdir(previous);sys.path[:]=old_paths
