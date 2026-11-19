"""Preview K1 effects/severity, record parameters and optionally check an unchanged mask."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.data.augmentations.weather import WeatherGenerator, CONDITIONS, cityscapes_distance, weather_seed


def analytic_scene(size=(640,320)):
    """Procedural QA fixture with KNOWN synthetic distances; never a real depth estimate."""
    width,height = size
    image = Image.new('RGB',size,(115,175,225))
    draw = ImageDraw.Draw(image)
    draw.polygon([(0,height),(width,height),(width*.58,height*.42),(width*.42,height*.42)],fill=(85,85,90))
    draw.rectangle((20,50,180,225),fill=(180,135,100))
    draw.rectangle((450,85,620,235),fill=(100,145,100))
    draw.rectangle((250,220,370,280),fill=(185,45,35))
    draw.rectangle((275,225,345,245),fill=(100,150,185))
    draw.rectangle((255,270,275,290),fill=(20,20,20))
    draw.rectangle((345,270,365,290),fill=(20,20,20))
    for y in range(170,height,35):
        draw.rectangle((width//2-3,y,width//2+3,y+15),fill=(240,230,180))
    depth = np.broadcast_to(np.linspace(140,5,height,dtype=np.float32)[:,None],(height,width)).copy()
    depth[:int(height*.42)] = 200.
    depth[50:226,20:181] = 35.
    depth[85:236,450:621] = 50.
    depth[220:291,250:371] = 12.
    return image,depth


def preview(output, image=None, distance=None, disparity=None, camera=None,
            mask=None, seed=42, skip_fog=False, invalid_policy='error'):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserve existing previews; choose an empty output directory')
    if (disparity is None) != (camera is None) or (distance is not None and disparity is not None):
        raise ValueError('Supply distance NPY OR both disparity and camera calibration')
    if image is None:
        if distance is not None or disparity is not None or mask is not None:
            raise ValueError('Explicit depth/mask inputs require --image')
        rgb, meters = analytic_scene()
        depth_info = dict(source='analytic procedural QA fixture', units='meters', known_synthetic_distance=True)
    else:
        with Image.open(image) as source:
            rgb = source.convert('RGB')
        meters, depth_info = None, None
        if distance is not None:
            meters = np.load(distance, allow_pickle=False)
            depth_info = dict(source=str(distance), units='meters', kind='user_declared_camera_ray_distance')
        elif disparity is not None:
            loaded = cityscapes_distance(disparity,camera,invalid_policy=invalid_policy)
            meters,depth_info = loaded.meters,loaded.metadata
        if meters is None and not skip_fog:
            raise ValueError('Real-image fog requires metric distance, or explicitly use --skip-fog')
    if meters is not None and np.asarray(meters).shape != (rgb.height,rgb.width):
        raise ValueError('Distance and RGB must have identical native geometry')
    mask_digest = None
    if mask is not None:
        mask_digest = hashlib.sha256(Path(mask).read_bytes()).hexdigest()
        with Image.open(mask) as labels:
            if labels.size != rgb.size:
                raise ValueError('Mask and RGB must be aligned')
    output.mkdir(parents=True,exist_ok=True)
    rgb.save(output/'clean.png')
    conditions = [c for c in CONDITIONS if c != 'clean' and not (c=='fog' and skip_fog)]
    tile_w = 256
    tile_h = max(1,round(rgb.height*tile_w/rgb.width))
    panel = Image.new('RGB',(tile_w*6,(tile_h+24)*len(conditions)),(245,245,245))
    draw = ImageDraw.Draw(panel)
    generator = WeatherGenerator()
    records=[]
    for row,condition in enumerate(conditions):
        top = row*(tile_h+24)
        panel.paste(rgb.resize((tile_w,tile_h)),(0,top+24))
        draw.text((6,top+5),f'{condition}: clean reference',fill=(10,10,10))
        for severity in range(1,6):
            sample = generator.apply(rgb,condition,severity,seed=weather_seed(seed,index=condition),distance_m=meters)
            path=output/f'{condition}_{severity}.png'
            sample.image.save(path)
            panel.paste(sample.image.resize((tile_w,tile_h)),(severity*tile_w,top+24))
            draw.text((severity*tile_w+6,top+5),f'{condition} severity {severity}',fill=(10,10,10))
            records.append(dict(sample.metadata,file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                mean_srgb=float(np.asarray(sample.image,dtype=np.float32).mean()/255.)))
    panel.save(output/'preview.png')
    report=dict(scope='K1 visual/software QA; not real-weather validation or segmentation improvement',
        source_image=str(image) if image else 'analytic QA fixture',source_size=list(rgb.size),
        distance=depth_info,seed=seed,skipped_conditions=['fog'] if skip_fog else [],
        mask_sha256=mask_digest,mask_unchanged=None if mask is None else mask_digest==hashlib.sha256(Path(mask).read_bytes()).hexdigest(),
        samples=records)
    (output/'metadata.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--image')
    parser.add_argument('--distance',help='Aligned camera-ray distance NPY, meters (not relative depth or disparity)')
    parser.add_argument('--disparity')
    parser.add_argument('--camera')
    parser.add_argument('--mask')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--skip-fog',action='store_true')
    parser.add_argument('--invalid-policy',choices=['error','far'],default='error')
    a=parser.parse_args()
    report=preview(a.output,a.image,a.distance,a.disparity,a.camera,a.mask,a.seed,a.skip_fog,a.invalid_policy)
    print('Created',len(report['samples']),'weather samples and preview.png in',a.output)
