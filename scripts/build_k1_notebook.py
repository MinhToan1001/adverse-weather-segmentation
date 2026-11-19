"""Generate a CPU Kaggle visual-QA notebook for K1 (no training or target data)."""
import ast
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def build():
    setup=(ROOT/'scripts/kaggle_bootstrap.py').read_text(encoding='utf-8')+'''
import os,sys,subprocess
CODE_INPUT=None  # Optional uploaded project-code.zip/folder; None clones the K1 branch
REPO=Path('/kaggle/working/k1_code')
OUT=Path('/kaggle/working/k1_preview')
assert not REPO.exists() and not OUT.exists(), 'Use a fresh session or new output paths'
if CODE_INPUT is None:
    subprocess.run(['git','clone','--depth','1','--single-branch','--branch','feat/k1-weather-augmentation',
                    'https://github.com/MinhToan1001/adverse-weather-segmentation.git',str(REPO)],check=True)
else:
    copy_source(resolve_source(Path(CODE_INPUT),Path('/kaggle/input')),REPO)
assert (REPO/'scripts/preview_weather.py').is_file(),'Need the K1 code branch/bundle'
os.chdir(REPO)
sys.path.insert(0,str(REPO))
def run(*args):
    subprocess.run([sys.executable,'-u',*map(str,args)],cwd=REPO,check=True)
run('-m','pip','install','numpy>=1.26,<3','Pillow>=10,<12')
'''
    demo='''from IPython.display import display
from PIL import Image
run('-m','scripts.preview_weather','--output',OUT/'analytic_demo')
display(Image.open(OUT/'analytic_demo/preview.png'))
print('This first scene has analytically specified synthetic distances; it is software QA only.')
'''
    select='''import numpy as np
INPUT=Path('/kaggle/input')
NUM_IMAGES=3
SEED=42
IMAGE_PATH=None  # Optional single RGB image; otherwise randomly select Cityscapes TRAIN frames
DISTANCE_NPY=None  # Optional single-image camera-ray metric distance; not arbitrary relative depth
DISPARITY_PATH=None  # Optional single-image original Cityscapes disparity PNG
CAMERA_PATH=None  # Matching camera JSON
INVALID_POLICY='far'  # EXPLICIT approximation: invalid stereo -> 200m; fraction recorded in metadata
if IMAGE_PATH is not None:
    images=[Path(IMAGE_PATH)]
else:
    available=sorted(p for p in INPUT.rglob('*_leftImg8bit.png')
                     if p.parent.parent.name=='train' and p.parent.parent.parent.name=='leftImg8bit')
    chosen=np.random.default_rng(SEED).choice(len(available),size=min(NUM_IMAGES,len(available)),replace=False)
    images=[available[int(i)] for i in chosen]
print('Source train images:',len(images))
if not images:
    print('Attach Cityscapes leftImg8bit/train to preview real source images. Analytic demo above is complete.')
'''
    source='''import json
reports=[]
for index,image in enumerate(images):
    stem=image.stem.removesuffix('_leftImg8bit')
    disparity=Path(DISPARITY_PATH) if DISPARITY_PATH is not None else None
    camera=Path(CAMERA_PATH) if CAMERA_PATH is not None else None
    if DISTANCE_NPY is None and disparity is None and camera is None:
        ds=sorted(INPUT.rglob(stem+'_disparity.png'))
        cs=sorted(INPUT.rglob(stem+'_camera.json'))
        assert len(ds)<=1 and len(cs)<=1,'Ambiguous depth inputs; set explicit single-image paths'
        if ds and cs:
            disparity,camera=ds[0],cs[0]
    destination=OUT/f'source_{index:02d}'
    args=['-m','scripts.preview_weather','--image',image,'--output',destination,'--seed',SEED]
    if DISTANCE_NPY is not None:
        assert len(images)==1,'DISTANCE_NPY belongs to one image'
        args+=['--distance',DISTANCE_NPY]
    elif disparity is not None or camera is not None:
        assert disparity is not None and camera is not None,'Need both disparity and camera'
        args+=['--disparity',disparity,'--camera',camera,'--invalid-policy',INVALID_POLICY]
    else:
        args+=['--skip-fog']
        print('No metric distance:',image.name,'-> fog explicitly skipped; night/rain/snow still previewed')
    run(*args)
    display(Image.open(destination/'preview.png'))
    report=json.loads((destination/'metadata.json').read_text())
    reports.append(report)
    print('Distance provenance:',report['distance'])
'''
    archive='''from IPython.display import FileLink
archive=Path(shutil.make_archive(str(OUT),'zip',root_dir=OUT))
display(FileLink(str(archive)))
print('Download previews + metadata for inspection. No model has been trained.')
'''
    contents=[('markdown','# K1 · Weather generation and visual QA\n\n'
        '**CPU is enough.** Enable Internet, optionally attach Cityscapes train images, disparity and camera calibration, then Run All. '
        'The notebook first previews a procedural scene with known synthetic distances, then selects up to3source TRAIN images with a fixed seed. '
        'No ACDC is loaded and no training is started.\n\n'
        'Fog requires metric distance. Missing distance skips fog on real images; no fake depth is invented. '
        'Stereo invalids use the explicitly selected far-fill approximation, recorded per image. '
        'Rain/snow are 2D overlays and night is a linearized-sRGB sensor approximation. '
        'Severity1–5 is an engineering preset, not a calibrated real-weather scale.'),
        ('code',setup),('code',demo),('code',select),('code',source),('code',archive),
        ('markdown','Before training: inspect randomly selected examples at all severities; verify depth/units/invalid fractions; '
         'compare against ordinary augmentation or Foggy Cityscapes under a controlled experiment. '
         'Do not use this same generator as the sole test benchmark. See docs/K1_WEATHER.md for equations, limits and integration.')]
    cells=[]
    for i,(kind,text) in enumerate(contents):
        cell=dict(id=f'k1-{i}',cell_type=kind,metadata={},source=text.splitlines(True))
        if kind=='code':
            ast.parse(text)
            cell.update(outputs=[],execution_count=None)
        cells.append(cell)
    path=ROOT/'notebooks/03_k1_weather_preview.ipynb'
    path.write_text(json.dumps(dict(cells=cells,nbformat=4,nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),
                      language_info=dict(name='python'))),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return path


if __name__=='__main__':
    print(build())
