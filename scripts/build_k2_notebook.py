"""Generate the independent K2 Kaggle notebook (one selected arm, not the DG suite)."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build():
    setup = (ROOT/'scripts/kaggle_bootstrap.py').read_text(encoding='utf-8') + '''
import os, sys, json, hashlib, subprocess, time
STARTED = time.monotonic()
ARM = 'k2'  # baseline, aug, film (FiLM without auxiliary loss), k2 (FiLM + condition CE)
SEED = 42
EPOCHS = 20
BATCH_SIZE = 2
WEATHER_MODE = 'auto'  # auto: enable fog only with depth; full: require all5; no-fog: explicitly train4
DEPTH_ROOT = None  # Optional root containing disparity/train and camera/train (may be CITYSCAPES_ROOT)
DISTANCE_ROOT = None  # Alternative folder of <original-image-stem>.npy camera-ray meters
INVALID_POLICY = 'far'  # Explicit approximation: missing stereo pixels -> 200m, or use 'error'
CITYSCAPES_ROOT = None  # Folder containing leftImg8bit/{train,val} and gtFine/{train,val}
CODE_INPUT = None  # None clones the branch; alternatively set to your uploaded project-code.zip/folder
RESUME_INPUT = None  # Previous k2_<arm>_seed42.zip or its Kaggle-expanded folder
GIT_REF = 'feat/k1-weather-augmentation'
BASE = Path('/kaggle/working')
WORK = BASE / f'k2_{ARM}_seed{SEED}'
REPO = BASE / f'k2_code_{ARM}_seed{SEED}'
ARCHIVE = BASE / f'k2_{ARM}_seed{SEED}.zip'
assert not WORK.exists() and not REPO.exists(), 'Use a fresh session; preserve previous ZIP first'
if RESUME_INPUT is not None:
    previous = Path(RESUME_INPUT)
    if previous.is_file():
        extract_safe(previous, WORK)
    else:
        found = [previous] if (previous/'k2_plan.json').is_file() else [p.parent for p in previous.rglob('k2_plan.json')]
        assert len(found) == 1, f'Set RESUME_INPUT to one K2 result: {found}'
        shutil.copytree(found[0], WORK)
    plan = json.loads((WORK/'k2_plan.json').read_text())
    assert plan['protocol'] == 'weatherseg_k2_source_v1'
    assert (plan['arm'],plan['seed'],plan['epochs'],plan['batch_size']) == (ARM,SEED,EPOCHS,BATCH_SIZE)
    original = plan['weather_inputs']
    WEATHER_MODE, DEPTH_ROOT, DISTANCE_ROOT, INVALID_POLICY = (
        original['mode'], original['depth_root'], original['distance_root'], original['invalid_policy'])
    copy_source(WORK/'source.zip' if (WORK/'source.zip').is_file() else WORK/'source', REPO)
    assert all(hashlib.sha256((REPO/n).read_bytes()).hexdigest() == h for n,h in plan['source_hashes'].items())
    CITYSCAPES_ROOT = Path(plan['city_root'])
elif CODE_INPUT is not None:
    copy_source(resolve_source(Path(CODE_INPUT), Path('/kaggle/input')), REPO)
else:
    subprocess.run(['git','clone','--depth','1','--single-branch','--branch',GIT_REF,
                    'https://github.com/MinhToan1001/adverse-weather-segmentation.git',str(REPO)],check=True)
assert (REPO/'scripts/run_k2.py').is_file(), 'Need the K2 branch/code bundle'
if CITYSCAPES_ROOT is None:
    found = sorted({p.parent for p in Path('/kaggle/input').rglob('leftImg8bit') if p.is_dir()
                    and all((p.parent/n).is_dir() for n in ['leftImg8bit/train','gtFine/train','leftImg8bit/val','gtFine/val'])})
    assert len(found) == 1, f'Set CITYSCAPES_ROOT explicitly: {found}'
    CITYSCAPES_ROOT = found[0]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
os.environ.update(USE_TF='0',USE_TORCH='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
print('K2 arm:', ARM, 'source:', CITYSCAPES_ROOT, 'code:', REPO)
'''
    install = '''def run(*args):
    subprocess.run([sys.executable,'-u',*map(str,args)],check=True,cwd=REPO)
run('-m','pip','install','transformers==4.57.6','numpy>=1.26,<3','Pillow>=10,<12',
    'PyYAML>=6,<7','matplotlib>=3.8,<4','safetensors>=0.4')
import torch
import torchvision
assert torch.cuda.is_available(), 'Enable a GPU accelerator'
assert tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])) >= (2,6), 'Need Torch>=2.6'
print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))
'''
    smoke = '''# Tiny random-init CUDA/AMP check, not a quality benchmark or pretrained training.
run('-m','scripts.verify_k2','--device','cuda','--output',BASE/'k2_smoke.json')
print((BASE/'k2_smoke.json').read_text())
'''
    train = '''from src.experiments.packaging import export_workspace
from IPython.display import FileLink, display
remaining = 9 - (time.monotonic() - STARTED)/3600
weather_args = ['--weather-mode',WEATHER_MODE,'--invalid-policy',INVALID_POLICY]
if DEPTH_ROOT is not None:
    weather_args += ['--depth-root',str(DEPTH_ROOT)]
if DISTANCE_ROOT is not None:
    weather_args += ['--distance-root',str(DISTANCE_ROOT)]
try:
    if remaining > 0:
        run('-m','scripts.run_k2','--city',CITYSCAPES_ROOT,'--root',WORK,'--archive',ARCHIVE,
            '--arm',ARM,'--seed',SEED,'--epochs',EPOCHS,'--batch-size',BATCH_SIZE,
            '--session-hours',remaining,*weather_args)
    else:
        print('Setup exhausted this session window; keep inputs and continue in a fresh session.')
finally:
    if WORK.exists() and not ARCHIVE.is_file():
        export_workspace(WORK,ARCHIVE)
    if ARCHIVE.is_file():
        print('Download result ZIP:',ARCHIVE)
        display(FileLink(str(ARCHIVE)))
'''
    summary = '''if (WORK/'status.json').is_file():
    print((WORK/'status.json').read_text())
if (WORK/'weather_preflight.json').is_file():
    print('Actual K1 conditions:',(WORK/'weather_preflight.json').read_text())
history = WORK/'run/history.jsonl'
if history.is_file():
    for line in history.read_text().splitlines():
        item = json.loads(line)
        print('epoch',item['epoch'],'Cityscapes selection mIoU:',item['val']['miou'],
              'condition training:',item.get('condition_train'))
print('Checkpoint:',WORK/'run/best')
print('Keep the full ZIP for resume. No ACDC evaluation or quality claim is made here.')
'''
    contents = [
        ('markdown', '# K2 · WeatherSeg-B0 FiLM + condition head\n\n'
         'Enable **GPU + Internet**, attach Cityscapes train/val, then Run All. '
         'Default runs **one K2 experiment, 20 epochs**, not all four ablations. '
         'Use a fresh session for each arm or resume. No ACDC input.\n\n'
         'Uses the K1 generator on this same branch: metric scattering fog, low-light shot/read noise and 2D rain/snow. '
         'WEATHER_MODE=auto trains four conditions if no depth exists; fog then has no positive auxiliary labels. '
         'For all five conditions attach metric distances or disparity+camera, set WEATHER_MODE=full and depth paths if needed. '
         'Auxiliary labels identify these synthetic transforms, not actual weather accuracy. '
         'The ImageNet encoder is downloaded and checksum-verified; no supervised ACDC weights are used. '
         'Full GPU training/runtime and accuracy remain to be measured.'),
        ('code', setup), ('code', install), ('code', smoke), ('code', train), ('code', summary),
        ('markdown', 'For resume: upload the last complete ZIP as a Kaggle input, set RESUME_INPUT, '
         'keep ARM/SEED/EPOCHS/BATCH_SIZE unchanged. The notebook restores the original code snapshot. '
         'Cityscapes must be mounted at the same path. Completed epochs are retained; an interrupted epoch repeats. '
         'The runner exports after every complete epoch and stops before the session budget using a conservative estimate. '
         'A platform hard kill may lose the current epoch; download the latest completed ZIP before ending a session.')]
    cells = []
    for index, (kind, text) in enumerate(contents):
        cell = dict(id=f'k2-{index}', cell_type=kind, metadata={}, source=text.splitlines(True))
        if kind == 'code':
            ast.parse(text)
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    path = ROOT/'notebooks/04_k2_weatherseg_b0_train.ipynb'
    path.write_text(json.dumps(dict(cells=cells, nbformat=4, nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),
                      language_info=dict(name='python'),accelerator='GPU')), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return path


if __name__ == '__main__':
    print(build())
