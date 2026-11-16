"""Generate three thin Kaggle launchers from one shared template."""
import ast
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS={
    'segformer_b0':'01_train_segformer_b0.ipynb',
    'segformer_b2':'02_train_segformer_b2.ipynb',
    'deeplabv3plus_resnet50':'03_train_deeplabv3plus_resnet50.ipynb',
}


def cell(kind,text):
    if kind=='code':ast.parse(text)
    result=dict(cell_type=kind,metadata={},source=text.splitlines(True))
    if kind=='code':result.update(outputs=[],execution_count=None)
    return result


def build(model):
    if model not in NOTEBOOKS:raise ValueError('Unknown model')
    boot=(ROOT/'scripts/kaggle_bootstrap.py').read_text(encoding='utf-8')
    texts=[('markdown','''# Project-CV · __MODEL__

Một model, hai giai đoạn: **Cityscapes 20 epoch → ACDC replay 20 epoch**. CE control, cùng data/training/evaluation với hai model còn lại. Encoder ImageNet + head mới;không nạp checkpoint project cũ.

**Fresh:** code ZIP mới + Cityscapes + ACDC, GPU + Internet. **Resume:** ZIP của chính model này + cùng dataset/mount;điền `RESUME_INPUT`. Khi resume,source code được lấy từ snapshot của lượt chạy đó.

Mỗi phiên dự báo tối đa10giờ;một model có thể cần nhiều phiên. Sau mỗi epoch:transactional last/best,previous generation và atomic resultZIP. Epoch dang dở sẽ chạy lại. Dùng SaveVersion→Save&RunAll và giữ ZIP/output giữa các phiên;checkpoint local không tự đồng bộ ra Drive.

## 1. Input và khôi phục
'''),('code',boot+'''
import os,sys,json,hashlib,time
NOTEBOOK_STARTED=time.monotonic()
os.environ.update(USE_TF='0',USE_TORCH='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import torch
assert torch.cuda.is_available(),'Settings -> Accelerator -> GPU'
print('GPU:',torch.cuda.get_device_name(0),'Torch:',torch.__version__)

MODEL_NAME='__MODEL__'
CODE_INPUT=Path('/kaggle/input')
CITYSCAPES_ROOT=None
ACDC_ROOT=None
RESUME_INPUT=None  # This model's result ZIP or extracted dataset root
SEED=42
WORK=Path('/kaggle/working/campaign___MODEL__')
REPO=Path('/kaggle/working/project___MODEL__')
ARCHIVE=Path('/kaggle/working/__MODEL___seed42.zip')
assert not WORK.exists() and not REPO.exists(),'Use a fresh session;preserve the previous result ZIP first'
if RESUME_INPUT is None:
    copy_source(resolve_source(CODE_INPUT,Path('/kaggle/input')),REPO)
    WORK.mkdir(parents=True)
else:
    previous=Path(RESUME_INPUT)
    if previous.is_file():extract_safe(previous,WORK)
    else:
        found=sorted({p.parent for p in previous.rglob('campaign.json') if (p.parent/'data/data.json').is_file()})
        if (previous/'campaign.json').is_file():found=[previous]
        assert len(found)==1,f'Set RESUME_INPUT to this model workspace: {found}'
        shutil.copytree(found[0],WORK)
    plan=json.loads((WORK/'campaign.json').read_text())
    assert plan['schema']=='selected_models_fresh_ce_20e_v1','Use this model notebook result;do not mix the old multi-model campaign'
    assert [j['model'] for j in plan['jobs']]==[MODEL_NAME,MODEL_NAME],'Result belongs to a different model'
    source=WORK/'source.zip' if (WORK/'source.zip').is_file() else WORK/'source'
    copy_source(source,REPO)
    assert all((REPO/name).is_file() and hashlib.sha256((REPO/name).read_bytes()).hexdigest()==checksum
               for name,checksum in plan['source_hashes'].items()),'Original source mismatch'
    if not (WORK/'source.zip').is_file():
        with zipfile.ZipFile(WORK/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in plan['source_hashes']:z.write(REPO/name,name)
            for name in ('requirements.txt','THIRD_PARTY_NOTICES.md','third_party/transformers/LICENSE'):
                if (REPO/name).is_file():z.write(REPO/name,name)
    SEED=plan['seed']

def unique_root(hint,marker,required):
    if hint is not None:root=Path(hint)
    else:
        found=sorted({p.parent for p in Path('/kaggle/input').rglob(marker)
                      if p.is_dir() and all((p.parent/item).is_dir() for item in required)})
        assert len(found)==1,f'Set dataset root explicitly: {found}'
        root=found[0]
    assert all((root/item).is_dir() for item in required),f'Invalid root: {root}'
    return root
CITYSCAPES_ROOT=unique_root(CITYSCAPES_ROOT,'leftImg8bit',('leftImg8bit/train','gtFine/train'))
ACDC_ROOT=unique_root(ACDC_ROOT,'rgb_anon_trainvaltest',('rgb_anon_trainvaltest/rgb_anon','gt_trainval/gt'))
os.chdir(REPO);sys.path.insert(0,str(REPO))
print('Model:',MODEL_NAME,'Data:',CITYSCAPES_ROOT,ACDC_ROOT)
'''),('markdown','''## 2. Môi trường và dữ liệu

Dataset root tự tìm theo cấu trúc. Torch/TorchVision dùng bản sẵn có của Kaggle. Giữ dataset version/mount khi resume.
'''),('code','''import subprocess
def run(*args):subprocess.run([sys.executable,'-u',*map(str,args)],check=True,cwd=REPO)
run('-m','pip','install','transformers==4.57.6','numpy>=1.26,<3','Pillow>=10,<12',
    'PyYAML>=6,<7','matplotlib>=3.8,<4','safetensors>=0.4')
import torchvision
assert tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])) >= (2,6),'Use Torch>=2.6'
print('TorchVision:',torchvision.__version__)
from src.experiments.packaging import export_workspace
from scripts.run_research_campaign import prepare_campaign
from IPython.display import display,FileLink
try:
    run('-m','scripts.prepare_research_data','--city',CITYSCAPES_ROOT,'--acdc',ACDC_ROOT,'--out',WORK/'data')
    prepare_campaign(WORK,WORK/'data/data.json',SEED,models=[MODEL_NAME])
except Exception:
    export_workspace(WORK,ARCHIVE)
    print('Preserved setup files:',ARCHIVE)
    raise
'''),('markdown','''## 3. Train source rồi tự chuyển sang ACDC

Mỗi epoch cập nhật ZIP. Khi gần hết cửa sổ phiên,queue dừng ở checkpoint;phiên sau dùng `RESUME_INPUT` để tiếp tục. ETA sau epoch đầu dựa trên thời gian thực tế của giai đoạn đang chạy;chưa đo giai đoạn sau thì chưa có ETA chính xác cho cả lượt.
'''),('code','''remaining=10-(time.monotonic()-NOTEBOOK_STARTED)/3600
try:
    if remaining>0:
        run('-m','scripts.run_research_campaign','--root',WORK,'--data',WORK/'data/data.json',
            '--archive',ARCHIVE,'--models',MODEL_NAME,'--seed',SEED,'--session-hours',remaining)
    else:print('Setup used the work window;preserving for a fresh session.')
finally:
    export_workspace(WORK,ARCHIVE)
    print('Download:',ARCHIVE)
    display(FileLink(str(ARCHIVE)))
'''),('markdown','''## 4. Tiến độ và kết quả

`training_complete` nghĩa là đủ hai giai đoạn×20epoch của model này. Điểm selection là monitor/dev;official-val được chấm riêng sau khi khóa checkpoint.
'''),('code','''from IPython.display import Markdown
from PIL import Image
if (WORK/'status.json').is_file():
    report=json.loads((WORK/'status.json').read_text())
    print('Status:',report['status'])
    rows=['| Stage | Epoch | Best | Selection mIoU | Remaining stage ETA |',
          '|---|---:|---:|---:|---:|']
    for job in report.get('jobs',[]):
        score=job.get('selection_miou');score='pending' if score is None else f'{100*score:.2f}%'
        eta=job.get('estimated_remaining_hours');eta='unmeasured' if eta is None else f'{eta:.1f}h + ZIP/setup'
        rows.append(f"| {job['stage']} | {job['completed_epochs']}/20 | {job.get('best_epoch','')} | {score} | {eta} |")
    display(Markdown('\\n'.join(rows)))
for path in sorted((WORK/'jobs').glob('*/run/report/learning_curves.png')):
    print(path.parent.parent.parent.name);display(Image.open(path))
''')]
    cells=[cell(kind,text.replace('__MODEL__',model)) for kind,text in texts]
    for i,c in enumerate(cells):c['id']=f'{model}-{i:03d}'
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),
        language_info=dict(name='python',version='3.11'),accelerator='GPU'),nbformat=4,nbformat_minor=5)
    path=ROOT/'notebooks'/NOTEBOOKS[model]
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return path


def build_all():
    return [build(model) for model in NOTEBOOKS]


if __name__=='__main__':
    for path in build_all():print('Created',path)
