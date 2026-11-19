"""Build one automatic three-strategy DG notebook per model."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS={
    'segformer_b2':'01_dg_segformer_b2.ipynb',
    'deeplabv3plus_resnet50':'02_dg_deeplabv3plus_resnet50.ipynb',
}


def build(model='segformer_b2'):
    if model not in NOTEBOOKS:raise ValueError('Unknown DG model')
    setup=(ROOT/'scripts/kaggle_bootstrap.py').read_text(encoding='utf-8')+"""
import os,sys,json,hashlib,time
STARTED=time.monotonic()
MODEL='__MODEL__'
# Only input paths need configuring. Training policy is frozen in the suite.
CODE_INPUT=Path('/kaggle/input')
CITYSCAPES_ROOT=None
RESUME_INPUT=None  # This model's automatic suite ZIP/extracted workspace, when continuing.
BASE=Path('/kaggle/working')
WORK=BASE/f'dg_{MODEL}_suite_seed42'
REPO=BASE/f'project_{MODEL}_suite'
ARCHIVE=BASE/f'dg_{MODEL}_suite_seed42.zip'
assert not WORK.exists() and not REPO.exists(),'Fresh session required; preserve previous result first'
if RESUME_INPUT is None:
    copy_source(resolve_source(CODE_INPUT,Path('/kaggle/input')),REPO)
else:
    previous=Path(RESUME_INPUT)
    if previous.is_file():extract_safe(previous,WORK)
    else:
        found=[previous] if (previous/'suite_plan.json').is_file() else [p.parent for p in previous.rglob('suite_plan.json')]
        assert len(found)==1,f'Set RESUME_INPUT to one automatic model suite: {found}'
        shutil.copytree(found[0],WORK)
    plan=json.loads((WORK/'suite_plan.json').read_text())
    assert plan['schema']=='automatic_dg_model_suite_v1' and plan['protocol']=='cityscapes_only_dg_v1'
    assert plan['model']==MODEL and plan['seed']==42,'Result belongs to another model/seed'
    copy_source(WORK/'source.zip' if (WORK/'source.zip').is_file() else WORK/'source',REPO)
    assert all(hashlib.sha256((REPO/n).read_bytes()).hexdigest()==h for n,h in plan['source_hashes'].items())
    if not (WORK/'source.zip').is_file():
        with zipfile.ZipFile(WORK/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in plan['source_hashes']:z.write(REPO/name,name)
            for name in ('requirements.txt','THIRD_PARTY_NOTICES.md','third_party/transformers/LICENSE'):
                if (REPO/name).is_file():z.write(REPO/name,name)
    CITYSCAPES_ROOT=Path(plan['city_root'])
assert (REPO/'scripts/run_dg_suite.py').is_file(),'Upload the NEW automatic-suite project-code.zip'
if CITYSCAPES_ROOT is None:
    found=sorted({p.parent for p in Path('/kaggle/input').rglob('leftImg8bit') if p.is_dir()
                  and all((p.parent/n).is_dir() for n in ['leftImg8bit/train','gtFine/train','leftImg8bit/val','gtFine/val'])})
    assert len(found)==1,f'Set CITYSCAPES_ROOT explicitly: {found}'
    CITYSCAPES_ROOT=found[0]
os.chdir(REPO);sys.path.insert(0,str(REPO))
os.environ.update(USE_TF='0',USE_TORCH='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
print('Automatic DG suite:',MODEL,'Cityscapes:',CITYSCAPES_ROOT)
"""
    install="""import subprocess
def run(*args):subprocess.run([sys.executable,'-u',*map(str,args)],check=True,cwd=REPO)
run('-m','pip','install','transformers==4.57.6','numpy>=1.26,<3','Pillow>=10,<12',
    'PyYAML>=6,<7','matplotlib>=3.8,<4','safetensors>=0.4')
import torch
assert torch.cuda.is_available(),'Enable GPU'
assert tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]))>=(2,6),'Require Torch>=2.6'
print(torch.__version__,torch.cuda.get_device_name(0))
"""
    execute="""from src.experiments.packaging import export_workspace
from IPython.display import FileLink,display
remaining=10-(time.monotonic()-STARTED)/3600
try:
    if remaining>0:
        run('-m','scripts.run_dg_suite','--city',CITYSCAPES_ROOT,'--model',MODEL,
            '--root',WORK,'--archive',ARCHIVE,'--session-hours',remaining)
    else:print('Setup used this session window. Preserve inputs and start a fresh session.')
finally:
    # The runner already exports after each epoch and on Python failures.
    # Avoid compressing all checkpoints twice on a normal return.
    if WORK.exists() and not ARCHIVE.is_file():export_workspace(WORK,ARCHIVE)
    if ARCHIVE.is_file():print('Download:',ARCHIVE);display(FileLink(str(ARCHIVE)))
"""
    summary="""if (WORK/'suite_status.json').is_file():
    status=json.loads((WORK/'suite_status.json').read_text())
    print('Suite:',status['status'])
    for job in status.get('jobs',[]):
        print(job['arm'],str(job['completed_epochs'])+'/20',job['status'],
              'Cityscapes val mIoU:',job.get('cityscapes_val_miou'))
    if status['status']=='session_checkpoint_ready':
        print('Preserve this ZIP. Next fresh session: set only RESUME_INPUT; completed epochs/profile are reused.')
"""
    texts=[('markdown','# Automatic DG suite · __MODEL__\n\n'
        '**Một lần Run All:** chuẩn bị Cityscapes → profile GPU một lần → baseline20 → aug20 → consistency20. '
        'Cả B2 và DeepLab đều chạy đủ ba chiến thuật. Không đổi ARM hoặc PROFILE_ONLY. '
        'Mỗi chiến thuật khởi tạo độc lập từ ImageNet, chọn best bằng Cityscapes val500; không ACDC input.\n\n'
        'Chỉ sửa CODE_INPUT/CITYSCAPES_ROOT nếu auto-find không rõ; lượt mới RESUME_INPUT=None. '
        'Lượt tiếp dùng ZIP suite của đúng model. Mỗi epoch cập nhật một ZIP chung. Gần hết cửa sổ phiên10giờ thì lưu/dừng; '
        'phiên sau tiếp tục, không tự giảm20epoch hoặc bỏ aug. Quota30giờ chung do Kaggle quản lý; chưa bảo đảm đủ cả6lượt. '
        'Dùng cùng code ZIP mới cho hai notebook. GPU/AMP/full accuracy còn cần kiểm chứng trên phần cứng.'),
        ('code',setup),('code',install),('code',execute),('code',summary),
        ('markdown','suite_complete nghĩa là đủ3×20epoch của model này. Sau cả hai suite, khóa6candidate rồi mới evaluate ACDC. '
         'Điểm trong bảng là Cityscapes val dùng selection, không phải kết quả ACDC. '
         'Tải ZIP suite trước khi xóa runtime; mất giữa epoch sẽ chạy lại epoch đó.')]
    cells=[]
    for i,(kind,text) in enumerate(texts):
        text=text.replace('__MODEL__',model)
        c=dict(id=f'dg-{model}-{i}',cell_type=kind,metadata={},source=text.splitlines(True))
        if kind=='code':ast.parse(text);c.update(outputs=[],execution_count=None)
        cells.append(c)
    path=ROOT/'notebooks'/NOTEBOOKS[model]
    path.write_text(json.dumps(dict(cells=cells,nbformat=4,nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),
                      language_info=dict(name='python'),accelerator='GPU')),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return path


def build_all():return [build(model) for model in NOTEBOOKS]
