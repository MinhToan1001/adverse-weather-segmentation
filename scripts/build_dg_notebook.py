"""Build the active Cityscapes-only DG baseline launcher."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "segformer_b2": "01_dg_segformer_b2_baseline.ipynb",
    "deeplabv3plus_resnet50": "02_dg_deeplabv3plus_resnet50.ipynb",
}


def build(model="segformer_b2"):
    if model not in NOTEBOOKS:raise ValueError("Unknown DG model")
    setup = (ROOT / 'scripts/kaggle_bootstrap.py').read_text(encoding='utf-8') + '''
import os,sys,json,hashlib,time
STARTED=time.monotonic()
MODEL='__MODEL__'
ARM='baseline'  # baseline | aug (two CE views) | consistency (two CE views + KL)
SEED=42
PAIR_PROBABILITY=0.5
CONSISTENCY_WEIGHT=0.1  # candidate setting, not validated for accuracy yet
PROFILE_ONLY=True  # First GPU session: measure timings, discard pilot weights. False runs20epochs.
CODE_INPUT=Path('/kaggle/input')
CITYSCAPES_ROOT=None
RESUME_INPUT=None  # full result ZIP/extracted workspace for this exact model/arm/seed
BASE=Path('/kaggle/working')
RUN_NAME=f'dg_{MODEL}_{ARM}_seed{SEED}'
WORK=BASE/RUN_NAME
REPO=BASE/f'project_{MODEL}_{ARM}'
ARCHIVE=BASE/f'{RUN_NAME}.zip'
assert not WORK.exists() and not REPO.exists(),'Fresh session required; preserve previous result first'
if RESUME_INPUT is None:
    copy_source(resolve_source(CODE_INPUT,Path('/kaggle/input')),REPO)
else:
    previous=Path(RESUME_INPUT)
    if previous.is_file():extract_safe(previous,WORK)
    else:
        candidates=[previous] if (previous/'dg_plan.json').is_file() else [p.parent for p in previous.rglob('dg_plan.json')]
        assert len(candidates)==1,f'Set RESUME_INPUT to one DG workspace: {candidates}'
        shutil.copytree(candidates[0],WORK)
    plan=json.loads((WORK/'dg_plan.json').read_text())
    assert plan['experiment_protocol']=='cityscapes_only_dg_v1','Reject supervised or mismatched results'
    assert plan['model']==MODEL and plan['arm']==ARM and plan['seed']==SEED,'Use the same model/arm/seed'
    PAIR_PROBABILITY=plan['pair_probability'];CONSISTENCY_WEIGHT=plan['consistency_weight']
    copy_source(WORK/'source.zip' if (WORK/'source.zip').is_file() else WORK/'source',REPO)
    assert all(hashlib.sha256((REPO/n).read_bytes()).hexdigest()==h for n,h in plan['source_hashes'].items())
    if not (WORK/'source.zip').is_file():
        with zipfile.ZipFile(WORK/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in plan['source_hashes']:z.write(REPO/name,name)
            for name in ('requirements.txt','THIRD_PARTY_NOTICES.md','third_party/transformers/LICENSE'):
                if (REPO/name).is_file():z.write(REPO/name,name)
    CITYSCAPES_ROOT=Path(plan['city_root'])
assert (REPO/'scripts/run_dg.py').is_file(),'Upload the new DG project-code.zip'
if CITYSCAPES_ROOT is None:
    candidates=sorted({p.parent for p in Path('/kaggle/input').rglob('leftImg8bit') if p.is_dir()
                       and all((p.parent/n).is_dir() for n in ['leftImg8bit/train','gtFine/train','leftImg8bit/val','gtFine/val'])})
    assert len(candidates)==1,f'Set CITYSCAPES_ROOT explicitly: {candidates}'
    CITYSCAPES_ROOT=candidates[0]
os.chdir(REPO);sys.path.insert(0,str(REPO))
os.environ.update(USE_TF='0',USE_TORCH='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
print('DG:',MODEL,ARM,'Cityscapes only:',CITYSCAPES_ROOT)
'''
    install = '''import subprocess
def run(*args):subprocess.run([sys.executable,'-u',*map(str,args)],check=True,cwd=REPO)
run('-m','pip','install','transformers==4.57.6','numpy>=1.26,<3','Pillow>=10,<12',
    'PyYAML>=6,<7','matplotlib>=3.8,<4','safetensors>=0.4')
import torch
assert torch.cuda.is_available(),'Enable GPU'
assert tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]))>=(2,6),'Require Torch>=2.6'
print(torch.__version__,torch.cuda.get_device_name(0))
'''
    train = '''from src.experiments.packaging import export_workspace
from IPython.display import FileLink,display
remaining=10-(time.monotonic()-STARTED)/3600
try:
    if remaining>0:
        args=['-m','scripts.run_dg','--city',CITYSCAPES_ROOT,'--root',WORK,'--archive',ARCHIVE,
              '--session-hours',remaining,'--model',MODEL,'--arm',ARM,'--seed',SEED,
              '--pair-probability',PAIR_PROBABILITY,'--consistency-weight',CONSISTENCY_WEIGHT]
        if PROFILE_ONLY:
            run(*args,'--prepare-only')
            run('-m','scripts.profile_dg','--workspace',WORK,'--out',WORK/'gpu_profile.json',
                '--pair-probability',PAIR_PROBABILITY if PAIR_PROBABILITY else .5,
                '--consistency-weight',CONSISTENCY_WEIGHT if CONSISTENCY_WEIGHT else .1)
            print('Pilot only. Review time/VRAM and the 30h budget before setting PROFILE_ONLY=False.')
        else:run(*args)
finally:
    if WORK.exists():export_workspace(WORK,ARCHIVE)
    if ARCHIVE.is_file():print('Download:',ARCHIVE);display(FileLink(str(ARCHIVE)))
'''
    texts = [('markdown', '# DG · __MODEL__ · Cityscapes only\n\n'
              'Train Cityscapes 2.975 ảnh, chọn best bằng Cityscapes val 500, 20 epoch. '
              '**Không thêm ACDC; không train hoặc chọn best bằng ảnh/nhãn ACDC.** '
              'ARM=baseline là C0; aug là C1; consistency là C2. Hai arm sau có style nguồn; chỉ C2 có KL. '
              'Các arm bắt đầu lại từ ImageNet, không học tiếp từ baseline20epoch. λ=0,1 là cấu hình ứng viên, chưa có kết quả GPU.\n\n'
              '**Lần GPU đầu: PROFILE_ONLY=True** chỉ đo Cityscapes, bỏ weights pilot. Sau khi xem ngân sách, False mới train20epoch.\n\n'
              'Input: code ZIP DG mới + Cityscapes train/val. Fresh: RESUME_INPUT=None. '
              'Resume: dùng kết quả DG của đúng lượt, cùng mount. Lưu ZIP sau mỗi epoch hoàn thành; '
              'epoch đang dở chạy lại. Không nạp ZIP supervised cũ. Chọn Save Version → Save & Run All.'),
             ('code', setup), ('code', install), ('code', train),
             ('markdown', 'Mỗi phiên chỉ chạy một ARM/model. Giữ seed, config và input mount khi resume. '
              'Hoàn thành: giữ dg_plan.json, frozen_candidate.json và run/best. '
              'Chỉ đánh giá ACDC sau khi khóa phương pháp/checkpoint. Điểm ACDC đã xem trong lịch sử '
              'không phải test độc lập. Chưa có kết quả GPU cho protocol mới này.')]
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


def build_all():
    return [build(model) for model in NOTEBOOKS]
