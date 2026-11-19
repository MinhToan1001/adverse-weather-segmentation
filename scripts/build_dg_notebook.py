"""Build the active Cityscapes-only DG baseline launcher."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    setup = (ROOT / 'scripts/kaggle_bootstrap.py').read_text(encoding='utf-8') + '''
import os,sys,json,hashlib,time
STARTED=time.monotonic()
CODE_INPUT=Path('/kaggle/input')
CITYSCAPES_ROOT=None
RESUME_INPUT=None  # DG result ZIP or extracted DG workspace, never supervised results
WORK=Path('/kaggle/working/dg_b2_baseline')
REPO=Path('/kaggle/working/project_dg')
ARCHIVE=Path('/kaggle/working/dg_b2_baseline_seed42.zip')
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
print('DG baseline: Cityscapes only:',CITYSCAPES_ROOT)
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
        run('-m','scripts.run_dg','--city',CITYSCAPES_ROOT,'--root',WORK,'--archive',ARCHIVE,
            '--session-hours',remaining)
finally:
    if WORK.exists():export_workspace(WORK,ARCHIVE)
    if ARCHIVE.is_file():print('Download:',ARCHIVE);display(FileLink(str(ARCHIVE)))
'''
    texts = [('markdown', '# DG baseline · SegFormer-B2 · Cityscapes only\n\n'
              'Train Cityscapes 2.975 ảnh, chọn best bằng Cityscapes val 500, 20 epoch. '
              '**Không thêm ACDC; không train hoặc chọn best bằng ảnh/nhãn ACDC.** '
              'Đây là baseline, chưa có module DG cải tiến.\n\n'
              'Input: code ZIP DG mới + Cityscapes train/val. Fresh: RESUME_INPUT=None. '
              'Resume: dùng kết quả DG của đúng lượt, cùng mount. Lưu ZIP sau mỗi epoch hoàn thành; '
              'epoch đang dở chạy lại. Không nạp ZIP supervised cũ. Chọn Save Version → Save & Run All.'),
             ('code', setup), ('code', install), ('code', train),
             ('markdown', 'Hoàn thành: giữ dg_plan.json, frozen_candidate.json và run/best. '
              'Chỉ đánh giá ACDC sau khi khóa phương pháp/checkpoint. Điểm ACDC đã xem trong lịch sử '
              'không phải test độc lập. Chưa có kết quả GPU cho protocol mới này.')]
    cells=[]
    for i,(kind,text) in enumerate(texts):
        c=dict(id=f'dg-{i}',cell_type=kind,metadata={},source=text.splitlines(True))
        if kind=='code':ast.parse(text);c.update(outputs=[],execution_count=None)
        cells.append(c)
    path=ROOT/'notebooks/01_dg_segformer_b2_baseline.ipynb'
    path.write_text(json.dumps(dict(cells=cells,nbformat=4,nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),
                      language_info=dict(name='python'),accelerator='GPU')),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return path
