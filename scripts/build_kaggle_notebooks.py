"""Build importable Kaggle notebooks and the exact local source bundle."""
import ast
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]

def md(text): return dict(cell_type='markdown',metadata={},source=text.splitlines(True))
def code(text):
    ast.parse(text)
    return dict(cell_type='code',metadata={},source=text.splitlines(True),execution_count=None,outputs=[])
def save(name,cells):
    for i,c in enumerate(cells): c['id']=f'cell-{i:03d}'
    notebook=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},'accelerator':'GPU'},nbformat=4,nbformat_minor=5)
    (ROOT/'notebooks'/name).write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

setup=code('''from pathlib import Path
import os, sys, subprocess, zipfile, json, hashlib
# EDIT these two paths to match your private Kaggle input datasets.
SOURCE_ZIP = Path('/kaggle/input/YOUR_CODE_DATASET/project-code.zip')
CITYSCAPES_ROOT = Path('/kaggle/input/YOUR_CITYSCAPES_DATASET/cityscapes')
REPO = Path('/kaggle/working/adverse-weather-segmentation')
# Kaggle may automatically extract an uploaded ZIP into the dataset root.
SOURCE_ROOT = None  # Optional folder containing src/, scripts/, configs/ if ZIP was extracted
assert (CITYSCAPES_ROOT/'leftImg8bit/train').is_dir(), 'Set CITYSCAPES_ROOT to the folder containing leftImg8bit and gtFine'
REPO.mkdir(parents=True,exist_ok=True)
if SOURCE_ZIP.is_file():
    with zipfile.ZipFile(SOURCE_ZIP) as bundle:
        for member in bundle.infolist():
            target=(REPO/member.filename).resolve()
            assert target.is_relative_to(REPO.resolve()), 'Unsafe archive member'
        bundle.extractall(REPO)
    print('Code ZIP SHA256:',hashlib.sha256(SOURCE_ZIP.read_bytes()).hexdigest())
else:
    import shutil
    source=Path(SOURCE_ROOT) if SOURCE_ROOT else SOURCE_ZIP.parent
    assert all((source/name).is_dir() for name in ['src','scripts','configs']), f'Set SOURCE_ROOT to extracted code folder: {source}'
    for name in ['src','scripts','configs','tests','requirements.txt','pyproject.toml']:
        item=source/name
        if item.is_dir():
            shutil.copytree(item,REPO/name,dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item,REPO/name)
    print('Copied extracted source from:',source)
os.chdir(REPO)
sys.path.insert(0,str(REPO))
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
def run(*args):
    subprocess.run([sys.executable,*map(str,args)],check=True)
print('Working directory:',REPO)
''')
deps=code('''# Enable Internet for first-time package and pretrained-weight downloads.
# Preserve Kaggle's CUDA PyTorch installation.
run('-m','pip','install','transformers==4.57.6','numpy>=1.26,<3','Pillow>=10,<12','PyYAML>=6,<7','tqdm>=4.66,<5','pytest>=8,<10','matplotlib>=3.8,<4')
import torch, transformers, yaml
print('torch',torch.__version__,'transformers',transformers.__version__)
print('CUDA available:',torch.cuda.is_available())
''')
prepare=code('''# Reuse valid processed masks only when you deliberately set this path.
PROCESSED_ROOT = REPO/'data/processed/cityscapes'
if PROCESSED_ROOT.exists():
    raise RuntimeError('Processed folder already exists. Validate/reuse it deliberately or use a NEW writable folder; no silent overwrite.')
run('scripts/convert_labels.py','--root',CITYSCAPES_ROOT,'--output-root',PROCESSED_ROOT,'--workers','4')
run('scripts/make_manifest.py','--dataset','cityscapes','--root',CITYSCAPES_ROOT,'--mask-root',PROCESSED_ROOT)
run('scripts/validate_manifest.py','--manifest','data/manifests/cityscapes_train.csv','data/manifests/cityscapes_val.csv','--check-files','--check-labels')
''')
save('01_kaggle_prepare_cityscapes.ipynb',[
md('# Chuẩn bị Cityscapes trên Kaggle\n\nUpload notebook này vào Kaggle. Thêm dataset riêng của bạn chứa Cityscapes và `project-code.zip` đi kèm. Sửa hai đường dẫn trong cell đầu. Không public ảnh/nhãn/raw data. Notebook chưa chạy trên Kaggle; code đã được kiểm chứng local. Chỉ dùng train/val Cityscapes, không mở ACDC. Cell chuyển nhãn có thể chạy vài phút.\n'),setup,deps,prepare,
code('''run('scripts/audit_labels.py','--manifest','data/manifests/cityscapes_train.csv','data/manifests/cityscapes_val.csv','--check-files','--check-labels','--raw-root',CITYSCAPES_ROOT,'--out','results/kaggle-cityscapes-qa/audit.json')
print(json.loads(Path('results/kaggle-cityscapes-qa/audit.json').read_text())['status'])
'''),
md('## Lưu kết quả\n\nSave Version của Kaggle với outputs để giữ processed masks và báo cáo. Notebook 02 có thể nhận thư mục `data/processed/cityscapes` này qua input `PREPARED_ROOT`, hoặc tự chuyển lại nhãn trong phiên mới. CSV phải được tạo lại khi mount path thay đổi. Không dùng kết quả chuẩn bị như số liệu segmentation.\n')])
save('02_kaggle_train_segformer_b0.ipynb',[
md('# Warm-up và train SegFormer-B0 trên Kaggle\n\nChọn GPU trong Session options, bật Internet để tải pretrained MiT-B0. Thêm Cityscapes và source ZIP. Mặc định chỉ chạy warm-up 1% train / 20 val; full run và resume đều cần bật rõ. Không dùng ACDC để chọn cấu hình. Notebook dùng source ZIP được tạo từ checkout đã review; ghi lại phiên bản source của lần chạy.\n'),setup,deps,
code('''assert torch.cuda.is_available(), 'Enable a GPU accelerator before training'
print(torch.cuda.get_device_name(0))
# Optional input from outputs of notebook 01; folder must contain gtFine/.
PREPARED_ROOT = None  # e.g. Path('/kaggle/input/YOUR_PREP_NOTEBOOK/data/processed/cityscapes')
if PREPARED_ROOT is None:
    PREPARED_ROOT=REPO/'data/processed/cityscapes'
    if not PREPARED_ROOT.exists():
        run('scripts/convert_labels.py','--root',CITYSCAPES_ROOT,'--output-root',PREPARED_ROOT,'--workers','4')
# Use a fresh source checkout or explicitly review existing CSVs before overwrite.
run('scripts/make_manifest.py','--dataset','cityscapes','--root',CITYSCAPES_ROOT,'--mask-root',PREPARED_ROOT)
run('scripts/validate_manifest.py','--manifest','data/manifests/cityscapes_train.csv','data/manifests/cityscapes_val.csv','--check-files','--check-labels')
'''),
code('''import csv, random
from src.data.validation import read_manifest
from src.data.preparation import write_manifest
rng=random.Random(42)
train_rows=read_manifest('data/manifests/cityscapes_train.csv')
val_rows=read_manifest('data/manifests/cityscapes_val.csv')
write_manifest(rng.sample(train_rows,30),'data/manifests/warmup_train.csv')
write_manifest(rng.sample(val_rows,20),'data/manifests/warmup_val.csv')
cfg=yaml.safe_load(Path('configs/segformer_b0_cityscapes.yaml').read_text())
cfg['data'].update(train_manifest='data/manifests/warmup_train.csv',val_manifest='data/manifests/warmup_val.csv')
cfg['train'].update(epochs=3,num_workers=2,warmup_steps=0)
cfg['output_dir']='outputs/kaggle_warmup'
Path('configs/kaggle_warmup.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
'''),
code('''import time
started=time.perf_counter()
run('-m','src.training.train','--config','configs/kaggle_warmup.yaml')
print('Warm-up wall seconds:',time.perf_counter()-started)
print(Path('outputs/kaggle_warmup/history.jsonl').read_text())
run('scripts/summarize_warmup.py','--run-dir','outputs/kaggle_warmup','--full-config','configs/segformer_b0_cityscapes.yaml','--out','outputs/kaggle_warmup/report')
print(Path('outputs/kaggle_warmup/report/warmup_report.json').read_text())
# Inspect loss.svg, resources.jsonl and the assumptions before freezing any budget.
'''),
code('''RUN_FULL_BASELINE=False  # Set True only after reviewing warm-up and compute budget.
FULL_CONFIG=Path('configs/segformer_b0_cityscapes.yaml')
if RUN_FULL_BASELINE:
    run('-m','src.training.train','--config',FULL_CONFIG)
'''),
code('''RUN_RESUME=False
# Restore the SAME source ZIP, original config, manifest contents/paths and
# the complete last/ checkpoint directory before resuming in a new session.
RESUME_FROM=REPO/'outputs/segformer_b0_cityscapes/last'
if RUN_RESUME:
    assert not RUN_FULL_BASELINE, 'Choose a fresh run OR resume, not both'
    run('-m','src.training.train','--config',FULL_CONFIG,'--resume',RESUME_FROM)
'''),
md('## Giữ checkpoint khi kết thúc phiên\n\nSave Version với outputs, hoặc tải archive bằng cell dưới. Kaggle có thể ngắt phiên; hãy lưu định kỳ. `last` chỉ khôi phục được epoch đã lưu. Giữ cả config.json, history.jsonl và toàn bộ best/last; không chỉ giữ model.safetensors. Đừng thay epoch/LR/batch trong config khi resume. AMP trên GPU cần được kiểm chứng bằng warm-up này; local chỉ kiểm chứng CPU.\n'),
code('''import shutil
EXPORT_RESULTS=False
if EXPORT_RESULTS:
    # Contains generated model artifacts and config/logs, no raw images/labels.
    for run_dir in (REPO/'outputs').iterdir():
        if run_dir.is_dir():
            archive=shutil.make_archive(str(Path('/kaggle/working')/run_dir.name),'zip',root_dir=run_dir)
            print('Export:',archive)
''')])

bundle=ROOT/'outputs/kaggle/project-code.zip'
bundle.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
    for folder in ['src','scripts','configs','tests']:
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and path.suffix in {'.py','.yaml','.yml'}:
                archive.write(path,path.relative_to(ROOT).as_posix())
    for name in ['requirements.txt','pyproject.toml']:
        archive.write(ROOT/name,name)
print('Created 2 notebooks; Python cells parsed successfully; source bundle:',bundle)
