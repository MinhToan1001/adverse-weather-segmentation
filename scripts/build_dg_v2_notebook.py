"""Build two B2 launchers (two experiments each) and one shared code bundle."""
import ast
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS={
    'baseline_weights':('05_dg_v2_segformer_b2.ipynb','B2 baseline + trọng số lớp','baseline → class_weights'),
    'weather_weights':('06_dg_v2_b2_weather.ipynb','B2 augmentation mới + augmentation và trọng số lớp','weather_aug → weather_aug_class_weights'),
}


def build():
    bootstrap=(ROOT/'scripts/kaggle_bootstrap.py').read_text(encoding='utf-8')
    settings="""from pathlib import Path
# Chỉ sửa các đường dẫn này nếu auto-find không rõ. Dùng cùng code ZIP cho hai tài khoản.
CODE_INPUT=Path('/kaggle/input')  # project-code-dg-v2.zip mới hoặc thư mục code đã giải nén
CITYSCAPES_ROOT=None             # Thư mục chứa leftImg8bit/train và gtFine/train; None = tự tìm
RESUME_INPUT=None                # Lượt đầu: None. Chạy tiếp: Path('...') đến kết quả CÙNG NHÓM
"""
    setup=bootstrap+'''
import os,sys,time,json,subprocess
MARKERS=('scripts/run_dg_v2_suite.py','src/training/dg_v2.py',
    'configs/dg_v2/appearance.yaml','configs/dg_v2/splits/fit.csv','configs/dg_v2/splits/dev.csv')
MODEL='segformer_b2'
if not globals().get('_DG_BOOTSTRAP_DONE'):STARTED=time.monotonic()
BASE=Path('/kaggle/working')
WORK=BASE/f'dg_v2_{MODEL}_{GROUP}_seed42'
REPO=BASE/f'project_dg_v2_{GROUP}'
ARCHIVE=BASE/f'dg_v2_{MODEL}_{GROUP}_seed42.zip'
settings_identity=(str(CODE_INPUT),str(RESUME_INPUT),GROUP)
if globals().get('_DG_BOOTSTRAP_DONE'):
    assert settings_identity==_DG_BOOTSTRAP_SETTINGS,'Đã đổi input giữa phiên. Giữ ZIP rồi dùng phiên mới.'
    assert CITYSCAPES_ROOT is None or Path(CITYSCAPES_ROOT)==_DG_CITY_ROOT,'Đã đổi Cityscapes mount giữa phiên.'
    CITYSCAPES_ROOT=_DG_CITY_ROOT
    assert REPO.is_dir(),'Code folder missing; preserve outputs and use a fresh session'
    print('Dùng lại code và checkpoint trong phiên hiện tại.')
else:
    assert not WORK.exists() and not REPO.exists(),f'Thư mục đã có: {WORK}. Giữ kết quả trước; dùng phiên mới để đổi input.'
    if RESUME_INPUT is None:
        copy_source(resolve_source(CODE_INPUT,Path('/kaggle/input')),REPO)
    else:
        previous=Path(RESUME_INPUT)
        if previous.is_file():extract_safe(previous,WORK)
        else:
            found=[previous] if (previous/'suite_plan.json').is_file() else [p.parent for p in previous.rglob('suite_plan.json')]
            assert len(found)==1,'Set RESUME_INPUT to one DG-v2 result folder or ZIP'
            shutil.copytree(found[0],WORK)
        plan=json.loads((WORK/'suite_plan.json').read_text())
        assert (plan['protocol']=='cityscapes_groupdev_dg_v2' and plan['model']==MODEL
            and plan['seed']==42 and plan['group']==GROUP),'Sai nhóm kết quả. Dùng notebook và ZIP cùng nhóm.'
        copy_source(WORK/'source.zip' if (WORK/'source.zip').exists() else WORK/'source',REPO)
        import hashlib
        assert all(hashlib.sha256((REPO/n).read_bytes()).hexdigest()==h for n,h in plan['source_hashes'].items()),'Source changed'
        if CITYSCAPES_ROOT is not None:
            assert Path(CITYSCAPES_ROOT)==Path(plan['city_root']),'Resume requires the original Cityscapes mount'
        CITYSCAPES_ROOT=Path(plan['city_root'])
    if CITYSCAPES_ROOT is None:
        found=sorted({p.parent for p in Path('/kaggle/input').rglob('leftImg8bit')
            if (p/'train').is_dir() and (p.parent/'gtFine/train').is_dir()})
        assert len(found)==1,f'Set CITYSCAPES_ROOT explicitly: {found}'
        CITYSCAPES_ROOT=found[0]
    assert all((Path(CITYSCAPES_ROOT)/name).is_dir() for name in ['leftImg8bit/train','gtFine/train']),'Cityscapes train folders missing'
    _DG_BOOTSTRAP_SETTINGS=settings_identity
    _DG_CITY_ROOT=Path(CITYSCAPES_ROOT)
    _DG_BOOTSTRAP_DONE=True
os.chdir(REPO);sys.path.insert(0,str(REPO))
os.environ.update(USE_TF='0',USE_TORCH='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
def run(*args):subprocess.run([sys.executable,'-u',*map(str,args)],check=True,cwd=REPO)
print('Cityscapes:',CITYSCAPES_ROOT,'Model:',MODEL,'Group:',GROUP)
'''
    install='''run('-m','pip','install','transformers==4.57.6','numpy>=1.26,<3','Pillow==11.3.0',
    'PyYAML>=6,<7','matplotlib>=3.8,<4','safetensors>=0.4')
import torch
assert torch.cuda.is_available(),'Enable GPU'
assert tuple(map(int,torch.__version__.split('+')[0].split('.')[:2]))>=(2,6),'Require PyTorch>=2.6'
from src.experiments.run_support import snapshot_source
if WORK.exists() and not (WORK/'source.zip').is_file():snapshot_source(WORK/'source.zip',plan['source_hashes'])
print('GPU:',torch.cuda.get_device_name(0),'Torch:',torch.__version__)
'''
    prepare='''run('-m','scripts.run_dg_v2_suite','--city',CITYSCAPES_ROOT,'--root',WORK,
    '--archive',ARCHIVE,'--model',MODEL,'--group',GROUP,'--prepare-only')
print((WORK/'data/split.json').read_text())
print((WORK/'data/class_audit.csv').read_text())
from IPython.display import Image,display
display(Image(filename=str(WORK/'data/preview.png')))
print('INPUT CHECK PASS. Hai thí nghiệm trong nhóm:',json.loads((WORK/'suite_plan.json').read_text())['arms'])
print('Preview chứa ảnh dataset: giữ notebook Private.')
'''
    execute='''remaining=10-(time.monotonic()-STARTED)/3600
if remaining>0:
    run('-m','scripts.run_dg_v2_suite','--city',CITYSCAPES_ROOT,'--root',WORK,
        '--archive',ARCHIVE,'--model',MODEL,'--group',GROUP,'--session-hours',remaining)
else:print('Setup used session window. Preserve ZIP and resume in a fresh session.')
'''
    summary='''from IPython.display import FileLink,display
status=json.loads((WORK/'suite_status.json').read_text())
print('Group:',status['group'],'Status:',status['status'])
for job in status['jobs']:print(job)
print('Download:',ARCHIVE);display(FileLink(str(ARCHIVE)))
if status['status']!='suite_complete':print('Giữ ZIP; mở phiên mới cùng notebook, đặt RESUME_INPUT để chạy tiếp.')
'''
    paths=[]
    for group,(name,title,sequence) in NOTEBOOKS.items():
        intro=(f'# DG v2 · {title}\n\n'
            f'**Nhóm `{group}`: {sequence}. Mỗi thí nghiệm 20 epoch, khởi tạo B2 độc lập.** '
            'Hai notebook có thể chạy song song trên hai tài khoản; dùng cùng code ZIP, Cityscapes và seed42. '
            'Không sửa ARM/loss/model. Các mã V0–V3 cũ đã thay bằng tên mô tả.\n\n'
            '**Hai input cho lượt mới: code ZIP `project-code-dg-v2.zip` mới + Cityscapes train2975.** '
            'Bật GPU/Internet và giữ Private. Không cần ACDC hay weights model cũ. '
            'Chạy tới INPUT CHECK PASS để xem dữ liệu/preview, rồi Save Version → Save & Run All.\n\n'
            'Tập huấn luyện2380 và tập chọn model595 chia theo thành phố; danh sách có trong configs/dg_v2/splits. '
            'Chọn best bằng 0,5 clean-dev + 0,5 trung bình dev low-light/noise/haze cố định. '
            'Baseline có augmentation RGB cơ bản; hai thí nghiệm weather đổi sang mixture mở rộng.\n\n'
            'Sau mỗi epoch committed, lưu best/last, optimizer/scheduler/scaler/RNG và ZIP nhóm này. '
            'Gần cửa sổ10giờ tự dừng; tải ZIP rồi dùng cùng notebook ở phiên mới với RESUME_INPUT. '
            'Hard-kill có thể bỏ qua finally: lấy ZIP hoàn chỉnh gần nhất. '
            '**Mỗi nhóm cần đủ hai thí nghiệm20/20; suite_complete chỉ nói nhóm này đã xong.** '
            'Chưa có kết quả GPU DG-v2; timing cần đo thực tế.')
        parts=[('markdown',intro),('code',settings),('code',f'GROUP={group!r}\n'+setup),('code',install),
            ('markdown','## 1. Kiểm input và ảnh tăng cường\nChỉ đọc Cityscapes train; không đọc official-val/test hoặc ACDC.'),
            ('code',prepare),('markdown',f'## 2. Tự chạy hai thí nghiệm: {sequence}\nCheckpoint/source/config/weights được khóa; không đổi nhóm giữa phiên.'),
            ('code',execute),('markdown','## 3. Tải ZIP của nhóm này\nGửi lại cả hai ZIP để kiểm tra và ghép bảng điểm source-dev. ACDC đánh giá sau khi chốt phương pháp.'),('code',summary)]
        cells=[]
        for i,(kind,text) in enumerate(parts):
            if kind=='code':ast.parse(text)
            cell=dict(cell_type=kind,id=f'{group}-{i:02d}',metadata={},source=text.splitlines(keepends=True))
            if kind=='code':cell.update(execution_count=None,outputs=[])
            cells.append(cell)
        notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
            language_info=dict(name='python')),nbformat=4,nbformat_minor=5)
        path=ROOT/'notebooks'/name
        path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
        paths.append(path)
    bundle=ROOT/'outputs/kaggle/project-code-dg-v2.zip';bundle.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for directory in ('src','scripts','configs','tests'):
            for p in sorted((ROOT/directory).rglob('*')):
                if p.is_file() and p.suffix in {'.py','.yaml','.yml','.json','.csv'}:z.write(p,p.relative_to(ROOT).as_posix())
        for name in ('requirements.txt','pyproject.toml','THIRD_PARTY_NOTICES.md','third_party/transformers/LICENSE'):
            z.write(ROOT/name,name)
    for path in [*paths,bundle]:print(path)
    return paths[0],bundle


if __name__=='__main__':build()
