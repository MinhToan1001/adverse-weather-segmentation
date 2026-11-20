"""Build opt-in optimized training and evaluation-only notebooks from shared cells."""
import ast
import json
from pathlib import Path

from scripts.build_k2_notebook import build as legacy_notebook


ROOT=Path(__file__).resolve().parents[1]


def _source(cell):
    return ''.join(cell['source'])


def _cell(kind,text,index):
    if kind=='code':ast.parse(text)
    cell=dict(id=f'workflow-{index}',cell_type=kind,metadata={},source=text.splitlines(True))
    if kind=='code':cell.update(outputs=[],execution_count=None)
    return cell


def _write(name,contents):
    notebook=dict(nbformat=4,nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),
                      language_info=dict(name='python'),accelerator='GPU'),
        cells=[_cell(kind,text,index) for index,(kind,text) in enumerate(contents)])
    path=ROOT/'notebooks'/name
    path.write_text(json.dumps(notebook,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return path


def _display_figures():
    return '''from IPython.display import Image as NotebookImage, display, FileLink
for name in ['training_pipeline','k2_architecture','learning_curves','class_learning','best_class_iou','condition_learning','weather_miou']:
    path = FIGURES / (name+'.png')
    if path.is_file(): display(NotebookImage(filename=str(path)))
if EVAL_ROOT.exists():
    for path in sorted(EVAL_ROOT.glob('*/panels/*.png')):
        display(NotebookImage(filename=str(path)))
'''


def build_all():
    original=json.loads(legacy_notebook().read_text(encoding='utf-8'))
    setup=_source(original['cells'][1])
    setup=setup.replace('EPOCHS = 20','EPOCHS = 40')
    setup=setup.replace("GIT_REF = 'feat/k1-weather-augmentation'", """GIT_REF = 'feat/k1-weather-augmentation'
RECIPE = 'balanced_v2'
EVAL_AFTER_TRAIN = True
ACDC_ROOT = None  # Auto-find rgb_anon/{fog,night,rain,snow}/val AFTER checkpoint freeze.
ACDC_MASK_ROOT = None  # Set separately only if gt/ is mounted in another dataset.""")
    setup=setup.replace("f'k2_", "f'k2_balanced_v2_")
    setup=setup.replace("assert plan['protocol'] == 'weatherseg_k2_source_v1'", """assert plan['protocol'] == 'weatherseg_k2_source_v1'
    assert plan.get('recipe','legacy_v1') == RECIPE, 'Use notebook04 to resume a legacy run'""")
    setup += '''\nEVAL_ROOT = BASE / f'k2_balanced_v2_{ARM}_seed{SEED}_evaluation'
EVAL_ARCHIVE = BASE / f'k2_balanced_v2_{ARM}_seed{SEED}_evaluation.zip'
FIGURES = WORK / 'figures'
print('Candidate recipe:', RECIPE, '| 640x1280 | warmup500 | source class-aware crop | cleaner K1')
print('No accuracy gain is guaranteed. Compare all arms under this SAME recipe/horizon.')
'''
    train=_source(original['cells'][4]).replace("'--arm',ARM", "'--recipe',RECIPE,'--arm',ARM")
    post='''# Source diagnostics first; target data is read only for a COMPLETE sealed candidate.
if (WORK/'run/history.jsonl').is_file():
    run('-m','scripts.plot_k2','--input',WORK,'--out',FIGURES)
state = json.loads((WORK/'status.json').read_text()) if (WORK/'status.json').is_file() else {}
try:
    if EVAL_AFTER_TRAIN and state.get('status') == 'training_complete':
        args = ['-m','scripts.evaluate_k2','--input',WORK,'--out',EVAL_ROOT,
                '--city',CITYSCAPES_ROOT,'--input-root','/kaggle/input','--device','cuda']
        if ACDC_ROOT is not None: args += ['--acdc',str(ACDC_ROOT)]
        if ACDC_MASK_ROOT is not None: args += ['--acdc-masks',str(ACDC_MASK_ROOT)]
        run(*args)
    else:
        print('Target evaluation not run:',state.get('status'),'| EVAL_AFTER_TRAIN:',EVAL_AFTER_TRAIN)
        print('Resume unfinished training with its full ZIP. Use notebook06 later for evaluation only.')
finally:
    # Keep trained checkpoints and diagnostics even if a target dataset is missing or evaluation fails.
    try:
        if (WORK/'run/history.jsonl').is_file():
            args = ['-m','scripts.plot_k2','--input',WORK,'--out',FIGURES]
            if EVAL_ROOT.exists(): args += ['--evaluation',str(EVAL_ROOT)]
            run(*args)
            run('-m','scripts.k2_results','share','--workspace',WORK,'--out',SHARE_ARCHIVE)
    finally:
        if WORK.exists(): export_workspace(WORK,ARCHIVE)
        if EVAL_ROOT.exists(): export_workspace(EVAL_ROOT,EVAL_ARCHIVE)
    for path in [ARCHIVE,SHARE_ARCHIVE,EVAL_ARCHIVE]:
        if path.is_file(): print('OUTPUT:',path); display(FileLink(str(path)))
''' + _display_figures()
    optimized=_write('05_k2_balanced_train_eval.ipynb',[
        ('markdown', '# K2 · Optimized candidate recipe + post-training evaluation\n\n'
         '**Recipe mới để thử, chưa chứng minh tăng điểm và không hứa vượt80%.** '
         'Giữ nguyên B0; tăng canvas640×1280, warmup500, crop ưu tiên lớp ít xuất hiện, '
         '40epoch,4workers; K1 dùng50% ảnh sạch và severity1–3. Mặc định chạy baseline mới trước.\n\n'
         'Gắn **Cityscapes + ACDC**, bật **GPU + Internet**. Train chỉ Cityscapes train2975, '
         'chọn best trên Cityscapes val500. Sau khi đủ40epoch và khóa checkpoint, tự đánh giá '
         'Cityscapes val500 + ACDC val406, xuất PNG/SVG và ZIP. ACDC test chính thức giấu nhãn, '
         'nên đây là đánh giá validation có nhãn, không phải điểm hidden test.\n\n'
         'Chạy một ARM mỗi phiên. Baseline/K2 phải cùng recipe mới; không quy chênh lệch so với '
         'run20epoch cũ cho FiLM. Thời gian/VRAM cần đo thực tế, có thể phải resume nhiều phiên. '
         'Checkpoint40epoch không được tạo bằng cách sửa horizon của ZIP20epoch. K3/K4 chưa bật.'),
        ('code',setup),('code',_source(original['cells'][2])),('code',_source(original['cells'][3])),
        ('code',train),('code',post),
        ('markdown','Nếu train chưa đủ40epoch, dùng ZIP đầy đủ để resume. Nếu train xong nhưng evaluation '
         'thiếu dữ liệu/hết phiên, dùng notebook06 với ZIP đã lưu để chỉ evaluate, không train lại. '
         'ZIP share chứa best model + log + biểu đồ; ZIP evaluation chứa metrics, confusion và ảnh dự đoán. '
         'Điểm ACDC được xem trong phát triển phải ghi là exploratory; không tune theo ACDC rồi gọi là test độc lập.')])
    bootstrap=(ROOT/'scripts/kaggle_bootstrap.py').read_text(encoding='utf-8')
    evaluate_setup=bootstrap+'''
import os, sys, subprocess
RESULT_INPUT = None  # Attach one full/shared K2 result ZIP/folder; set explicitly when multiple results exist.
CITYSCAPES_ROOT = None
ACDC_ROOT = None
ACDC_MASK_ROOT = None
CODE_INPUT = None
GIT_REF = 'feat/k1-weather-augmentation'
BASE = Path('/kaggle/working')
REPO = BASE/'k2_evaluation_code'
EVAL_ROOT = BASE/'k2_post_training_evaluation'
EVAL_ARCHIVE = BASE/'k2_post_training_evaluation.zip'
FIGURES = EVAL_ROOT/'figures'
assert not REPO.exists() and not EVAL_ROOT.exists(), 'Use a fresh session/output; preserve existing evaluation'
if CODE_INPUT is not None:
    copy_source(resolve_source(Path(CODE_INPUT),Path('/kaggle/input')),REPO)
else:
    subprocess.run(['git','clone','--depth','1','--single-branch','--branch',GIT_REF,
        'https://github.com/MinhToan1001/adverse-weather-segmentation.git',str(REPO)],check=True)
assert (REPO/'scripts/evaluate_k2.py').is_file(), 'Need the latest K2 evaluation code bundle'
os.chdir(REPO);sys.path.insert(0,str(REPO))
os.environ.update(USE_TF='0',USE_TORCH='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
print('Evaluation only: no optimizer, no target weight updates, no training restart.')
'''
    evaluate_code='''from scripts.evaluate_k2 import resolve_result
from src.experiments.packaging import export_workspace
from IPython.display import FileLink, display
RESULT_INPUT = resolve_result(RESULT_INPUT,'/kaggle/input')
print('Reading frozen result:',RESULT_INPUT)
args = ['-m','scripts.evaluate_k2','--input',RESULT_INPUT,'--out',EVAL_ROOT,
        '--input-root','/kaggle/input','--device','cuda']
if CITYSCAPES_ROOT is not None: args += ['--city',str(CITYSCAPES_ROOT)]
if ACDC_ROOT is not None: args += ['--acdc',str(ACDC_ROOT)]
if ACDC_MASK_ROOT is not None: args += ['--acdc-masks',str(ACDC_MASK_ROOT)]
try:
    run(*args)
finally:
    try:
        if EVAL_ROOT.exists():
            run('-m','scripts.plot_k2','--input',RESULT_INPUT,'--out',FIGURES,'--evaluation',EVAL_ROOT)
    finally:
        if EVAL_ROOT.exists():
            export_workspace(EVAL_ROOT,EVAL_ARCHIVE)
            print('Evaluation ZIP:',EVAL_ARCHIVE);display(FileLink(str(EVAL_ARCHIVE)))
''' + _display_figures()
    evaluation=_write('06_k2_evaluate_report.ipynb',[
        ('markdown','# K2 · Evaluate a completed checkpoint, no retraining\n\n'
         'Gắn **ZIP kết quả K2/baseline + Cityscapes + ACDC**, bật **GPU + Internet**, Run All. '
         'Dùng được cho recipe20epoch cũ và balanced_v2 đã đủ horizon. Nếu có nhiều ZIP, đặt RESULT_INPUT.\n\n'
         'Kiểm seal/hash/source trước khi đọc target; giữ nguyên weights. Báo Cityscapes val500, '
         'ACDC val406 theo từng thời tiết/lớp, confusion và ảnh RGB/GT/prediction/error. '
         'Đây không phải hidden-test score hoặc K4 adaptation; không tự chọn checkpoint bằng ACDC.'),
        ('code',evaluate_setup),('code',_source(original['cells'][2])),('code',evaluate_code),
        ('markdown','Tải k2_post_training_evaluation.zip trong Output. Nếu lần đánh giá trước bị ngắt, '
         'giữ ZIP để kiểm tra rồi dùng phiên/output mới cho lần đánh giá lại cùng checkpoint. '
         'Nhãn test ACDC chính thức được giữ trên server; muốn điểm test cần nộp dự đoán riêng.')])
    return [optimized,evaluation]


if __name__=='__main__':
    for path in build_all():print(path)
