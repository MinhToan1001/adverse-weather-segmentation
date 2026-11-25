"""Ready-to-run legacy controls, each with an independent training workspace."""
import json

from scripts.build_k2_notebook import build as legacy_notebook
from scripts.build_k2_workflow import _write, _source, _display_figures


REFERENCE = '''# Read only source-training artifacts from the completed B0 reference.
from scripts.k2_results import ResultFiles, summarize
if REFERENCE_INPUT is None:
    found = sorted(Path('/kaggle/input').rglob('k2_baseline_comparison_bundle.zip'))
    assert len(found) == 1, f'Attach Version 2 output of the evaluation notebook, or set REFERENCE_INPUT: {found}'
    REFERENCE_INPUT = found[0]
reference = summarize(REFERENCE_INPUT)
with ResultFiles(REFERENCE_INPUT) as files:
    reference_plan = files.json('k2_plan.json')
assert (reference['arm'], reference['recipe'], reference['seed'], reference['planned_epochs']) == (
    'baseline', RECIPE, SEED, EPOCHS), 'Reference must be the completed B0 legacy control'
assert reference['complete'] and reference_plan['batch_size'] == BATCH_SIZE
assert reference['image_size'] == [512,1024]
assert ARM in ('aug','film') and RECIPE == 'legacy_v1' and WEATHER_MODE == 'no-fog'
print('CONTROL:', ARM, '| FiLM:', ARM == 'film', '| auxiliary condition loss: OFF')
print('MATCHED RECIPE:', RECIPE, '| seed:', SEED, '| epochs:', EPOCHS, '| canvas: 512x1024')
print('K1: clean/night/rain/snow, equal sampling, severity 1-5; fog OFF to match the pilot.')
print('B0 reference:', REFERENCE_INPUT)
print('Only this control will train. Existing B0 and K2 checkpoints are not retrained.')
'''


POST = '''# Audit the trained control BEFORE reading any ACDC images or labels.
from scripts.k2_results import compare
from src.utils.artifacts import write_json
state = json.loads((WORK/'status.json').read_text()) if (WORK/'status.json').is_file() else {}
try:
    if state.get('status') == 'training_complete':
        audit = compare([REFERENCE_INPUT, WORK])
        write_json(WORK/'source_control_audit.json', audit)
        print('Matched source configuration, manifests, core code and B0 initialization: PASSED')
        if EVAL_AFTER_TRAIN:
            args = ['-m','scripts.evaluate_k2','--input',WORK,'--out',EVAL_ROOT,
                    '--city',CITYSCAPES_ROOT,'--input-root','/kaggle/input','--device','cuda']
            if ACDC_ROOT is not None: args += ['--acdc',str(ACDC_ROOT)]
            if ACDC_MASK_ROOT is not None: args += ['--acdc-masks',str(ACDC_MASK_ROOT)]
            run(*args)
    else:
        print('Training is incomplete:',state.get('status'),state.get('completed_epochs'))
        print('Resume this SAME arm/seed/recipe/horizon from the full ZIP; target evaluation is deferred.')
finally:
    try:
        if (WORK/'run/history.jsonl').is_file():
            args = ['-m','scripts.plot_k2','--input',WORK,'--out',FIGURES]
            if EVAL_ROOT.exists(): args += ['--evaluation',str(EVAL_ROOT)]
            run(*args)
            if (WORK/'source_control_audit.json').is_file():
                shutil.copy2(WORK/'source_control_audit.json', FIGURES/'source_control_audit.json')
            run('-m','scripts.k2_results','share','--workspace',WORK,'--out',SHARE_ARCHIVE)
            if (WORK/'source_control_audit.json').is_file() and SHARE_ARCHIVE.is_file():
                # Keep the original initialization for the final multi-arm audit,
                # without copying the large optimizer/RNG training-state pickle.
                temporary = CONTROL_AUDIT_ARCHIVE.with_suffix('.partial.zip')
                shutil.copy2(SHARE_ARCHIVE,temporary)
                with zipfile.ZipFile(temporary,'a',zipfile.ZIP_DEFLATED) as bundle:
                    bundle.write(WORK/'run/initial/model.safetensors','run/initial/model.safetensors')
                temporary.replace(CONTROL_AUDIT_ARCHIVE)
    finally:
        if WORK.exists(): export_workspace(WORK,ARCHIVE)
        if EVAL_ROOT.exists(): export_workspace(EVAL_ROOT,EVAL_ARCHIVE)
    for path in [ARCHIVE,SHARE_ARCHIVE,CONTROL_AUDIT_ARCHIVE,EVAL_ARCHIVE]:
        if path.is_file(): print('OUTPUT:',path); display(FileLink(str(path)))
''' + _display_figures()


def build_all():
    original = json.loads(legacy_notebook().read_text(encoding='utf-8'))
    paths = []
    for number,arm,title in [(7,'aug','B0 + K1'),(8,'film','B0 + K1 + FiLM')]:
        setup = _source(original['cells'][1]).replace("ARM = 'baseline'", f"ARM = '{arm}'")
        setup += '''\nRECIPE = 'legacy_v1'
REFERENCE_INPUT = None  # Auto-find k2_baseline_comparison_bundle.zip in the attached evaluation output.
EVAL_AFTER_TRAIN = True
ACDC_ROOT = None
ACDC_MASK_ROOT = None
EVAL_ROOT = BASE / f'k2_{ARM}_seed{SEED}_evaluation'
EVAL_ARCHIVE = BASE / f'k2_{ARM}_seed{SEED}_evaluation.zip'
FIGURES = WORK / 'figures'
CONTROL_AUDIT_ARCHIVE = BASE / f'k2_{ARM}_seed{SEED}_comparison_bundle.zip'
'''
        train = _source(original['cells'][4]).replace("'--arm',ARM", "'--recipe',RECIPE,'--arm',ARM")
        intro = f'''# Legacy control: {title}

This notebook trains **only ARM={arm}**, seed42,20epochs,512x1024,batch2/accumulation4,
the original optimizer/LR/warmup and the original K1 policy. Auxiliary condition loss is OFF.
It compares against the completed legacy B0/K2 pilot; do not switch to balanced_v2 here.

Attach **Cityscapes**, **ACDCdataset**, and **Version2 output** of
`nguyenminhtoan/k2-frozen-evaluation-cityscapes-acdc` (contains `k2_baseline_comparison_bundle.zip`).
Enable GPU + Internet, then **Save Version -> Save & Run All** to persist checkpoints and outputs.
Use a separate notebook/session for the other control. GPU availability and quota may limit concurrency.

Train only Cityscapes train2975; select best on Cityscapes val500. After the complete20epoch
horizon and a successful source/initialization audit, evaluate Cityscapes val500 + ACDC val406.
ACDC is exploratory validation, not hidden test. K3/K4 are OFF. No accuracy gain is guaranteed.

K1 matches the existing pilot: clean/night/rain/snow equally sampled, severity1-5, no metric fog.
The `film` arm also has an unused condition head, with zero loss and no condition supervision.
'''
        paths.append(_write(f'{number:02d}_k2_{arm}_legacy_train_eval.ipynb',[
            ('markdown',intro),('code',setup),('code',_source(original['cells'][2])),
            ('code',REFERENCE),('code',train),('code',POST),
            ('markdown','The full k2_<arm>_seed42.zip contains resume state; keep its original arm/seed/20epoch horizon '
             'and Cityscapes mount. Training stops before the9hour budget when another epoch will not fit. '
             'If incomplete, use a fresh session with RESUME_INPUT set to that full ZIP. The share ZIP is for '
             'inference/review only. Evaluation failures preserve the full trained checkpoint; use notebook06 '
             'after fixing inputs. The comparison_bundle ZIP also includes original initialization weights '
             'for the final matched-control audit, without optimizer state. Do not resume aug as film, '
             'or resume from the B0/K2 checkpoints.')]))
    return paths


if __name__ == '__main__':
    for path in build_all(): print(path)
