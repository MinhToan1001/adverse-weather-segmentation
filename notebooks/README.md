# Kaggle notebooks

Import `01_kaggle_prepare_cityscapes.ipynb` to prepare/QA source data, and
`02_kaggle_train_segformer_b0.ipynb` to warm up/train/resume on GPU.

Add your own Cityscapes dataset as a private Kaggle input and add the accompanying
local `outputs/kaggle/project-code.zip` as another code input. This ZIP contains
source/config/tests only, with no raw data, masks, checkpoints or secrets.
Use the reviewed main checkout. Run `python scripts/build_kaggle_notebooks.py`
to build the source ZIP locally; generated ZIPs and model weights are not committed.

Edit SOURCE_ZIP and CITYSCAPES_ROOT in the first cell. For notebook 02, enable a
GPU and Internet. By default it runs only a 30-image training / 20-image validation
warm-up. Full training and resume have separate explicit switches. Review measured
GPU runtime/memory before enabling the full baseline.

Resume in another session needs the original config, stable data paths and the
complete last/ directory. Save Version with outputs or export checkpoints before
ending the session. Do not upload Cityscapes data to a public repository/dataset.

The warm-up training flow was executed on Kaggle Tesla T4; the latest bootstrap
and report additions have local tests. See `docs/WARMUP_REVIEW_2026-11-02.md`.
Use `scripts/build_kaggle_notebooks.py` to regenerate them and rebuild the bundle.
GPU execution guide: `docs/GPU_RUN_GUIDE.md`.

Notebook 02 now exports resources.jsonl and report/loss.png, loss.svg, loss.csv, warmup_report.json. Send these with history.jsonl and config.json after GPU warm-up. Budget is never automatically frozen. Re-upload the regenerated source ZIP after code changes.
