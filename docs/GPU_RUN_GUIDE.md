# Run the baseline on Colab or Kaggle

GPU execution will use Colab or Kaggle, as selected by the user. Local Windows
is used for data preparation and QA. No full training run has started.

## Transfer the checked code

The local khang checkout has uncommitted work and has not been pushed. A clone
of remote khang alone does not yet contain the verified pipeline. Transfer the
reviewed source checkout or publish the intended changes first. Raw datasets,
checkpoints, environment directories and local visual QA images stay out of git.

## Prepare a writable working directory

Use a writable repository directory under /content (Colab) or /kaggle/working
(Kaggle). Kaggle input mounts are read-only. Place datasets on storage accessible
to your own session. ACDC is not required for Cityscapes training.

Use the environment's CUDA PyTorch and confirm availability before training:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
assert torch.cuda.is_available(), 'Select an accelerator in the session settings'
print(torch.cuda.get_device_name(0))
```

Install the remaining project requirements in that environment and record exact
versions. The local tested transformers version is 4.57.6; the local CPU PyTorch
wheel must not replace the environment's CUDA build. Re-run tests in the GPU
environment; local CPU tests do not certify CUDA AMP.

## Build manifests in the GPU environment

Run from the repository root. Substitute CITYSCAPES_ROOT with the mounted folder
containing leftImg8bit/ and gtFine/. Processed masks must be written to the
writable working directory. Prefer a fresh output tree; converter refuses to
overwrite existing masks. Existing validated processed masks may instead be
transferred and checked without converting again.

```bash
python scripts/convert_labels.py --root CITYSCAPES_ROOT --output-root data/processed/cityscapes --workers 4
python scripts/make_manifest.py --dataset cityscapes --root CITYSCAPES_ROOT --mask-root data/processed/cityscapes
python scripts/validate_manifest.py --manifest data/manifests/cityscapes_train.csv data/manifests/cityscapes_val.csv --check-files --check-labels
```

Regenerate CSVs when mount paths differ; do not reuse Windows absolute paths.
The local CSVs use repo-relative data paths and work unchanged only if that
layout is preserved. Use --overwrite explicitly if replacing existing CSVs.

## Warm-up before the full run

Measure a source-only subset run, GPU memory and optimizer-step time first.
Choose the full-run budget from those measurements. Keep subset results separate
from the full baseline. Config changes must be recorded before the run; do not
tune using ACDC. Use a new output directory for each new run.

```bash
python -m src.training.train --config configs/segformer_b0_cityscapes.yaml
python -m src.training.train --config configs/segformer_b0_cityscapes.yaml --resume outputs/segformer_b0_cityscapes/last
```

Save best/, last/, config.json and history.jsonl to persistent storage regularly.
Resume restores the last completed epoch with the original config/horizon;
it cannot recover batches since the last saved checkpoint. Session interruption
and storage quotas depend on the provider; no full-run duration is promised.
Only evaluate ACDC after the baseline checkpoint/config have been frozen.
