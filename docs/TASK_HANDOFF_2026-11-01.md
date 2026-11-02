# Task handoff after both datasets arrived

Date: 2026-11-01. Branch: khang. Code and evidence remain local; no commit/push.

## Toan-task03

All three required manifests exist (Cityscapes train 2,975; val 500; ACDC val
406). Each was regenerated using its real root/mask-root and compared byte for
byte: identical. Cross-manifest paths are disjoint, fields/splits/condition
counts match, and every paired file exists. Generator/validator tests passed.
Source pixel QA passed in the prior full audit. Target pixel QA is deferred by
the frozen DG protocol; it is not replaced with a synthetic quality claim.

## Toan-task04

`docs/data_guide.md`, `DATA_PREPARATION.md`, `DATASET_QA_REPORT.md` provide the
actual package roots, commands, trainId semantics and QA. ACDC's supplied README
confirms semantic labelTrainIds; invalid masks are a separate encoding. License
and README hashes are recorded. Unknown release IDs/provider checksums remain
explicit limitations. Raw data and local visual sheets are not in the code ZIP.

## Toan-task05

`results/in-progress-2026-11-01/dataset-readiness.json` records counts, manifest
hashes, pair regeneration, image/mask bytes and provenance. The source loader was
checked on 20 real train/val samples; target Dataset construction and role/split
checks passed with Image.open blocked, and no target item was iterated. Existing
synthetic target-loader tests cover geometry without using ACDC feedback. The
actual ACDC pixel/loader pass belongs to frozen evaluation. CPU-only local
hardware is documented; cloud GPU resources will be measured under Khang-task06.

The three tasks' preparation artifacts are now available for both datasets under
the pre-freeze protocol. This does not certify archive origin or close Gate B
without its separate review/sign-off.

## Khang-task06 remains in progress

Training now records `resources.jsonl` separately from deterministic history:
train/validation/epoch seconds, optimizer updates, image counts, GPU model,
peak allocated/reserved memory and torch/CUDA versions. `summarize_warmup.py`
exports loss CSV, PNG/SVG and a report. CPU logs cannot produce a GPU estimate.
GPU estimates require matching model, image size, normalization, augmentation,
batch/accumulation/worker and AMP settings between measured/full configs. Budget
remains unfrozen until a real GPU warm-up is reviewed; estimates assume linear
scaling and exclude session uncertainty.

Notebook 02 calls this reporter and remains source-only. Both notebooks and the
source ZIP were regenerated. On Kaggle: enable GPU and Internet; set SOURCE_ZIP
and CITYSCAPES_ROOT; run warm-up, then retain the complete run directory. Return
`report/warmup_report.json`, `resources.jsonl`, `history.jsonl`, `config.json` and
`report/loss.png`. We can then review VRAM/runtime and freeze a feasible config.
Do not launch a full run or evaluate ACDC based on an unreviewed estimate.

## Validation

- Full suite: 82 passed (`results/in-progress-2026-11-01/tests.txt`).
- Final estimator matching/CPU guard change: targeted test passed again.
- `outputs/profiling-check-2026-11-01/verification.json`: exact train/resume model,
  history and scheduler parity still passed after adding resource logging.
- The CPU reporter example is explicitly `CPU_ONLY_NOT_A_GPU_BUDGET`, with null
  projected GPU time. Do not interpret its synthetic loss as a source benchmark.
- Notebook Python cells parse successfully; no Kaggle/Colab GPU execution yet.
