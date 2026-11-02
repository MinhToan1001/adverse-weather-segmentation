# GPU warm-up review and baseline config v1

Reviewed: 2026-11-02. Kaggle artifact: E:/Download/kaggle_warmup.zip.
Technical outcome: PASS. Khang-task06 deliverables are present; full baseline
has not started. This review does not close project gates or certify final model
quality. Code remains local on khang, not committed/pushed.

| Epoch | Train loss | mIoU on 20-image source val |
|---|---:|---:|
| 1 | 2.771777 | 13.4902% |
| 2 | 2.520373 | 15.8626% |
| 3 | 2.210898 | 16.1903% |

Loss fell 20.235%. Source subset metrics are preliminary, not full validation
or ACDC results. Best epoch is 3; best and last model tensor contents are equal.

Hardware: Tesla T4, PyTorch 2.11.0+cu128, CUDA 12.8, float16 AMP enabled.
Peak allocated memory: 0.9867 GiB; peak reserved: 1.2988 GiB. Each epoch contains
30 batches, 8 optimizer updates with accumulation 4; logs report 24 updates.
Epoch wall times: 14.752, 6.596, 6.662 seconds; total recorded 28.009 seconds.
Steady train time is about 3.31 seconds per 30-image epoch. The initial epoch
includes startup overhead; timings exclude dataset preparation/package downloads.

Both best/ and last/ include config, safetensors and training_state.pt.
The best model reloads locally without network and produces finite logits
(1,19,16,24) for a synthetic (1,3,64,96) input. Head and ignore index are 19/255.
The pickle-based training_state.pt is retained and hashed, not unpickled during
this artifact review. Optimizer-state contents were not independently certified.

## Frozen baseline v1

`configs/segformer_b0_cityscapes_v1.yaml` is an unchanged copy of the baseline
config checked against the GPU warm-up for model, preprocessing and batch/AMP
settings. Full run uses all 2,975 train / 500 val samples, 512x1024 canvas,
micro-batch 1, accumulation 4, AdamW lr 6e-5, weight decay 0.01, 30 epochs,
1,500 optimizer-step warm-up and poly decay, seed 42, two workers, AMP float16.
The short GPU warm-up used 3 epochs and no LR warm-up; it is a separate experiment.

Small-subset extrapolation predicts 405.90 seconds/epoch and 3.38 GPU-hours for
30 epochs. Allocate **6 GPU-hours as a planning budget**, with the original
protocol upper budget of 20 GPU-hours. This is not a runtime guarantee; real
reads/cache/full validation/checkpoint I/O and session behavior can differ.
Monitor the first full epoch and revise the resource estimate transparently if
needed, without tuning on ACDC. The trainer has no automatic wall-time cap.
Provider quota/session availability is not certified by this run.

## Start the full baseline in the current Kaggle session

Warm-up and full run have separate output directories. Initialize the full model
from ImageNet MiT-B0 again; do not resume the warm-up checkpoint. Use the reviewed
full config already present in the Kaggle source. Its model/data/batch settings
match frozen v1; verify 30 epochs, full source manifests and warmup_steps=1500.

```python
run('-m', 'src.training.train', '--config', 'configs/segformer_b0_cityscapes.yaml')
```

Run once, or enable RUN_FULL_BASELINE in the existing notebook (not both).
Preserve best/, last/, config.json, history.jsonl and resources.jsonl. Resume only
from the last/ directory of this full run, with its original config/horizon.
Do not evaluate ACDC until the full baseline is frozen.

## Evidence and limits

Evidence is stored under `results/warmup-2026-11-02/`; model files remain under
ignored `outputs/kaggle-warmup-2026-11-02/`. The original submitted report keeps
budget_frozen=false; the separate review-decision.json records this review.
Artifact hashes, logs, loss curve and checkpoint verification are preserved.
The ZIP omits subset CSVs and exact source fingerprint, limiting exact independent
reproduction of this warm-up. Future full-run exports should include manifests,
config and source hash (no dataset image/label files).
