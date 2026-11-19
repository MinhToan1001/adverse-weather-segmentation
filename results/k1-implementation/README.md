# K1 implementation QA · 2026-11-19

- Full Python suite: **189 passed, 4 skipped** (77.45s, CPU; pretrained tests not enabled).
- K1 adds26tests: linear-light fog equation, metric stereo decoding/ray distance,
  explicit invalid handling, deterministic effects/global RNG isolation, unchanged
  mask geometry/condition IDs, preview provenance and actual Kaggle setup/source
  preview cells using an uploaded code ZIP.
- Node API: **1 passed**. `pip check`: no broken requirements.
- All three notebooks/code bundle rebuilt; no executed cells or outputs committed.
- A20-image severity grid on the procedural QA scene was generated and visually
  inspected locally. Image artifacts stay under ignored `outputs/k1-validation/`.

No real Cityscapes images/calibration are available in this local checkout for
visual validation. Tests use synthetic images and analytically specified metric
distances. This does not establish physical realism, target-distribution match,
GPU performance or segmentation mIoU improvement. The Kaggle notebook is for
preview, not training. See [usage and assumptions](../../docs/K1_WEATHER.md).
