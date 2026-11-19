# K1 + K2 integration QA · 2026-11-19

K2 uses the existing K1 generator and its condition IDs, not the discarded
constant-fog draft from the separate local WIP branch.

- Full suite: **200 passed, 4 skipped**, CPU,123.78s. Pretrained tests were not enabled.
- Eleven new tests cover exact B0/FiLM identity, stage-to-stage propagation,
  gradients through all four adapters and condition head, weighted loss,
  strict checkpoint loading and shared inference, source-depth preflight,
  ablation configs, source-only guards, actual tiny training/resume and actual
  fresh/resume notebook setup cells using source archives.
- Tiny shared-trainer QA:4synthetic Cityscapes-shaped train images,2val,
  64×96canvas,2batches/epoch,2epochs,random init. Continuous and resumed
  final weights match exactly. This is not accuracy or convergence evidence.
- Offline smoke feeds K1 fog/night images from a procedural scene with known
  synthetic distances through K2; identity, gradients and checkpoint roundtrip pass.
- Node API:1passed. All4notebooks/code ZIP rebuilt and validated.

WeatherSeg-B0 has3,817,720parameters (98,693more than plain B0). GPU/CUDA AMP
and full Kaggle training have not been run locally. The notebook performs a
CUDA/AMP smoke before its main run. No real-weather classification accuracy,
segmentation improvement, full-run VRAM or runtime is claimed.

[Kaggle instructions](../../docs/K2_TRAINING.md) · [Notebook04](../../notebooks/04_k2_weatherseg_b0_train.ipynb)
