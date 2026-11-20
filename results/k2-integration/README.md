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

## K2 ablation delivery · 2026-11-20

- Full Python suite: **212 passed, 4 skipped**, CPU,137.06s; Node API: **1 passed**.
- Five arms now include `film_clean`, with no K1 or constant-clean auxiliary loss.
  The four existing arm recipes and core model/trainer/data code are unchanged.
- Actual fresh and resume notebook setup cells pass using uploaded source bundles;
  code cells parse and contain no stored execution outputs. Default is one
  `baseline` run, not an automatic five-run queue. No K3/K4 is added.
- Epoch subprocess heartbeats preserve trainer/RNG behavior and surface child
  failures. Review packaging errors do not block the full failure/resume archive.
- Artifact tests verify complete/partial status, candidate/checkpoint hashes,
  ZIP/folder inputs, lightweight exports, actual common initialization tensors,
  and rejection of changed recipes, core code, initialization or incomplete arms.
  Kaggle POSIX paths work in Windows audits; original snapshot hashes are checked
  before normalizing CRLF/LF for source comparisons.
- The user's original full Kaggle `k2` seed42 ZIP was audited:20contiguous epochs,
  best epoch20,60.6791% Cityscapes val500 mIoU; frozen config/manifests/candidate
  seals and common B0 initialization were checked. Its normalized core source
  and recipe match this revision. Reusing it with future controls additionally
  requires the comparison audit to match those controls' initial tensors/data.
- A real small ZIP was exported from that pilot: approximately13.7MiB, best
  inference weights + metadata/logs; the original full ZIP was preserved.

[GPU pilot record](../k2-kaggle-seed42/summary.json). No additional full GPU arm,
ACDC evaluation, multi-seed stability or segmentation improvement is established.
