# Local audit and offline progress

Date: 2026-11-01. Branch: khang. All changes are local, not committed/pushed.

The project has a verified synthetic training pipeline through Khang-task05;
there is no trained Cityscapes baseline. Notion status alone is not completion
proof. At the start of this audit, the paper CSV contained only a header and
paper reviews were empty templates; scheduler/resume existed without a run log.

| Tasks | Verified status after this audit |
|---|---|
| Khang/Toan 01 | Prior work exists; overall Gate A sign-off still pending in protocol |
| Khang/Toan 02 | Pretrained and metrics tests passed |
| Khang 03 | Complete: geometry, normalization and padding now explicit in both configs; config is consumed by training and evaluation |
| Toan 03 | Generator/validator tested; full real manifests pending downloads |
| Khang 04 | Complete for CPU: config snapshot, seed, scheduler, AMP fallback, checkpoints and epoch-boundary resume verified |
| Toan 04/05 | Guide/status ready; real dataset QA and loader acceptance blocked |
| Khang 05 | Synthetic smoke complete: decreasing loss, best/last and exact resume evidence |
| Khang 06 | Real-data warm-up and GPU budget not established; do not extrapolate tiny-model CPU timing |
| Toan 06 | First literature matrix complete for HRDA, Refign, VLTSeg; sources and limitations recorded |
| Toan 07 | Evaluator code already groups conditions; real run/resource monitoring pending |
| Khang 07/08; Toan 08 | Blocked by dataset and baseline checkpoint |
| Khang 09/10/11; Toan 09/10 | Paper feasibility/review/inference integration remains separate work; matrix is only initial screening |
| Khang 12-20; Toan 11-20 | Depend on failure analysis, approved method, models or results; not accepted early from scaffold code |

## Evidence

- `results/offline-audit-2026-11-01/tests.txt`: 79 unit/integration checks passed
  before the final Windows resume fix; the dedicated training audit below ran
  after that fix.
- `verification.json`: three epochs, full vs resumed model weights and history
  exactly equal; scheduler equal; changed resume horizon rejected; partial
  accumulation group tested. AMP requested but correctly disabled on CPU.
- `history.jsonl`: synthetic train loss 2.9760345 -> 2.8155369 -> 2.7751137.
  Scores in this history are synthetic fixture scores, not research metrics.
- Local smoke checkpoints and copied config:
  `outputs/offline-audit-2026-11-01-r02/{full,resumed}/{best,last}`.
  The two best model files have identical SHA-256 recorded in verification.json.
- First failed run/log retained: `training.txt`; corrected run: `training-r02.txt`.

## Fixed issues and limits

1. Windows resume originally failed with safetensors error 1224 when saving over
   a CPU memory-mapped checkpoint. Clone checkpoint tensors before constructing
   the optimizer; the corrected run passes exact resume parity.
2. Resume now requires the original config/scheduler horizon (output directory
   may differ). Old checkpoint states lacking config cannot be verified and are
   rejected. This is epoch-boundary resume, not recovery of an unsaved batch.
3. A fresh run refuses an existing history file. Non-finite loss fails clearly.
4. Scheduler/global optimizer step advances only if AMP did not skip the update.
5. GPU AMP execution is not verified on this CPU environment; no GPU-hours or
   full-run budget estimate is invented. Version ranges are not a lockfile.
6. Real manifests, QA, trained baseline, target evaluation, research proposal,
   ablation and demo results are not complete.

Reproduce in a NEW output directory:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.venv/Scripts/python.exe scripts/verify_offline_training.py --output outputs/offline-audit-repeat
```

The script uses only generated fixture images, with a small randomly initialized
SegFormer from the smoke config. It never accesses raw Cityscapes or ACDC.

Additional independent deliverable: `research/BENCHMARK_FEASIBILITY.md` completes
Khang-task09's screening/decision note. Execution remains blocked by a separate
CUDA stack, checkpoint audit and data; it is not a completed benchmark run.
