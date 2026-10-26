# Risk register v1 — adverse-weather segmentation (Phase 1)

| Field | Value |
|---|---|
| Owner | @khangcules10 |
| Approver | @MinhToan1001 |
| Created / reviewed | 2026-10-26 / weekly at each gate |
| Scale | Probability (P) and Impact (I): L = low, M = medium, H = high |
| Related | `docs/experiment_protocol.md`, `docs/DATA_CONTRACT.md`, `docs/ROADMAP_30_DAYS.md` |

A risk is reviewed at every gate. If its trigger fires, the contingency is executed and the decision is logged in the gate issue. No schedule slip is accepted silently, and no result is fabricated to protect the deadline.

## Summary

| ID | Risk | P | I | Owner | Mitigation (at least one) | Trigger → contingency |
|---|---|---|---|---|---|---|
| **R1** | GPU unavailable, too slow, or VRAM < 8 GB | H | H | A | Book one GPU early; 512² / batch 4 with AMP; batch 2 + gradient accumulation; pre-agreed Kaggle/Colab or lab-server fallback; measure GPU-hours per epoch in the smoke run | No stable GPU by **2026-11-01** → run a documented 25% Cityscapes subset with the same protocol, label it `subset`, and keep full-run results out of claims |
| **R2** | Dataset access delayed (Cityscapes approval, ACDC download) | M | H | B | Register/download on day 1; verify links and disk space; keep checksums; start with synthetic smoke while waiting | Data missing by **2026-10-30** → run the full pipeline on synthetic + Cityscapes subset, freeze code/metrics, and mark DG numbers `pending`; never invent them |
| **R3** | Licence / terms violation (redistribution, committing data or weights) | L | H | B | Write `docs/DATASETS.md` with source, terms, citation and allowed use; keep raw data in git-ignored `data/`; check `.gitignore` before each push; no re-upload of images or labels; share checkpoints only as release assets | Any doubt about terms → stop distribution, keep artifacts local, ask the dataset owner, document the decision |
| **R4** | Schedule slip (30-day plan, 2 people, coursework) | H | M | A | Gate-based plan with hard dates; weekly 15-min sync; cut Phase 2 before cutting the Phase 1 baseline; 20% buffer before 2026-11-24 | A gate slips > 3 days → drop the lowest-value issue in that gate, record it in the gate issue, protect Gate C/D evidence |
| **R5** | Wrong label mapping (`labelIds` vs `trainId`, ACDC naming) | M | H | B | Converter + unit tests; validate every mask against `0..18 ∪ {255}`; 20 images/domain visual audit; `validate_manifest.py --check-labels` in CI | Audit finds mismatched colours/IDs → block the manifest, fix the converter, re-audit 40 images, re-freeze the split hash |
| **R6** | Data leakage / test-set contamination | M | H | B | Disjoint manifests by path and content hash; no ACDC in train/val; one-shot ACDC evaluation rule (DG-3/DG-4); log every ACDC evaluation invocation | Suspected leakage → freeze the run, mark it `invalid`, rebuild the manifest from official splits, rerun and report both runs |
| **R7** | Non-reproducible results (seeds, environment drift) | M | M | A | Pin dependencies; fix seeds; commit config + environment; deterministic flags where possible; independent rerun before Gate C closes | Rerun differs by > 0.5 mIoU → investigate data order/augmentation, report variance, do not cherry-pick the better run |
| **R8** | Pretrained weights unavailable (offline/blocked Hugging Face) | M | M | A | Pre-download `nvidia/mit-b0` and cache it; record revision; mirror on an approved drive | Weights still unavailable at Gate C → train from random init and label it `no-pretrain`; never compare it as if it were the pretrained baseline |
| **R9** | Storage / bandwidth limits (datasets, checkpoints, outputs) | M | L | B | Check free disk (≥ 100 GB); delete redundant archives after checksum; store checkpoints outside git; small artifacts only in `results/` | Disk < 20 GB free → move raw data to an external drive, keep manifests and hashes, document the new root path |
| **R10** | Personnel unavailability / bus factor (2-person team) | M | H | A | Every task has one owner + one reviewer; run cards and README allow the other person to reproduce; shared notes per gate; no undocumented local state | One person unavailable > 3 days → the other takes the gate's critical path, reviewer becomes author for that PR and the switch is recorded |
| **R11** | Metric implementation bug (mIoU, ignore 255) | M | H | B | Unit tests for perfect/worst/empty-class cases; compare against a second implementation on a fixed fixture; review the confusion matrix accumulation | Numbers look implausible → recompute from raw predictions with an independent script, fix tests first, then rerun |
| **R12** | Overclaiming / weak novelty in Phase 2 | M | M | A | One pre-registered modification; literature matrix with ≥ 5 papers; negative results reported; Gate D review before writing claims | No measurable gain → report the negative result and the failure analysis; do not reframe a failed ablation as a contribution |

## Standing controls

1. **Weekly risk review** at the gate sync: update P/I, close resolved risks, add new ones with an owner and a mitigation.
2. **No silent scope change**: any cut or protocol change goes through `docs/experiment_protocol.md` change control.
3. **Evidence over assertion**: every mitigation is verifiable (a file, a CI check, a log, a hash), not a promise.
4. **Escalation**: a trigger that blocks a gate is raised in the gate issue the same day and decided by @MinhToan1001 within 24 hours.
