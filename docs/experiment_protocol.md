# Experiment protocol v1 — Domain Generalization, Cityscapes → ACDC

| Field | Value |
|---|---|
| Status | **Draft for Gate A review** (frozen as v1.0 only after @MinhToan1001 approves) |
| Owner / author | @khangcules10 |
| Approver | @MinhToan1001 |
| Version / date | 1.0-draft / 2026-10-26 |
| Applies to | Phase 1 baseline (SegFormer-B0) and Phase 2 controlled extension |
| Related | `docs/DATA_CONTRACT.md`, `docs/RISKS.md`, `docs/ROADMAP_30_DAYS.md`, `docs/issues_backlog.csv` |

## Tóm tắt

Mục tiêu của giai đoạn này là đo mức suy giảm độ chính xác của một mô hình phân đoạn ngữ nghĩa 19 lớp khi chuyển từ điều kiện thời tiết bình thường sang thời tiết xấu, và kiểm chứng xem **một** cải tiến nhỏ có cải thiện khả năng tổng quát hóa hay không. Phạm vi là SegFormer-B0 huấn luyện **chỉ trên Cityscapes train**, chọn checkpoint **chỉ bằng Cityscapes val**, và đánh giá **một lần duy nhất** trên ACDC val (rain/fog/snow/night) như tập đích chưa từng thấy. Câu hỏi nghiên cứu: (RQ1) mô hình mất bao nhiêu mIoU khi sang ACDC; (RQ2) điều kiện nào và lớp nào suy giảm nặng nhất; (RQ3) một module cải tiến duy nhất có tăng mIoU trung bình trên ACDC mà không giảm mIoU Cityscapes val hay không. Ngoài phạm vi: domain adaptation có nhãn ACDC, thay đổi kiến trúc backbone, test-set tuning và mọi so sánh khác label map/split.

## 1. Objective, scope and research questions

**Objective.** Establish a reproducible, leakage-free measurement of cross-domain generalization for 19-class road-scene semantic segmentation: train on clear-weather Cityscapes, select the model on Cityscapes validation, then report performance on the adverse-weather ACDC validation set. The same frozen protocol is the reference for exactly one controlled extension in Phase 2.

**Research questions.**

- **RQ1 — Generalization gap.** How large is the drop in 19-class mIoU when a Cityscapes-trained SegFormer-B0 is evaluated on ACDC rain, fog, snow and night, relative to Cityscapes val?
- **RQ2 — Failure structure.** Which conditions and which classes dominate the degradation, and what visual/semantic failure modes explain it?
- **RQ3 — Mitigation (Phase 2 only).** Does one pre-registered modification (e.g. weather-robust augmentation or test-time normalization) improve mean ACDC mIoU over the frozen baseline without materially reducing Cityscapes val mIoU, under the same budget and seeds?

**In scope**

- 19-class Cityscapes label space (`0..18`, `255` = ignore) shared with ACDC.
- SegFormer-B0 (`nvidia/mit-b0` encoder, 19-class decoder head).
- Cityscapes train → Cityscapes val → ACDC val evaluation, single-scale, no TTA in the primary result.
- One controlled extension with an ablation table, fixed seeds and identical evaluation code.

**Out of scope (Phase 1)**

- Any use of ACDC labels (train or val) for training, checkpoint selection, hyper-parameter search, thresholding or architecture choice.
- Unsupervised/weakly supervised domain adaptation, ACDC train-set fine-tuning, Cityscapes test-set access.
- Backbone scaling beyond B0, multi-model ensembles, test-time augmentation as a headline result.
- Claims that compare numbers produced with different label maps, splits, image sizes or metric code.

## 2. Domain generalization protocol

**Definition used here.** Domain generalization (DG) means: the model never sees the target domain — no ACDC image, label, statistic or feedback of any kind — during training, model selection or tuning. ACDC val is a **held-out target domain** evaluated once, after the model is frozen. This is *not* domain adaptation, and results must not be described as adaptation.

| Rule | Requirement |
|---|---|
| **DG-1 Source only** | Training uses Cityscapes `train` (2,975 images) only. No ACDC image, crop, pseudo-label, statistic or augmentation derived from ACDC. |
| **DG-2 Selection only on source val** | Checkpoint = epoch with best 19-class mIoU on Cityscapes `val` (500 images). Ties → earliest epoch. `last` and `best` are both kept, only `best` is reported. |
| **DG-3 Target evaluated once** | ACDC `val` is evaluated **exactly once**, with the frozen `best` checkpoint, using the committed evaluation script and config. That run is the primary DG result. |
| **DG-4 No target feedback** | No ACDC-driven change to LR, epochs, augmentation, threshold, post-processing, class weights or architecture. If any such change is made, it invalidates the one-shot run: restart from a fresh checkpoint and report every run. |
| **DG-5 Exploratory work is labelled** | Any per-condition inspection, error analysis or sanity check on ACDC val after the one-shot run is marked **exploratory** in the run card and never replaces the primary number. |
| **DG-6 Same yardstick** | Baseline and extension use the same manifest, image size, ignore index, metric implementation and checkpoint-selection rule. Differences in any of these must be reported as a protocol deviation, not as a method gain. |
| **DG-7 Hold-out hygiene** | ACDC `test` labels are withheld; they are never used. ACDC `train` (1,600 images) stays unused in Phase 1; using it later defines a separate adaptation protocol with its own label. |
| **DG-8 Deviations are recorded** | Every deviation, failed run and environment change is logged with `docs/EXPERIMENT_LOG_TEMPLATE.md` and summarised in the report. No silent reruns. |

### 2.1 Data

| Dataset | Split | Images | Role |
|---|---|---|---|
| Cityscapes | train | 2,975 | Training only |
| Cityscapes | val | 500 | Checkpoint selection and in-domain reference |
| Cityscapes | test | 1,525 | **Not used** (labels withheld) |
| ACDC | val | 406 (fog 100, night 106, rain 100, snow 100) | Target-domain evaluation, **one shot** |
| ACDC | train | 1,600 | Not used in Phase 1 (would be adaptation) |

- Labels: single-channel PNG in Cityscapes **trainId** space, values `0..18` and `255` only. RGB palettes, raw `labelIds` and ACDC colour masks are invalid targets (`docs/DATA_CONTRACT.md`).
- Manifests: `data/manifests/cityscapes_train.csv`, `cityscapes_val.csv`, `acdc_val.csv` with columns `image,mask,domain,split`. Paths are relative to the repository root.
- Versioning: record dataset release/version, download date, file count and manifest SHA-256 in the data card. A split is frozen once Gate B closes.
- ACDC ambiguity pixels are unlabeled in the official annotations; they are mapped to `255` and ignored, never filled by hand.
- Leakage check: image-path and content-hash disjointness between Cityscapes train/val and ACDC val, and no duplicate across splits.

### 2.2 Model and training

| Item | Value |
|---|---|
| Architecture | `SegformerForSemanticSegmentation`, MiT-B0 encoder, 19 labels, `ignore_index=255` |
| Initialisation | `nvidia/mit-b0` ImageNet weights (revision pinned in config/run card) |
| Input | 512×512 random crop (train); resize to 512×512 for evaluation |
| Augmentation | Random horizontal flip 0.5, random scale 0.5–2.0, random crop; colour jitter only if pre-registered |
| Optimiser | AdamW, lr 6e-5, weight decay 0.01, poly decay power 1.0, warm-up 1,500 iters |
| Batch / epochs | 4 (reduce or accumulate if VRAM < 12 GB), 30 epochs |
| Precision | AMP fp16/bf16 if stable; fall back to fp32 and record it |
| Seed | 42 primary; 43/44 for the final baseline only if compute allows |
| Selection | Best Cityscapes val mIoU, evaluated every epoch |
| Budget | 1 GPU ≥ 8 GB; ≤ 20 GPU-hours per full run; smoke test ≤ 10 min CPU |

If data or compute does not arrive on schedule, the fallback is a documented subset run with the same protocol — never an invented metric (see `docs/RISKS.md`, R1/R2/R4).

### 2.3 Evaluation

**Primary metric.** 19-class mean IoU (mIoU) with `ignore_index=255`, computed from a single confusion matrix accumulated over the whole split. Classes with zero ground-truth pixels are excluded and the excluded list is reported.

**Required reporting.**

- Cityscapes val: overall mIoU + per-class IoU.
- ACDC val: mIoU for each condition (fog, night, rain, snow) and their unweighted mean; per-class IoU per condition.
- 95% bootstrap confidence interval over images (1,000 resamples) for each ACDC condition.
- Optional secondary: pixel accuracy (not a substitute for mIoU), parameters, inference latency and hardware.
- Exact command, config hash, checkpoint SHA-256 and commit hash for every reported number.

**Inference settings.** Single scale, no TTA, no per-condition checkpoint selection, no test-time threshold tuning. Logits are bilinearly upsampled to the label resolution before argmax.

**Artifacts.** `results/dg_eval/metrics.csv` (long format: `run_id,dataset,domain,class,iou,miou,n_images`), `results/tables/*.md`, `results/figures/*.png`. CSV is the source of truth; the report copies it.

## 3. Required deliverables

| ID | Deliverable | Location | Owner | Gate |
|---|---|---|---|---|
| D1 | Best Cityscapes-trained checkpoint + SHA-256 | local `outputs/`, release asset; **never committed** | A | C |
| D2 | Metrics CSV: Cityscapes val + ACDC per condition/class + CI | `results/baseline/`, `results/dg_eval/` | B | C, D |
| D3 | Ablation table: baseline vs one modification, same protocol | `results/ablation/` | A | D |
| D4 | Demo: CLI for image/folder → colour mask + overlay + metrics | `scripts/demo.py`, `results/demo/` | A + B | E |
| D5 | Report: method, data, protocol, results, failure analysis, limitations | `docs/report.md` | B (A reviews) | E |
| D6 | Slide deck (10–12 slides) for review | `docs/slides/` | B | E |
| D7 | Reproducibility pack: env, commands, hashes, run cards | `docs/reproducibility.md` | B | E |

Hard rule: checkpoints, raw datasets and external licences are **not** committed. Git tracks code, configs, docs, small CSVs and small figures only.

## 4. Repository structure

```text
.
├── .github/            CI + PR template
├── configs/            frozen training/eval configs (smoke + Cityscapes baseline)
├── data/
│   ├── manifests/      committed CSV manifests (no images)
│   └── raw/            local only, git-ignored
├── docs/               protocol, data contract, risks, ownership, roadmap, report
├── outputs/            local run outputs, git-ignored
├── research/           literature matrix, paper reviews, proposal, gap analysis
├── results/            committed metrics/tables/figures/demo artifacts
│   ├── baseline/
│   ├── dg_eval/
│   ├── ablation/
│   ├── tables/
│   ├── figures/
│   └── demo/
├── scripts/            data prep, validation, demo, GitHub automation
├── src/                data / models / training / evaluation packages
└── tests/              unit tests (metrics, data contract)
```

`results/README.md` defines what may be committed; large files go to a GitHub release and are referenced by hash.

## 5. Gates A–E

Each gate is a GitHub milestone; a gate closes only when its exit criteria are checked by the approver. No work from the next gate is merged before the current gate closes, except explicitly labelled spike work.

| Gate | Name | Exit criteria | Due | Sign-off |
|---|---|---|---|---|
| **A** | Scope & protocol frozen | Protocol approved; repo skeleton + README; 40 issues with owner/label/milestone; risk register with mitigations | 2026-10-28 | @MinhToan1001 |
| **B** | Data contract verified | Licences documented; Cityscapes + ACDC manifests validated (`--check-files --check-labels`); 20 images/domain audit; class frequencies; leakage check clean | 2026-10-31 | @MinhToan1001 |
| **C** | Reproducible baseline | Smoke test green; config frozen; full Cityscapes run reproduced twice; checkpoint + metrics CSV + run card; CI green | 2026-11-08 | @MinhToan1001 |
| **D** | DG evaluation + ablation | One-shot ACDC eval of frozen checkpoint; per-domain/per-class table with CI; failure analysis; one approved modification + controlled ablation | 2026-11-16 | @MinhToan1001 |
| **E** | Deliverables & handoff | Demo, report, slides, README/results index, release tag v0.1 + artifact manifest; reproducibility check from a clean clone | 2026-11-24 | @MinhToan1001 |

## 6. Reproducibility and logging

- Every run gets a run card from `docs/EXPERIMENT_LOG_TEMPLATE.md`: commit, config diff, data version/hash, seed, environment, metrics, failures.
- Pin dependency versions after the smoke test (lockfile or exact pins); record CUDA, driver, GPU and PyTorch versions.
- Log both `last` and `best` checkpoints; report only `best` for DG numbers.
- CI must pass (`python -m pytest -q`) before merge. Numbers enter the report only from committed CSVs produced by committed code.
- Determinism: fix Python/NumPy/PyTorch seeds; enable deterministic algorithms where they do not break throughput; report residual nondeterminism.

## 7. Change control

1. This file is versioned. Any change after Gate A requires a PR, an entry in the table below and approval from @MinhToan1001.
2. A protocol change made after seeing ACDC results is a new experiment, not an improvement to the old one.
3. If a primary run is invalidated, keep its artifacts, mark it `invalid` in `results/`, and rerun from scratch.

| Version | Date | Change | Reason | Approved by |
|---|---|---|---|---|
| 1.0-draft | 2026-10-26 | Initial DG protocol, deliverables, gates | Gate A deliverable | pending |

## 8. Acceptance checklist (Gate A)

- [x] Objective, scope and RQs written (Section 1)
- [x] DG protocol fixed: Cityscapes train → Cityscapes val selection → one-shot ACDC val (Section 2)
- [x] Required outputs listed (Section 3)
- [x] Repository structure covers `src/ configs/ scripts/ tests/ docs/ research/ results/` (Section 4)
- [x] 40 tasks mapped to Gates A–E, owners and milestones (GitHub issues + `docs/issues_backlog.csv`)
- [x] Risk register with at least one mitigation per risk (`docs/RISKS.md`)
- [ ] Reviewer approval by @MinhToan1001 (Gate A sign-off)

## References

- ACDC, ICCV 2021 / extended 2024: <https://acdc.vision.ee.ethz.ch/>, splits: 1,600 train / 406 val / 2,000 test adverse images.
- Cityscapes: <https://www.cityscapes-dataset.com/>
- SegFormer: <https://huggingface.co/docs/transformers/model_doc/segformer>
