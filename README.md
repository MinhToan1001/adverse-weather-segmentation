# Adverse Weather Semantic Segmentation | NCKH CV

**Goal:** Robust 19-class road-scene semantic segmentation across clear / rain / fog / snow / night domains. Phase 1: reproducible **SegFormer-B0 baseline**; Phase 2: research extensions after a baseline review.

> Status: repository scaffold, frozen experiment protocol (pending Gate A approval) and executable **synthetic smoke test**. Not a trained baseline and not scientific results yet.

## Protocol at a glance

- **Train** on Cityscapes `train` only. **Select the checkpoint** on Cityscapes `val`. The baseline gets one ACDC `val` pass after it is frozen. If ACDC results inform the proposed method, report that extension's ACDC score as exploratory, not confirmatory DG evidence.
- No ACDC image, label or statistic may influence training, checkpoint selection, hyper-parameters or post-processing. This is **domain generalization**, not domain adaptation.
- Full rules, metrics and gates: [`docs/experiment_protocol.md`](docs/experiment_protocol.md). Risks: [`docs/RISKS.md`](docs/RISKS.md). Data contract: [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md).

| Gate | Milestone | Due | Exit evidence |
|---|---|---|---|
| **A** | Scope & protocol frozen | 2026-10-28 | Protocol approved, repo skeleton, 40 issues, risk register |
| **B** | Data contract verified | 2026-10-31 | Licences, source data audit, ACDC file-level manifest QA without viewing target pixels |
| **C** | Reproducible baseline | 2026-11-08 | Smoke + full Cityscapes run reproduced twice, checkpoint + metrics CSV |
| **D** | DG evaluation + ablation | 2026-11-16 | One-shot ACDC val, per-domain table + CI, failure analysis, controlled ablation |
| **E** | Deliverables & handoff | 2026-11-24 | Demo, report, slides, results index, release v0.1 |

Task tracking: GitHub Issues (40 tasks) and the versioned backlog [`docs/issues_backlog.csv`](docs/issues_backlog.csv).

## Teams & ownership

- **Lead / Person A (@khangcules10):** GitHub administrator, protocol and split decisions, model/training code, baseline run, proposed method, PR review and final acceptance.
- **Person B (@MinhToan1001):** dataset terms and preparation, label mapping and audits, manifests, metrics, per-weather evaluation, literature matrix, plots and demo support.
- No modification to `main` without an approved Pull Request by A. See `docs/OWNERSHIP.md`, `docs/ROADMAP_30_DAYS.md`.

## Repository map

```text
├── .github/        CI workflow + PR template
├── configs/        frozen experiment configs (smoke, Cityscapes baseline)
├── data/
│   ├── manifests/  committed CSV manifests (no images)
│   └── raw/        local datasets only, git-ignored
├── docs/           protocol, data contract, risks, ownership, roadmap, report
├── outputs/        local run outputs, git-ignored
├── research/       literature matrix, paper reviews, proposal, gap analysis
├── results/        committed metrics, tables, figures, demo artifacts
├── scripts/        data prep, validation, demo, GitHub automation
├── src/            data / models / training / evaluation packages
└── tests/          unit tests
```

`results/` is the only place for committed experiment evidence; see [`results/README.md`](results/README.md) for naming and size rules.

## Required deliverables

| # | Artifact | Where |
|---|---|---|
| 1 | Best Cityscapes-trained checkpoint + SHA-256 | local `outputs/` + release asset (never committed) |
| 2 | Metrics CSV: Cityscapes val and ACDC per condition/class + bootstrap CI | `results/baseline/`, `results/dg_eval/` |
| 3 | Ablation table (baseline vs one modification, same protocol) | `results/ablation/` |
| 4 | Demo CLI: image/folder → colour mask + overlay | `scripts/demo.py`, `results/demo/` |
| 5 | Report (method, data, metrics, results, limits) | `docs/report.md` |
| 6 | Slide deck | `docs/slides/` |
| 7 | Reproducibility pack (env, commands, hashes, run cards) | `docs/reproducibility.md` |

## Quick start (Windows PowerShell or Linux)

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python scripts/make_synthetic_dataset.py
python scripts/validate_manifest.py --manifest data/manifests/synthetic_train.csv
python -m src.training.train --config configs/segformer_b0_smoke.yaml
python -m src.evaluation.evaluate --config configs/segformer_b0_smoke.yaml --checkpoint outputs/smoke/best
```

Smoke mode uses an **offline random initialization**, small synthetic segmentation images, 2 training batches, **no model downloads**. It validates only the pipeline, not model quality. Training smoke takes CPU time.

## Real baseline (after dataset/licence approval)

1. Download official datasets separately. Raw images/labels/checkpoints are **not committed**; follow `docs/RISKS.md` R2/R3.
2. Make CSV manifests with `image,mask,domain,split` relative to repository root or absolute paths; mask MUST be a single-channel integer image with contiguous Cityscapes train IDs `0..18`, and `255` for void. **Do not pass RGB/palette masks or raw labelIds**.
3. Set `configs/segformer_b0_cityscapes.yaml`: manifest paths, GPU batch size, epochs, pretrained checkpoint. Dataset-specific remapping for ACDC/Cityscapes must be audited.
4. Run `python scripts/validate_manifest.py --manifest ... --check-files --check-labels`.
5. Train and select the checkpoint on Cityscapes val. The config uses an aspect-preserving 512×1024 padded input, batch size 1 with gradient accumulation 4, poly learning-rate decay, warm-up and CUDA AMP. Resume an interrupted run with `python -m src.training.train --config configs/segformer_b0_cityscapes.yaml --resume outputs/segformer_b0_cityscapes/last`.
6. After freezing the checkpoint and config, evaluate the ACDC validation manifest in one inference pass: `python -m src.evaluation.evaluate --config configs/segformer_b0_cityscapes.yaml --checkpoint outputs/segformer_b0_cityscapes/best --manifest data/manifests/acdc_val.csv --run-id b0-seed42 --result-status confirmatory --output-csv results/dg_eval/metrics.csv --output-json results/dg_eval/metrics.json`. The evaluator requires a clean committed code tree.
7. Report seed, device, dataset version, classes, split, mIoU by weather and per class. The bootstrap interval resamples cached per-image confusion matrices and does not rerun inference.

Train a new 19-class head starting from an encoder pretrained on ImageNet: `nvidia/mit-b0` (SegFormer family). The Hugging Face implementation supports `SegformerForSemanticSegmentation` and `ignore_index=255`. For serious experiments, use a validated CUDA PyTorch installation and pin dependencies after smoke.

## Evaluation protocol (summary)

- Primary metric: 19-class mIoU, `255` ignored; per-class IoU and per-condition breakdown are mandatory.
- ACDC val = 406 images (fog 100, night 106, rain 100, snow 100); before the baseline is frozen, verify only file-level metadata and pairing. Do not rerun the baseline for tuning. A target-informed proposed model may be evaluated once but its result is exploratory.
- Report bootstrap 95% CI over images, result status, exact command, manifest/config hashes, checkpoint SHA-256 and commit hash.
- Test-set tuning, per-condition checkpoint selection and TTA in the primary result are prohibited.

## References

- https://huggingface.co/docs/transformers/model_doc/segformer
- https://www.cityscapes-dataset.com/
- https://acdc.vision.ee.ethz.ch/

## Research integrity

No test-set tuning, no fabricated results, no comparison across different label maps/splits. Record failed experiments. The enhancement module remains a proposal until literature review and ablation.
