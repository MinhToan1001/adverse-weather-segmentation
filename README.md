# Adverse Weather Semantic Segmentation | NCKH CV

**Goal:** Robust 19-class road-scene semantic segmentation across clear / rain / fog / snow / night domains. Phase 1: reproducible **SegFormer-B0 baseline**; Phase 2: research extensions after a baseline review.

> Status: repository scaffold, frozen experiment protocol (pending Gate A approval) and executable **synthetic smoke test**. Not a trained baseline and not scientific results yet.

## Protocol at a glance

- **Train** on Cityscapes `train` only. **Select the checkpoint** on Cityscapes `val`. **Evaluate once** on ACDC `val` (rain/fog/snow/night) as an unseen target domain.
- No ACDC image, label or statistic may influence training, checkpoint selection, hyper-parameters or post-processing. This is **domain generalization**, not domain adaptation.
- Full rules, metrics and gates: [`docs/experiment_protocol.md`](docs/experiment_protocol.md). Risks: [`docs/RISKS.md`](docs/RISKS.md). Data contract: [`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md).

| Gate | Milestone | Due | Exit evidence |
|---|---|---|---|
| **A** | Scope & protocol frozen | 2026-10-28 | Protocol approved, repo skeleton, 40 issues, risk register |
| **B** | Data contract verified | 2026-10-31 | Licences, manifests validated, 20-image/domain audit, leakage check |
| **C** | Reproducible baseline | 2026-11-08 | Smoke + full Cityscapes run reproduced twice, checkpoint + metrics CSV |
| **D** | DG evaluation + ablation | 2026-11-16 | One-shot ACDC val, per-domain table + CI, failure analysis, controlled ablation |
| **E** | Deliverables & handoff | 2026-11-24 | Demo, report, slides, results index, release v0.1 |

Task tracking: GitHub Issues (40 tasks) and the versioned backlog [`docs/issues_backlog.csv`](docs/issues_backlog.csv).

## Teams & ownership

- **Lead / Person A (@MinhToan1001):** GitHub administrator, methodology and dataset split approvals, model/training code, baseline run, final acceptance.
- **Person B (@khangcules10):** official dataset acquisition, image/label mapping and audits, EDA, metrics, experimental tables, reproducibility checks.
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
2. Make CSV manifests with `image,mask,domain,split` relative to repository root or absolute paths; mask MUST contain contiguous Cityscapes train IDs `0..18`, and `255` for void. **Do not pass raw labelIds**.
3. Set `configs/segformer_b0_cityscapes.yaml`: manifest paths, GPU batch size, epochs, pretrained checkpoint. Dataset-specific remapping for ACDC/Cityscapes must be audited.
4. Run `python scripts/validate_manifest.py --manifest ... --check-files --check-labels`.
5. Train and evaluate; report seed, device, dataset version, classes, split, mIoU per domain. Document any deviation from the original paper.

Train a new 19-class head starting from an encoder pretrained on ImageNet: `nvidia/mit-b0` (SegFormer family). The Hugging Face implementation supports `SegformerForSemanticSegmentation` and `ignore_index=255`. For serious experiments, use a validated CUDA PyTorch installation and pin dependencies after smoke.

## Evaluation protocol (summary)

- Primary metric: 19-class mIoU, `255` ignored; per-class IoU and per-condition breakdown are mandatory.
- ACDC val = 406 images (fog 100, night 106, rain 100, snow 100); evaluated **once** after the checkpoint is frozen.
- Report bootstrap 95% CI over images, exact command, config hash, checkpoint SHA-256 and commit hash.
- Test-set tuning, per-condition checkpoint selection and TTA in the primary result are prohibited.

## References

- https://huggingface.co/docs/transformers/model_doc/segformer
- https://www.cityscapes-dataset.com/
- https://acdc.vision.ee.ethz.ch/

## Research integrity

No test-set tuning, no fabricated results, no comparison across different label maps/splits. Record failed experiments. The enhancement module remains a proposal until literature review and ablation.
