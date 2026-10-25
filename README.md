# Adverse Weather Semantic Segmentation | NCKH CV

**Goal:** Robust 19-class road-scene semantic segmentation across clear / rain / fog / snow / night domains. Phase 1: reproducible **SegFormer-B0 baseline**; Phase 2: research extensions after a baseline review.

> Status: repository scaffold and executable **synthetic smoke test**, not a trained baseline or scientific results.

## Teams & ownership
- **Lead / Person A:** GitHub administrator, methodology and dataset split approvals, model/training code, baseline run, final acceptance.
- **Person B:** official dataset acquisition, image/label mapping and audits, EDA, metrics, experimental tables, reproducibility checks.
- No modification to `main` without an approved Pull Request by A. See `docs/OWNERSHIP.md`, `docs/ROADMAP_30_DAYS.md`.

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

## Real baseline (after dataset/license approval)
1. Download official datasets separately. Raw images/labels/checkpoints are **not committed**.
2. Make CSV manifests with `image,mask,domain,split` relative to repository root or absolute paths; mask MUST contain contiguous Cityscapes train IDs `0..18`, and `255` for void. **Do not pass raw labelIds**.
3. Set `configs/segformer_b0_cityscapes.yaml`: manifest paths, GPU batch size, epochs, pretrained checkpoint. Dataset-specific remapping for ACDC/Cityscapes must be audited.
4. Run `python scripts/validate_manifest.py --manifest ... --check-files --check-labels`.
5. Train and evaluate; report seed, device, dataset version, classes, split, mIoU per domain. Document any deviation from the original paper.

Train a new 19-class head starting from an encoder pretrained on ImageNet: `nvidia/mit-b0` (SegFormer family). The Hugging Face implementation supports `SegformerForSemanticSegmentation` and `ignore_index=255`. For serious experiments, use a validated CUDA PyTorch installation and pin dependencies after smoke.

## References
- https://huggingface.co/docs/transformers/model_doc/segformer
- https://www.cityscapes-dataset.com/
- https://acdc.vision.ee.ethz.ch/

## Research integrity
No test-set tuning, no fabricated results, no comparison across different label maps/splits. Record failed experiments. The enhancement module remains a proposal until literature review and ablation.
