# SegFormer — Cityscapes source baseline

Verified 2026-11-19 against the NeurIPS proceedings, paper Table 1(a), section 4.1 and the authors' implementation.

**Paper:** Enze Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.

- [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html)
- [Paper, Table 1(a) and section 4.1](https://arxiv.org/html/2105.15203v3)
- [Official training/evaluation implementation and model downloads](https://github.com/NVlabs/SegFormer)
- [Published NVIDIA B2 Cityscapes checkpoint in Transformers format](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024)

## Published Cityscapes reference

| Model | Cityscapes val mIoU — single-scale | Cityscapes val mIoU — multi-scale |
| --- | ---: | ---: |
| SegFormer-B0 / MiT-B0 | 76.2% | 78.1% |
| SegFormer-B2 / MiT-B2 | 81.0% | 82.2% |

Single-scale evaluates one scale; multi-scale combines predictions at several scales. These are source-paper reference numbers, not measurements produced by this project, and are not ACDC scores.

The paper uses ImageNet-1K encoder pretraining followed by supervised segmentation training. Section 4.1 specifies 1024 × 1024 Cityscapes training crops and sliding-window evaluation. A 512 × 1024 fit-to-canvas research evaluator is a different recipe; do not claim it reproduces the paper's Cityscapes scores without checking the matching inference conditions and checkpoint.

## Role in the user's study

The agreed question is: **can supervised ACDC fine-tuning improve a Cityscapes-trained SegFormer on adverse conditions while retaining source-domain performance?** The source paper need not itself report ACDC results to serve as the model/source benchmark for this extension.

Use two clearly separated result tables:

1. **Cityscapes source reference:** report the paper's numbers as published, then actual Cityscapes measurements of the selected checkpoint under the documented evaluation recipe. Re-evaluate frozen adapted models on the same Cityscapes set to measure retention.
2. **ACDC extension:** evaluate the same published checkpoint before adaptation (S0), conventional supervised fine-tuning (FT20), and proposed improvements on the same held-out ACDC images. These ACDC numbers must all be measured by this project; Cityscapes reference numbers cannot populate this table.

ACDC train/fit labels supervise fine-tuning; internal dev selects checkpoints; official val/test measures performance. Do not fine-tune on the evaluation images. Comparing S0 with a supervised model measures the practical benefit of target supervision/adaptation. Comparing FT20 with a controlled improved arm is needed to attribute gains to the proposed strategy.

The existing B2 result of 63.02% is from the earlier project-trained source initialization and internal dev. It is not a published-checkpoint S0 or FT20 result. Ordinary fine-tuning, replay and augmentation are established techniques; research novelty requires a concrete, validated contribution beyond simply applying them.

See the [baseline protocol](../BASELINE_PROTOCOL.md). No new checkpoint evaluation or training was performed during this literature verification.
