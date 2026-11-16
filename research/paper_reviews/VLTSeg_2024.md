# VLTSeg paper review

Reviewed 2026-11-06 against Hümmer et al., ACCV 2024, the supplement and the
public reimplementation. The paper calls the segmentation method VLTSeg.

## Data and protocol

This is domain generalization (DG). Downstream supervised fine-tuning uses a
labeled source split. In the real-to-real Cityscapes experiment, the source is
Cityscapes train and the targets include ACDC, BDD100K and Mapillary val. The
paper also studies GTA5 and SYNTHIA as separate source settings. The ACDC test
submission trains for 20,000 iterations on 1,024 × 1,024 crops from Cityscapes
train without ACDC data. It evaluates native 1,920 × 1,080 images with multiscale
inference at four scale ratios (1.0, 1.25, 1.5, 1.75) and the test augmentation
procedure in supplement §6.2. The Cityscapes test submission instead uses a
Mapillary pretraining stage followed by Cityscapes fine-tuning. [ACCV paper, §5.4 and Table 6](https://openaccess.thecvf.com/content/ACCV2024/html/Hummer_Strong_but_simple_A_Baseline_for_Domain_Generalized_Dense_Perception_ACCV_2024_paper.html) · [Supplement, §6](https://openaccess.thecvf.com/content/ACCV2024/supplemental/Hummer_Strong_but_simple_ACCV_2024_supplemental.pdf)

## Architecture and objectives

VLTSeg transfers vision-language pretraining into a supervised segmenter. The
main high-capacity variant uses an EVA-02-CLIP ViT-L/14 vision encoder,
intermediate features projected for the Mask2Former decoder and a 19-class
head. The paper studies frozen encoders versus full fine-tuning. Full
fine-tuning updates encoder and decoder with AdamW using a lower encoder
learning rate. The paper uses the standard Mask2Former segmentation objective
without an extra DG loss. The public reimplementation specifies query
classification cross-entropy, sigmoid mask cross-entropy and Dice loss with
weights 2, 5 and 5, respectively, and Hungarian matching. Those exact weights
describe the public config. [ACCV paper, §3 and §5.5](https://openaccess.thecvf.com/content/ACCV2024/papers/Hummer_Strong_but_simple_A_Baseline_for_Domain_Generalized_Dense_Perception_ACCV_2024_paper.pdf) · [Supplement, Table 12](https://openaccess.thecvf.com/content/ACCV2024/supplemental/Hummer_Strong_but_simple_ACCV_2024_supplemental.pdf) · [Reimplementation model config](https://github.com/VLTSeg/VLTSeg/blob/main/configs/_base_/models/eva-clip%2Bmask2former.py)

## Reported scores

| Evidence | Source | Target split | mIoU |
|---|---|---|---:|
| ACCV 2024 Table 6 | Cityscapes train | ACDC val | 73.3% |
| ACCV 2024 abstract and supplement §6.2 | Cityscapes train | ACDC test, multiscale/TTA | 77.9% |
| arXiv:2312.02021v2 abstract, Dec 2023 | Cityscapes train | ACDC test | 76.48% |
| Reimplementation README, Cityscapes_1 | Cityscapes | ACDC val / test with TTA | 73.16% / 77.28% |
| Reimplementation README, Cityscapes_2 | Cityscapes | ACDC val / test with TTA | 73.92% / 77.11% |

The val and test entries in each row name separate splits. The December 2023
preprint and ACCV paper are different versions. The repository entries belong to
a third-party implementation and separate checkpoints. All scores in this table
are reported externally. [ACCV paper](https://openaccess.thecvf.com/content/ACCV2024/html/Hummer_Strong_but_simple_A_Baseline_for_Domain_Generalized_Dense_Perception_ACCV_2024_paper.html) · [arXiv v2](https://arxiv.org/abs/2312.02021v2) · [Checkpoint table](https://github.com/VLTSeg/VLTSeg#checkpoints)

## Reproduction limits

The project page links an unofficial MMSegmentation reimplementation. Code is
MIT and released checkpoints are CC BY-NC 4.0. The paper describes the
EVA-02-L/14 encoder as roughly 304 million parameters and notes the greater
memory and training cost. The release lists a CUDA 12.1 stack. GPU memory and
runtime for our configuration are unmeasured. Checkpoint source, class order,
ignore handling and inference settings need verification before evaluation.
Even on the same val split, architecture, pretraining, resolution and inference
differences limit what a score gap can establish. The separate feasibility note
records the checkpoint candidate and run conditions. [Authors' project](https://vltseg.github.io/) · [Reimplementation](https://github.com/VLTSeg/VLTSeg) · [Zenodo release](https://zenodo.org/records/14766160) · [ACCV paper, §5.5](https://openaccess.thecvf.com/content/ACCV2024/papers/Hummer_Strong_but_simple_A_Baseline_for_Domain_Generalized_Dense_Perception_ACCV_2024_paper.pdf)
