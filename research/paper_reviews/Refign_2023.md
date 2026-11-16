# Refign paper review

Reviewed 2026-11-06 against Brüggemann et al., WACV 2023, and the authors' release.

## Data and protocol

Refign addresses semantic segmentation under adverse weather and lighting with
unsupervised domain adaptation (UDA). For Cityscapes → ACDC, labeled Cityscapes
train images supervise the source task. ACDC train supplies 1,600 unlabeled
adverse images and corresponding clear reference frames. The paper reports 406
validation images and 2,000 test images. Table 1 reports ACDC **test** mIoU over
19 classes. The published DAFormer config sets ignore index 255. [Paper, §3–4 and Table 1](https://openaccess.thecvf.com/content/WACV2023/papers/Bruggemann_Refign_Align_and_Refine_for_Adaptation_of_Semantic_Segmentation_to_WACV_2023_paper.pdf) · [DAFormer config](https://github.com/brdav/refign/blob/main/configs/cityscapes_acdc/refign_daformer.yaml)

## Architecture and objectives

The segmentation variants in the paper build on DACS with DeepLabv2 or
DAFormer with an ImageNet-pretrained MiT-B5 encoder. An EMA teacher predicts
class probabilities for the adverse and reference frames. UAWarpC, with a
VGG-16 encoder and GLU-Net-style decoder, estimates dense flow and uncertainty.
It warps the reference probabilities into target coordinates. A non-parametric
fusion weights reference evidence by alignment confidence and prediction
uncertainty, with a rule for large static classes. The teacher's fused outputs
supply pseudo-labels for self-training. Refign alternates target refinement with
self-training on reference images. Deployment uses the segmentation network
alone. [Paper, Algorithm 1 and §3](https://openaccess.thecvf.com/content/WACV2023/papers/Bruggemann_Refign_Align_and_Refine_for_Adaptation_of_Semantic_Segmentation_to_WACV_2023_paper.pdf)

The DAFormer implementation uses pixel-weighted cross-entropy for source
labels and pseudo-labeled training, with feature-distance regularization enabled
in its config. The alignment network trains separately on MegaDepth with direct
and composed warp consistency. Its probabilistic objective weights regression by
predicted variance and penalizes excess uncertainty. The implementation described
in the paper uses Huber residuals for robustness. The alignment network stays
frozen during segmentation adaptation. [Paper, §3.1 and Eq. 1–4](https://openaccess.thecvf.com/content/WACV2023/papers/Bruggemann_Refign_Align_and_Refine_for_Adaptation_of_Semantic_Segmentation_to_WACV_2023_paper.pdf) · [DAFormer config](https://github.com/brdav/refign/blob/main/configs/cityscapes_acdc/refign_daformer.yaml)

## Reported scores and relevance

| Evidence | Variant | Evaluation | mIoU |
|---|---|---|---:|
| WACV Table 1 | Refign-DACS | ACDC test | 48.0% |
| WACV Table 1 | Refign-DAFormer | ACDC test | 65.5% |
| Authors' repository | Refign-HRDA, Cityscapes-pretrained backbone | ACDC test | 72.1% |

The official repository lists configs and checkpoints and labels the source code
MIT. The HRDA entry belongs to a different variant from the paper's DAFormer
row. The scores above are verified against the cited sources. Checkpoint
execution and artifact-specific licensing were outside this review. [Paper, Table 1](https://openaccess.thecvf.com/content/WACV2023/papers/Bruggemann_Refign_Align_and_Refine_for_Adaptation_of_Semantic_Segmentation_to_WACV_2023_paper.pdf) · [Official results](https://github.com/brdav/refign#trained-models-and-results)

Refign is a UDA reference because its training needs target images and
geographically paired references. Motion, occlusion and viewpoint changes make
pairwise alignment unreliable for some pixels. Our baseline trains on Cityscapes
alone and evaluates ACDC val, so the different training access and test split
exclude these scores from the primary DG benchmark. The confidence-based
supervision idea can inform a separate source-only experiment using augmented
source views. Any design informed by our ACDC results follows the project's
exploratory rule.
