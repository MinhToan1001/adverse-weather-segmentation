# Third-party notices

The explicit SegFormer variant modules `src/models/legacy/segformer_b0.py` and
`src/models/segformer_b2.py` adapt the segmentation assembly/forward convention
from Hugging Face Transformers4.57.6, originally credited to NVIDIA and the
Hugging Face team. They retain the Apache2.0 header. The complete license is
included at `third_party/transformers/LICENSE`.

The variants reuse Transformers' encoder, decoder and pretrained-loading blocks;
they are not a claim that Project-CV invented SegFormer. Dataset and checkpoint
terms remain those of their respective providers. This notice does not assign
a license to unrelated project files.

`src/models/deeplabv3plus_resnet50.py` is a project implementation based on
Chen et al., "Encoder-Decoder with Atrous Separable Convolution for Semantic
Image Segmentation", ECCV2018, https://arxiv.org/abs/1802.02611.
It imports the ResNet50 backbone/optional ImageNet weights from TorchVision
(BSD-3-Clause; https://github.com/pytorch/vision/blob/main/LICENSE).
No segmentation_models.pytorch dependency or source is included.

Shared Focal and Dice objectives are project implementations of ideas from
Lin et al., ICCV2017 (https://arxiv.org/abs/1708.02002) and Milletari et al.,2016
(https://arxiv.org/abs/1606.04797), with the exact project reductions implemented
in src/training/losses.py. These optional historical objectives are not enabled
in the current DG recipe and are not claims of new loss functions.

K1 weather effects in `src/data/augmentations/weather/` are project-written
implementations of atmospheric scattering and shot/read-noise approximations,
plus heuristic 2D rain/snow overlays. Conceptual references, exact assumptions
and limitations are listed in `docs/K1_WEATHER.md`. No Foggy Cityscapes or
Unprocessing source code, dataset, calibration file or pretrained weight is
bundled. This is not a claim to reproduce either complete published pipeline.

`src/models/proposed/weatherseg_b0.py` adapts the SegFormer encoder traversal
from Hugging Face Transformers4.57.6 (NVIDIA / Hugging Face, Apache-2.0), with
stage-wise statistical conditioning and a project auxiliary head. The FiLM
conditioning idea is credited to Perez et al., AAAI2018,
https://arxiv.org/abs/1709.07871. No claim of inventing FiLM or reproducing its
original visual-reasoning experiments is made.
