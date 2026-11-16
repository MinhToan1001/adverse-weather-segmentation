# Third-party notices

The explicit SegFormer variant modules `src/models/segformer_b0.py` and
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
(https://arxiv.org/abs/1606.04797), with the exact project reductions documented
in docs/RESEARCH_FRAMEWORK.md. They are not claims of new loss functions.
