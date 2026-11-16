# Separate model implementations

| Model | File | Parameters |
| --- | --- | ---: |
|B0 baseline|[segformer_b0.py](segformer_b0.py)|3,719,027|
|B2|[segformer_b2.py](segformer_b2.py)|27,361,235|
|DeepLabV3+R50|[deeplabv3plus_resnet50.py](deeplabv3plus_resnet50.py)|26,695,619|

registry.py selects a model,verifies pinned ImageNet initialization and strictly loads project checkpoints. segformer.py contains shared SegFormer loading/custom-size QA helpers. Architecture stays separate from loss,data,optimizer and evaluation. Every supported model returns19-class pixel logits;training policies are shared.

B0/B2 use Transformers blocks and have separate class/config/forward definitions. DeepLab uses TorchVision ResNet50 plus project separable ASPP/decoder,output stride16. This is not a claim of inventing the architectures or exactly reproducing their paper recipes. See root THIRD_PARTY_NOTICES.md and docs/PRESENT_TO_LECTURER.md.
