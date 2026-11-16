"""DeepLabV3+ with ResNet50, output stride16 and separable ASPP/decoder.

Project implementation of Chen et al., ECCV2018 (arXiv:1802.02611).
ResNet blocks/optional ImageNet weights are provided by torchvision (BSD-3).
This file contains the architecture only; losses and data policy are shared.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers.modeling_outputs import SemanticSegmenterOutput


@dataclass
class DeepLabConfig:
    model_type: str = 'project_deeplabv3plus_resnet50'
    num_labels: int = 19
    semantic_loss_ignore_index: int = 255
    output_stride: int = 16
    decoder_channels: int = 256
    low_level_channels: int = 48


def conv_bn_relu(in_channels, out_channels, kernel=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=dilation * (kernel // 2),
                  dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


def separable_conv(in_channels, out_channels, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, padding=dilation, dilation=dilation,
                  groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        conv_bn_relu(in_channels, out_channels),
    )


class AtrousSpatialPyramidPooling(nn.Module):
    """Parallel receptive fields plus global context, fused to256 channels."""

    def __init__(self, channels=256):
        super().__init__()
        self.branches = nn.ModuleList([conv_bn_relu(2048, channels)] + [
            separable_conv(2048, channels, dilation=rate) for rate in (6, 12, 18)])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv_bn_relu(2048, channels))
        self.fuse = nn.Sequential(conv_bn_relu(5 * channels, channels), nn.Dropout(.1))

    def forward(self, features):
        pooled = F.interpolate(self.pool(features), features.shape[-2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([branch(features) for branch in self.branches] + [pooled], dim=1))


class DeepLabV3PlusResNet50(nn.Module):
    """RGB → ResNet C1/C4 → ASPP + low-level fusion → stride4 class logits."""

    def __init__(self, config=None, imagenet_encoder=False):
        super().__init__()
        self.config = config or DeepLabConfig()
        if (self.config.model_type != DeepLabConfig.model_type or self.config.output_stride != 16
                or self.config.decoder_channels != 256 or self.config.low_level_channels != 48
                or self.config.num_labels != 19 or self.config.semantic_loss_ignore_index != 255):
            raise ValueError('Require the declared19-class DeepLabV3+ R50 architecture')
        weights = ResNet50_Weights.IMAGENET1K_V1 if imagenet_encoder else None
        self.encoder = resnet50(weights=weights, replace_stride_with_dilation=[False, False, True])
        self.encoder.fc = nn.Identity()
        self.encoder.avgpool = nn.Identity()
        self.aspp = AtrousSpatialPyramidPooling()
        self.low_projection = conv_bn_relu(256, 48)
        self.decode_head = nn.Sequential(separable_conv(256 + 48, 256),
                                         separable_conv(256, 256), nn.Conv2d(256, 19, 1))

    def forward(self, pixel_values, labels=None):
        encoder = self.encoder
        x = encoder.maxpool(encoder.relu(encoder.bn1(encoder.conv1(pixel_values))))
        low = encoder.layer1(x)
        high = encoder.layer4(encoder.layer3(encoder.layer2(low)))
        context = F.interpolate(self.aspp(high), low.shape[-2:], mode='bilinear', align_corners=False)
        logits = self.decode_head(torch.cat([context, self.low_projection(low)], dim=1))
        loss = None
        if labels is not None:
            # Compatibility only. Research training computes the selected loss outside the model.
            values = F.interpolate(logits, labels.shape[-2:], mode='bilinear', align_corners=False)
            loss = F.cross_entropy(values, labels, ignore_index=255)
        return SemanticSegmenterOutput(loss=loss, logits=logits)

    def is_head_parameter(self, name):
        return not name.startswith('encoder.')

    def save_pretrained(self, directory, **options):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'config.json').write_text(json.dumps(asdict(self.config), indent=2) + '\n', encoding='utf-8')
        save_file({key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()},
                  str(directory / 'model.safetensors'))

    @classmethod
    def from_pretrained(cls, directory, **options):
        directory = Path(directory)
        config = DeepLabConfig(**json.loads((directory / 'config.json').read_text()))
        model = cls(config, imagenet_encoder=False)
        model.load_state_dict(load_file(str(directory / 'model.safetensors')), strict=True)
        return model
