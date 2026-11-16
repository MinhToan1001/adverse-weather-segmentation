# Copyright 2021 NVIDIA The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for Project-CV: explicit B0 variant, architecture checks and teaching notes.
"""SegFormer-B0: a separate, executable model definition.

Encoder: [32, 64, 160, 256] channels, [2, 2, 2, 2] Transformer blocks.
Decoder: project each stage to 256 channels, concatenate 1024, fuse to 256, classify 19.
For an H×W image, output logits are B×19×H/4×W/4 (divisible input sizes).
Attention/MLP/patch blocks are the pretrained-compatible Transformers implementation.
"""
from torch import nn
from transformers import SegformerConfig
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.models.segformer.modeling_segformer import (
    SegformerDecodeHead, SegformerModel, SegformerPreTrainedModel,
)


HIDDEN_SIZES = [32, 64, 160, 256]
DEPTHS = [2, 2, 2, 2]
DECODER_HIDDEN_SIZE = 256


def make_config(num_labels=19):
    """The architecture of B0 is declared here, independently of B2."""
    return SegformerConfig(
        num_labels=num_labels, semantic_loss_ignore_index=255,
        hidden_sizes=list(HIDDEN_SIZES), depths=list(DEPTHS),
        decoder_hidden_size=DECODER_HIDDEN_SIZE,
        num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1],
        patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
    )


def matches_config(config):
    return (list(config.hidden_sizes) == HIDDEN_SIZES and list(config.depths) == DEPTHS
            and config.decoder_hidden_size == DECODER_HIDDEN_SIZE
            and list(config.num_attention_heads) == [1, 2, 5, 8])


class SegFormerB0(SegformerPreTrainedModel):
    """B0 encoder + B0 decoder, with its own forward and checkpoint entry point."""

    def __init__(self, config=None):
        config = make_config() if config is None else config
        if not matches_config(config):
            raise ValueError("SegFormerB0 requires the B0 encoder/depth/head configuration")
        super().__init__(config)
        self.segformer = SegformerModel(config)       # Four encoder stages: 32/64/160/256.
        self.decode_head = SegformerDecodeHead(config)  # Four 256-channel projections.
        self.post_init()

    def forward(self, pixel_values, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        """RGB → B0 features → fused 256-channel decoder → class logits / CE loss."""
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        include_features = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        if labels is not None and self.config.num_labels < 1:
            raise ValueError("Number of labels must be positive")

        # 1. Features at four scales: H/4, H/8, H/16 and H/32.
        encoded = self.segformer(pixel_values, output_attentions=output_attentions,
                                 output_hidden_states=True, return_dict=return_dict)
        features = encoded.hidden_states if return_dict else encoded[1]

        # 2. Project → upsample → concatenate → fuse → 19-class classifier.
        logits = self.decode_head(features)

        # 3. Train loss at label resolution; the trainer owns optimization.
        loss = None
        if labels is not None:
            native = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            if self.config.num_labels > 1:
                loss = nn.CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)(native, labels)
            else:
                valid = ((labels >= 0) & (labels != self.config.semantic_loss_ignore_index)).float()
                loss = nn.BCEWithLogitsLoss(reduction="none")(native.squeeze(1), labels.float())
                loss = (loss * valid).mean()
        if not return_dict:
            extra = encoded[1:] if include_features else encoded[2:]
            result = (logits,) + extra
            return (loss,) + result if loss is not None else result
        return SemanticSegmenterOutput(loss=loss, logits=logits,
            hidden_states=encoded.hidden_states if include_features else None, attentions=encoded.attentions)


def build_model(checkpoint=None, **options):
    """No checkpoint: random B0; checkpoint supplied: load compatible pretrained B0."""
    return SegFormerB0(make_config(**options)) if checkpoint is None else SegFormerB0.from_pretrained(checkpoint, **options)
