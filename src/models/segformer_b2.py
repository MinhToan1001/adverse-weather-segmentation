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
# Modified for Project-CV: explicit B2 variant, architecture checks and teaching notes.
"""SegFormer-B2: a separate, executable model definition.

Encoder: [64, 128, 320, 512] channels, [3, 4, 6, 3] Transformer blocks.
Decoder: project each stage to 768 channels, concatenate 3072, fuse to 768, classify 19.
For an H×W image, output logits are B×19×H/4×W/4 (divisible input sizes).
Attention/MLP/patch blocks are the pretrained-compatible Transformers implementation.
"""
from torch import nn
from transformers import SegformerConfig
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.models.segformer.modeling_segformer import (
    SegformerDecodeHead, SegformerModel, SegformerPreTrainedModel,
)


HIDDEN_SIZES = [64, 128, 320, 512]
DEPTHS = [3, 4, 6, 3]
DECODER_HIDDEN_SIZE = 768


def make_config(num_labels=19):
    """The architecture of B2 is declared here, independently of B0."""
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


class SegFormerB2(SegformerPreTrainedModel):
    """B2 encoder + B2 decoder, with its own forward and checkpoint entry point."""

    def __init__(self, config=None):
        config = make_config() if config is None else config
        if not matches_config(config):
            raise ValueError("SegFormerB2 requires the B2 encoder/depth/head configuration")
        super().__init__(config)
        self.segformer = SegformerModel(config)          # Four encoder stages: 64/128/320/512.
        self.decode_head = SegformerDecodeHead(config)  # Four 768-channel projections.
        self.post_init()

    def forward(self, pixel_values, labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        """RGB → B2 features → fused 768-channel decoder → class logits / CE loss."""
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        include_features = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        if labels is not None and self.config.num_labels < 1:
            raise ValueError("Number of labels must be positive")

        # 1. Wider/deeper encoder; the four spatial scales stay the same as B0.
        encoded = self.segformer(pixel_values, output_attentions=output_attentions,
                                 output_hidden_states=True, return_dict=return_dict)
        features = encoded.hidden_states if return_dict else encoded[1]

        # 2. Project each stage to 768, concatenate 3072, fuse to 768, classify.
        logits = self.decode_head(features)

        # 3. Same supervised label contract; optimization remains in training/.
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
    """No checkpoint: random B2; checkpoint supplied: load compatible pretrained B2."""
    return SegFormerB2(make_config(**options)) if checkpoint is None else SegFormerB2.from_pretrained(checkpoint, **options)
