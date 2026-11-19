# Copyright 2021 NVIDIA The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for Project-CV: stage FiLM, condition head and project outputs.
# Encoder traversal adapted from Hugging Face Transformers' SegFormer
# (NVIDIA / Hugging Face, Apache-2.0); see third_party/transformers/LICENSE.
"""K2: MiT-B0 with stage-wise self-conditioned FiLM and a condition head.

FiLM predicts delta-gamma/beta from spatial channel mean/variance. Zero final
MLP projections give exact identity at initialization. Modulation happens before
the next patch embedding, not just on the decoder's skip features.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput, SemanticSegmenterOutput
from transformers.models.segformer.modeling_segformer import (
    SegformerEncoder, SegformerModel, SegformerDecodeHead, SegformerPreTrainedModel,
)

from src.models.legacy.segformer_b0 import make_config as b0_config, matches_config
from src.data.augmentations.weather import CONDITIONS


MODEL_NAME = 'weatherseg_b0'


@dataclass
class WeatherSegOutput(SemanticSegmenterOutput):
    condition_logits: torch.Tensor | None = None
    segmentation_loss: torch.Tensor | None = None
    condition_loss: torch.Tensor | None = None


class StatisticsFiLM(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(nn.Linear(2 * channels, max(8, channels // reduction)),
                                 nn.GELU(), nn.Linear(max(8, channels // reduction), 2 * channels))
        self.reset_identity()

    def reset_identity(self):
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError('FiLM requires BxCxHxW features with the configured channels')
        # FP32 statistics avoid small spatial variances underflowing in AMP.
        variance, mean = torch.var_mean(features.float(), dim=(2, 3), correction=0)
        delta, beta = self.mlp(torch.cat((mean, variance), dim=1)).to(features.dtype).chunk(2, dim=1)
        return features * (1 + delta[:, :, None, None]) + beta[:, :, None, None]


class FiLMEncoder(SegformerEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.film = nn.ModuleList(StatisticsFiLM(c, config.film_reduction) for c in config.hidden_sizes)

    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False, return_dict=True):
        hidden_states = pixel_values
        features = () if output_hidden_states else None
        attentions = () if output_attentions else None
        batch_size = pixel_values.shape[0]
        for embedding, blocks, norm, film in zip(self.patch_embeddings, self.block, self.layer_norm, self.film):
            hidden_states, height, width = embedding(hidden_states)
            for block in blocks:
                outputs = block(hidden_states, height, width, output_attentions)
                hidden_states = outputs[0]
                if output_attentions:
                    attentions += (outputs[1],)
            hidden_states = norm(hidden_states)
            hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            hidden_states = film(hidden_states)
            if output_hidden_states:
                features += (hidden_states,)
        if not return_dict:
            return tuple(x for x in (hidden_states, features, attentions) if x is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=features, attentions=attentions)


def make_config(film_reduction=4, condition_loss_weight=.1):
    if isinstance(film_reduction, bool) or not isinstance(film_reduction, int) or film_reduction < 1:
        raise ValueError('film_reduction must be a positive integer')
    if not math.isfinite(condition_loss_weight) or condition_loss_weight < 0:
        raise ValueError('condition_loss_weight must be finite and nonnegative')
    config = b0_config()
    config.project_model = MODEL_NAME
    config.film_reduction = film_reduction
    config.condition_names = list(CONDITIONS)
    config.condition_loss_weight = float(condition_loss_weight)
    config.architectures = ['WeatherSegB0']
    return config


class WeatherSegB0(SegformerPreTrainedModel):
    def __init__(self, config=None):
        config = config or make_config()
        if (not matches_config(config) or not config.reshape_last_stage or config.num_labels != 19
                or config.semantic_loss_ignore_index != 255 or getattr(config, 'project_model', None) != MODEL_NAME):
            raise ValueError('K2 requires the marked 19-class MiT-B0 config with spatial stage outputs')
        if list(config.condition_names) != list(CONDITIONS):
            raise ValueError('K2 condition order must be clean/fog/night/rain/snow')
        make_config(config.film_reduction, config.condition_loss_weight)  # Validate saved settings too.
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.segformer.encoder = FiLMEncoder(config)
        self.decode_head = SegformerDecodeHead(config)
        self.condition_head = nn.Linear(config.hidden_sizes[-1], len(CONDITIONS))
        self.post_init()
        # post_init initializes Linear layers, so restore FiLM identity afterwards.
        for film in self.segformer.encoder.film:
            film.reset_identity()

    def is_head_parameter(self, name):
        return name.startswith(('decode_head.', 'condition_head.')) or '.film.' in name

    @classmethod
    def from_b0(cls, baseline, **options):
        """Copy all baseline weights (including head) for an exact identity control."""
        if (not matches_config(baseline.config) or baseline.config.num_labels != 19
                or baseline.config.semantic_loss_ignore_index != 255 or not baseline.config.reshape_last_stage):
            raise ValueError('Expected a complete 19-class B0 control')
        config = make_config(**options)
        config.id2label, config.label2id = baseline.config.id2label, baseline.config.label2id
        # Preserve dropout/normalization settings, not just the tensor shapes.
        for name in ('hidden_dropout_prob', 'attention_probs_dropout_prob', 'classifier_dropout_prob',
                     'drop_path_rate', 'layer_norm_eps', 'initializer_range', 'hidden_act'):
            setattr(config, name, getattr(baseline.config, name))
        # New parameters must not shift the global dropout/data RNG of a matched
        # plain-B0 control. All modules are constructed on CPU before .to(device).
        with torch.random.fork_rng(devices=[]):
            result = cls(config)
        info = result.load_state_dict(baseline.state_dict(), strict=False)
        if info.unexpected_keys or any('.film.' not in k and not k.startswith('condition_head.') for k in info.missing_keys):
            raise ValueError('Unexpected B0 weight transfer mismatch')
        return result

    def forward(self, pixel_values, labels=None, condition_labels=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        include_features = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        encoded = self.segformer(pixel_values, output_hidden_states=True,
                                 output_attentions=output_attentions, return_dict=True)
        logits = self.decode_head(encoded.hidden_states)
        condition_logits = self.condition_head(encoded.hidden_states[-1].mean((2, 3)))
        segmentation_loss = condition_loss = loss = None
        if labels is not None:
            from src.training.losses import SegmentationLoss
            segmentation_loss = SegmentationLoss()(logits, labels)
            loss = segmentation_loss
        if condition_labels is not None:
            if (condition_labels.shape != (pixel_values.shape[0],) or condition_labels.dtype != torch.long
                    or not ((condition_labels >= 0) & (condition_labels < len(CONDITIONS))).all()):
                raise ValueError('condition_labels must be a batch of int64 IDs in [0,4]')
            condition_loss = F.cross_entropy(condition_logits.float(), condition_labels)
            weighted = self.config.condition_loss_weight * condition_loss
            loss = weighted if loss is None else loss + weighted
        result = WeatherSegOutput(loss=loss, logits=logits,
            hidden_states=encoded.hidden_states if include_features else None,
            attentions=encoded.attentions, condition_logits=condition_logits,
            segmentation_loss=segmentation_loss, condition_loss=condition_loss)
        return result if return_dict else result.to_tuple()
