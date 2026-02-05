# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import inspect
import pkgutil
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Type

import torch.nn as nn

from modelopt.torch.puzzletron.anymodel.model_descriptor import (
    ModelDescriptor,
    ModelDescriptorFactory,
)
from modelopt.torch.puzzletron.anymodel.puzzformer.no_op import MatchingZeros, Same
from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.block_config import BlockConfig
from modelopt.torch.puzzletron.pruning.ffn_intermediate_pruning_mixin import (
    FFNIntermediateLayerDescriptor,
    FFNIntermediatePruningMixIn,
)
from modelopt.torch.puzzletron.pruning.pruning_mixin import PruningMixIn


def get_dynamic_modules(module_cls_str: str) -> List[Type[nn.Module]]:
    import transformers_modules

    matches = []
    for finder, modname, ispkg in pkgutil.walk_packages(
        transformers_modules.__path__, transformers_modules.__name__ + "."
    ):
        module = importlib.import_module(modname)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__name__ == module_cls_str:
                matches.append(obj)

    return matches


@dataclass
class NemotronHV2FFNIntermediateLayerDescriptor(FFNIntermediateLayerDescriptor):
    down_proj_name: str = "mixer.down_proj"
    ffn_prefix_name: str = "backbone.layers.{layer_idx}.mixer"
    linear_weight_names: List[str] = field(default_factory=lambda: ["down_proj", "up_proj"])


@ModelDescriptorFactory.register_decorator("nemotron_h_v2")
class NemotronHV2ModelDescriptor(ModelDescriptor):
    _DECODER_LAYER_CLS: Type[nn.Module] = None

    @staticmethod
    def decoder_layer_cls():
        decoder_cls_list = get_dynamic_modules("NemotronHBlock")
        if not decoder_cls_list:
            raise AssertionError(
                "NemotronH contains dynamic modules that should be cached beforehand, make sure to load your config using `load_model_config` or manually call `force_cache_dynamic_modules(config, checkpoint_dir)`"
            )
        return decoder_cls_list

    @staticmethod
    def block_config_to_layer_overrides(block_config: BlockConfig):
        override_kwargs = {}
        if block_config.ffn is not None and block_config.ffn.intermediate_size is not None:
            override_kwargs["intermediate_size"] = block_config.ffn.intermediate_size

        if (
            block_config.attention is not None
            and block_config.attention.num_key_value_heads is not None
        ):
            override_kwargs["num_key_value_heads"] = block_config.attention.num_key_value_heads

        if block_config.ffn is not None and block_config.ffn.moe is not None:
            override_kwargs["moe_intermediate_size"] = block_config.ffn.moe.expert_intermediate_dim
            override_kwargs["n_routed_experts"] = block_config.ffn.moe.num_local_experts

        return override_kwargs

    @staticmethod
    def _block_no_op_post_init(decoder_layer):
        """
        Due to the subblock structure of NemotronH always one of the subblock is set to no-op, for a real no-op both attention & ffn no-op should be set to True.
        """
        block_config = decoder_layer.config.block_configs[decoder_layer.layer_idx]
        ffn_no_op = block_config.ffn is not None and block_config.ffn.no_op
        attn_no_op = block_config.attention is not None and block_config.attention.no_op
        if ffn_no_op and attn_no_op:
            decoder_layer.norm = Same()
            decoder_layer.mixer = MatchingZeros()

    @staticmethod
    def attn_no_op_post_init(decoder_layer):
        NemotronHV2ModelDescriptor._block_no_op_post_init(decoder_layer)

    @staticmethod
    def mlp_no_op_post_init(decoder_layer):
        NemotronHV2ModelDescriptor._block_no_op_post_init(decoder_layer)

    @classmethod
    def create_dummy_block(cls, original_layer: nn.Module, block_index: int) -> nn.Module:
        dummy_block = super().create_dummy_block(original_layer, block_index)
        # Required by `NemotronHModel.forward`.
        dummy_block.block_type = original_layer.block_type
        return dummy_block

    @staticmethod
    def init_rotary_embedding(model, runtime):
        """
        NemotronH has no positional embeddings
        """
        pass

    @staticmethod
    def input_embedding_name():
        return "backbone.embeddings"

    @staticmethod
    def output_embedding_name():
        return "lm_head"

    @staticmethod
    def final_norm_name():
        return "backbone.norm_f"

    @staticmethod
    def layer_block_name(index: int):
        return f"backbone.layers.{index}"

    @classmethod
    def layer_structure(cls) -> Dict[str, Any]:
        """Define Nemotron-H v2 model structure using class-based approach.

        Nemotron-H is a hybrid architecture where each layer can be:
        - Mamba (SSM): Uses NemotronHMamba2Mixer
        - Attention: Uses NemotronHAttention
        - MLP (FFN): Uses NemotronHMLP

        All three share the same parent path "mixer".
        The `norm.weight` is shared and classified based on the mixer type in that layer.
        """
        return {
            "layer_pattern": "backbone.layers.{layer_idx}",
            "attention": {
                # Both Mamba and Attention count as "attention" subblock
                "module_classes": ["NemotronHMamba2Mixer", "NemotronHAttention"],
                "include_by_name": ["norm.weight"],  # Shared norm, assigned per-layer
            },
            "ffn": {
                "module_classes": ["NemotronHMLP"],
                "include_by_name": ["norm.weight"],  # Shared norm, assigned per-layer
            },
            "global_modules": {
                "embeddings": ["backbone.embeddings.weight"],
                "lm_head": ["backbone.norm_f.weight", "lm_head.weight"],
            },
        }

    @staticmethod
    def pruning_mixins() -> Dict[str, PruningMixIn]:
        return {
            "ffn_intermediate": FFNIntermediatePruningMixIn(
                NemotronHV2FFNIntermediateLayerDescriptor()
            ),
            # TODO: Add expert removal support when ExpertRemovalPruningMixIn is migrated
        }
