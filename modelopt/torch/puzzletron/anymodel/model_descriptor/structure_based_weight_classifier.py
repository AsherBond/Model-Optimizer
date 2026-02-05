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

"""Weight classification using layer_structure() definitions.

This module provides class-based weight classification for model descriptors.
"""

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import torch.nn as nn
from transformers import AutoModelForCausalLM

__all__ = ["StructureBasedWeightClassifier"]


class StructureBasedWeightClassifier:
    """Handles weight classification using layer_structure definitions.

    This classifier uses model structure inspection (via device_map='meta') to
    determine which weights belong to attention vs FFN subblocks, with per-layer
    caching for efficiency.
    """

    # Class-level cache: {descriptor_class_name: {layer_idx: {module_name: subblock_type}}}
    _classification_cache: Dict[str, Dict[int, Dict[str, str]]] = {}

    @classmethod
    def classify_weights(
        cls,
        model: nn.Module,
        structure: Dict[str, Any],
        weight_names: Iterable[str],
        num_hidden_layers: int,
        descriptor_class_name: str,
    ) -> Dict[str, List[str]]:
        """Classify weights into subblock groups.

        Args:
            model: Model loaded with device_map="meta" (structure only, no weights)
            structure: Output from ModelDescriptor.layer_structure()
            weight_names: Iterable of all weight names from model's state_dict
            num_hidden_layers: Number of layers in the model
            descriptor_class_name: Name of the descriptor class (for caching)

        Returns:
            Dictionary mapping group names to lists of weight names:
            {
                "embeddings": ["model.embed_tokens.weight"],
                "lm_head": ["model.norm.weight", "lm_head.weight"],
                "block_0_attention": [
                    "model.layers.0.input_layernorm.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.self_attn.k_proj.weight",
                    "model.layers.0.self_attn.v_proj.weight",
                    "model.layers.0.self_attn.o_proj.weight",
                ],
                "block_0_ffn": [
                    "model.layers.0.post_attention_layernorm.weight",
                    "model.layers.0.mlp.gate_proj.weight",
                    "model.layers.0.mlp.up_proj.weight",
                    "model.layers.0.mlp.down_proj.weight",
                ],
                "block_1_attention": [...],
                "block_1_ffn": [...],
                ...
            }
        """
        # Get classification map (cached after first call)
        classification_map = cls._get_classification_map(
            model, num_hidden_layers, structure, descriptor_class_name
        )

        weight_groups = defaultdict(list)
        layer_pattern_re = cls._layer_pattern_to_regex(structure["layer_pattern"])

        for weight_name in weight_names:
            # Try to match layer pattern
            match = layer_pattern_re.match(weight_name)

            if match:
                # Layer weight - classify as attention/ffn
                layer_idx = int(match.group(1))
                classification = cls._classify_layer_weight(
                    weight_name, layer_idx, structure, classification_map
                )

                if classification:
                    group_name = f"block_{layer_idx}_{classification}"
                    weight_groups[group_name].append(weight_name)
                else:
                    raise ValueError(
                        f"Could not classify layer weight: {weight_name}\n"
                        f"  Layer: {layer_idx}\n"
                        f"  Classification map: {classification_map}"
                    )
            else:
                # Global weight - classify as embeddings/lm_head
                classification = cls._classify_global_weight(
                    weight_name, structure["global_modules"]
                )

                if classification:
                    weight_groups[classification].append(weight_name)
                else:
                    raise ValueError(
                        f"Could not classify global weight: {weight_name}\n"
                        f"  Not a layer weight (doesn't match pattern: {structure['layer_pattern']})\n"
                        f"  Not in global_modules: {structure['global_modules']}"
                    )

        return dict(weight_groups)

    @classmethod
    def _get_classification_map(
        cls,
        model: nn.Module,
        num_hidden_layers: int,
        structure: Dict[str, Any],
        descriptor_class_name: str,
    ) -> Dict[int, Dict[str, str]]:
        """Get cached classification map or create new one.

        Args:
            model: Model loaded with device_map="meta"
            num_hidden_layers: Number of layers
            structure: Output from layer_structure()
            descriptor_class_name: Name of the descriptor class (for caching)

        Returns:
            Per-layer classification map: {0: {"self_attn": "attention"}, 1: {...}, ...}
        """
        # Use descriptor class name as cache key (e.g., "LlamaModelDescriptor")
        cache_key = descriptor_class_name

        if cache_key not in cls._classification_cache:
            cls._classification_cache[cache_key] = cls._inspect_model_structure_per_layer(
                model, num_hidden_layers, structure
            )
            print(f"Cached classification map for {cache_key} ({num_hidden_layers} layers)")
        else:
            print(f"Using cached classification map for {cache_key}")

        return cls._classification_cache[cache_key]

    @classmethod
    def _inspect_model_structure_per_layer(
        cls, model: nn.Module, num_hidden_layers: int, structure: Dict[str, Any]
    ) -> Dict[int, Dict[str, str]]:
        """Inspect model structure and classify modules by inspecting all layers.

        Works for both regular models (all layers same) and hybrid models (layers differ).

        Args:
            model: Model loaded with device_map="meta" (structure only, no weights)
            num_hidden_layers: Number of layers in the model
            structure: Output from layer_structure()

        Returns:
            Per-layer classification map: {0: {"self_attn": "attention"}, 1: {...}, ...}
        """
        # Get layers path from layer_pattern
        # "model.layers.{layer_idx}" -> "model.layers"
        # "backbone.layers.{layer_idx}" -> "backbone.layers"
        layer_pattern = structure["layer_pattern"]
        layers_path = layer_pattern.replace(".{layer_idx}", "")

        # Navigate to layers: "model.layers" -> model.model.layers
        layers = model
        for attr in layers_path.split("."):
            layers = getattr(layers, attr)

        # Classify by module type for each layer
        attention_classes = structure["attention"].get("module_classes", [])
        ffn_classes = structure["ffn"].get("module_classes", [])

        per_layer_map = {}
        for layer_idx in range(num_hidden_layers):
            layer = layers[layer_idx]
            layer_map = {}

            # Phase 1: Classify modules by their class type
            for module_name, module in layer.named_children():
                module_type = type(module)

                if module_type in attention_classes:
                    layer_map[module_name] = "attention"
                elif module_type in ffn_classes:
                    layer_map[module_name] = "ffn"

            # Phase 2: Apply include_by_name for modules not yet classified
            # Only add includes to a subblock type if that subblock type was found in this layer
            # (e.g., only add norm.weight to attention if this layer has attention modules)
            # Note: include_by_name uses weight names (e.g., "input_layernorm.weight")
            for subblock_type in ["attention", "ffn"]:
                # Check if this subblock type exists in this layer
                has_subblock = any(v == subblock_type for v in layer_map.values())

                if has_subblock and subblock_type in structure:
                    # Apply includes for this subblock type
                    for include_weight in structure[subblock_type].get("include_by_name", []):
                        # Extract module name: "input_layernorm.weight" -> "input_layernorm"
                        module_name = (
                            include_weight.rsplit(".", 1)[0]
                            if "." in include_weight
                            else include_weight
                        )

                        # Check for conflicts with Phase 1 classifications
                        if module_name in layer_map:
                            raise ValueError(
                                f"Configuration error in layer {layer_idx}: "
                                f"'{module_name}' is already classified as '{layer_map[module_name]}' "
                                f"by module_classes, but also listed in include_by_name for '{subblock_type}'. "
                                f"Remove it from include_by_name."
                            )

                        layer_map[module_name] = subblock_type

            per_layer_map[layer_idx] = layer_map

        return per_layer_map

    @classmethod
    def _layer_pattern_to_regex(cls, layer_pattern: str) -> re.Pattern:
        """Convert layer_pattern template to regex for matching layer weights.

        Args:
            layer_pattern: Template like "model.layers.{layer_idx}"

        Returns:
            Compiled regex like r"^model\\.layers\\.(\\d+)\\."
        """
        # Escape special regex characters
        escaped = re.escape(layer_pattern)
        # Replace escaped placeholder with capture group
        pattern = escaped.replace(r"\{layer_idx\}", r"(\d+)")
        # Anchor to start and ensure there's a dot after
        return re.compile(f"^{pattern}\\.")

    @classmethod
    def _classify_layer_weight(
        cls,
        weight_name: str,
        layer_idx: int,
        structure: Dict[str, Any],
        classification_map: Dict[int, Dict[str, str]],
    ) -> Optional[str]:
        """Classify a layer weight as 'attention' or 'ffn' using classification map.

        Args:
            weight_name: Full weight name (e.g., "model.layers.0.self_attn.q_proj.weight")
            layer_idx: Layer index extracted from weight name
            structure: Output from layer_structure()
            classification_map: Per-layer module name to subblock type mapping
                               Format: {0: {"self_attn": "attention"}, 1: {...}, ...}

        Returns:
            'attention', 'ffn', or None if cannot classify
        """
        layer_pattern = structure["layer_pattern"]
        layer_prefix = layer_pattern.format(layer_idx=layer_idx)

        # Remove layer prefix to get module path
        # "model.layers.0.self_attn.q_proj.weight" → "self_attn.q_proj.weight"
        if not weight_name.startswith(layer_prefix):
            return None

        module_path = weight_name[len(layer_prefix) + 1 :]  # +1 for the dot

        # Extract first component: "self_attn.q_proj.weight" → "self_attn"
        first_component = module_path.split(".")[0]

        # Look up in per-layer classification map
        layer_map = classification_map[layer_idx]
        return layer_map.get(first_component)

    @classmethod
    def _classify_global_weight(
        cls, weight_name: str, global_modules: Dict[str, Any]
    ) -> Optional[str]:
        """Classify a global (non-layer) weight.

        Args:
            weight_name: Weight name (e.g., "model.embed_tokens.weight")
            global_modules: global_modules dict from layer_structure()

        Returns:
            'embeddings', 'lm_head', or None
        """
        # Get embeddings and lm_head paths (both must be lists)
        embeddings_paths = global_modules.get("embeddings", [])
        lm_head_paths = global_modules.get("lm_head", [])

        # Check for duplicates
        duplicates = set(embeddings_paths) & set(lm_head_paths)
        if duplicates:
            raise ValueError(
                f"Weight(s) appear in both embeddings and lm_head in global_modules: {duplicates}"
            )

        # Check embeddings (exact match only)
        for path in embeddings_paths:
            if weight_name == path:
                return "embeddings"

        # Check lm_head (exact match only)
        for path in lm_head_paths:
            if weight_name == path:
                return "lm_head"

        return None
