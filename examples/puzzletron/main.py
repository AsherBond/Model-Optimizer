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

"""
Main script for running the puzzletron algorithm on large language models (based on Puzzle paper https://arxiv.org/abs/2411.19146).

This script provides three modes:
1. Default mode: Runs the full puzzletron pipeline
2. MIP-only mode: Runs only the MIP search and realize models phase
3. MIP sweep mode: Runs MIP for multiple memory compression rates (enabled via config)

Usage:
    # Full puzzletron pipeline
    torchrun main.py --config ./configs/llama_3.2_1B_pruneffn_memory.yaml

    # Only MIP search and realize models phase
    torchrun main.py --config ./configs/llama_3.2_1B_pruneffn_memory.yaml --mip-only

    # MIP sweep mode (set mip.sweep.enabled: true in config)
    torchrun main.py --config ./configs/llama_3.2_1B_pruneffn_memory.yaml --mip-only
"""

import argparse
import json
from datetime import timedelta
from pathlib import Path

from transformers import AutoConfig

import modelopt.torch.nas as mtn
import modelopt.torch.puzzletron.mip.mip_and_realize_models as mip_and_realize_models
import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.nas.plugins.puzzletron_nas_plugin import PuzzletronModel
from modelopt.torch.puzzletron.tools.hydra_utils import (
    initialize_hydra_config_for_dir,
    register_hydra_resolvers,
)
from modelopt.torch.puzzletron.tools.logger import mprint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compress large language models using the Puzzletron algorithm (based on Puzzle paper https://arxiv.org/abs/2411.19146)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the main config YAML file (e.g., ./configs/llama_3.2_1B_pruneffn_memory.yaml)",
    )
    parser.add_argument(
        "--mip-only",
        action="store_true",
        help="Run only the MIP search and realize models phase (skip pruning and NAS scoring)",
    )

    return parser.parse_args()


def run_full_puzzletron(hydra_config_path: str):
    """Run the full puzzletron pipeline.

    Args:
        config_path: Path to the YAML configuration file
    """
    mprint("Puzzletron Progress 1/8: starting puzzletron pipeline")
    dist.setup(timeout=timedelta(10))

    # Register Hydra custom resolvers (needed for config resolution)
    register_hydra_resolvers()

    hydra_config_path = Path(hydra_config_path).resolve()
    hydra_config_dir = str(hydra_config_path.parent)
    hydra_config_name = hydra_config_path.stem

    # Load hydra config
    hydra_cfg = initialize_hydra_config_for_dir(
        config_dir=hydra_config_dir,
        config_name=hydra_config_name,
        overrides=[],
    )

    # Convert model (convert from HF to DeciLM, score pruning activations,
    # prune the model and save pruned checkpoints)
    input_model = PuzzletronModel()
    converted_model = mtn.convert(
        input_model,
        mode=[
            (
                "puzzletron",
                {
                    "puzzle_dir": str(hydra_cfg.puzzle_dir),
                    "input_model_path": hydra_cfg.input_hf_model_path,
                    "hydra_config_dir": hydra_config_dir,
                    "hydra_config_name": hydra_config_name,
                    "dataset_path": str(hydra_cfg.dataset_path),
                },
            )
        ],
    )

    # Run NAS search (build replacement library and compute stats,
    # compute one block scores, run MIP and realize models)
    mtn.search(
        converted_model,
        constraints={},  # this is not used as the search space is defined in the hydra config
        dummy_input=None,  # Not used
        config={},  # this is not used as the search space is defined in the hydra config
    )

    dist.cleanup()
    mprint("Puzzletron Progress 8/8: puzzletron pipeline completed (multi-gpu)")


def get_teacher_memory_from_subblock_stats(hydra_cfg) -> float:
    """Calculate teacher model memory from subblock_stats.json.

    Args:
        hydra_cfg: Hydra configuration object

    Returns:
        Total teacher memory in MiB
    """
    puzzle_dir = Path(hydra_cfg.puzzle_dir)

    # Load model config to get number of layers and teacher architecture
    teacher_dir = Path(hydra_cfg.teacher_dir)
    model_config = AutoConfig.from_pretrained(teacher_dir, trust_remote_code=True)
    num_layers = model_config.num_hidden_layers
    teacher_ffn_intermediate = model_config.intermediate_size
    teacher_num_kv_heads = model_config.num_key_value_heads  # For GQA models

    # Get the MIP configuration
    mip_subblock_args = hydra_cfg.mip.subblock_stats_args[0]
    batch_size = mip_subblock_args["batch_size"]
    weights_dtype = str(mip_subblock_args["weights_dtype"])
    activations_dtype = str(mip_subblock_args["activations_dtype"])
    kv_cache_dtype = str(mip_subblock_args["kv_cache_dtype"])

    # Load subblock_stats.json
    subblock_stats_path = puzzle_dir / "subblock_stats.json"
    with open(subblock_stats_path) as f:
        subblock_stats_list = json.load(f)

    # Find the entry matching our MIP configuration and teacher's n_embd
    matching_stats = None
    for stats_entry in subblock_stats_list:
        args = stats_entry["args"]
        if (
            args["batch_size"] == batch_size
            and args["weights_dtype"] == weights_dtype
            and args["activations_dtype"] == activations_dtype
            and args["kv_cache_dtype"] == kv_cache_dtype
            and args.get("n_embd") == model_config.hidden_size
        ):
            matching_stats = stats_entry
            break

    if matching_stats is None:
        raise ValueError(
            f"No subblock_stats entry found for batch_size={batch_size}, "
            f"dtypes=({weights_dtype}, {activations_dtype}, {kv_cache_dtype}), "
            f"n_embd={model_config.hidden_size}"
        )

    # Get non-block memory (embeddings, LM head, etc.)
    total_memory = matching_stats.get("non_block", {}).get("memory_mib", 0.0)

    # Find the teacher FFN and Attention subblocks
    # Note: Each subblock is EITHER attention OR ffn, not both
    # We need to find BOTH and add their memory together
    teacher_ffn_subblock = None
    teacher_attention_subblock = None

    for subblock in matching_stats.get("subblocks", []):
        subblock_class = subblock.get("subblock_config_class", "")
        subblock_config = subblock.get("subblock_config", {})

        # Check for FFN subblocks with teacher's intermediate_size
        if "FFN" in subblock_class:
            ffn_size = subblock_config.get("intermediate_size")
            if ffn_size == teacher_ffn_intermediate and not subblock_config.get("no_op", False):
                teacher_ffn_subblock = subblock

        # Check for Attention subblocks with teacher's num_key_value_heads
        elif "Attention" in subblock_class:
            kv_heads = subblock_config.get("num_key_value_heads")
            if kv_heads == teacher_num_kv_heads and not subblock_config.get("no_op", False):
                teacher_attention_subblock = subblock

    if teacher_ffn_subblock is None:
        raise ValueError(
            f"Could not find teacher FFN subblock with intermediate_size={teacher_ffn_intermediate}"
        )

    if teacher_attention_subblock is None:
        raise ValueError(
            f"Could not find teacher Attention subblock with num_key_value_heads={teacher_num_kv_heads}"
        )

    # Calculate total teacher memory: non_block + (ffn_memory + attention_memory) * num_layers
    per_layer_memory = teacher_ffn_subblock["memory_mib"] + teacher_attention_subblock["memory_mib"]
    total_memory += per_layer_memory * num_layers

    return total_memory


def run_mip_sweep(hydra_cfg):
    """Run MIP for multiple memory compression rates and generate CSV with results.

    This function is called when mip.sweep.enabled is True in the config.

    Args:
        hydra_cfg: Hydra configuration object with mip.sweep settings
    """
    mprint("=" * 80)
    mprint("MIP Sweep Mode Enabled")
    mprint("=" * 80)

    # Get sweep configuration
    sweep_cfg = hydra_cfg.mip.sweep
    compression_rates = sweep_cfg.memory_compression_rates
    output_csv = sweep_cfg.output_csv
    puzzle_dir = Path(hydra_cfg.puzzle_dir)

    mprint(f"Compression rates: {compression_rates}")
    mprint(f"Output CSV: {output_csv}")
    mprint(f"Puzzle directory: {puzzle_dir}")

    # Calculate teacher memory from subblock_stats and replacement_library
    teacher_memory = get_teacher_memory_from_subblock_stats(hydra_cfg)
    mprint(
        f"Teacher memory (from subblock_stats): {teacher_memory:.1f} MiB ({teacher_memory / 1024:.1f} GiB)"
    )

    # TODO: Implement sweep logic
    # 1. For each compression rate:
    #    - Calculate target_memory = teacher_memory * rate
    #    - Run MIP with this target
    #    - Realize and validate model
    # 2. Collect all results and generate CSV

    mprint("=" * 80)
    mprint("MIP sweep functionality will be implemented in next phase")
    mprint("=" * 80)


def run_mip_only(hydra_config_path: str):
    """Run only the MIP search and realize models phase.

    This assumes that pruning, replacement library building, NAS scoring, and subblock stats calculation
    have already been completed.

    Args:
        hydra_config_path: Path to the YAML configuration file
    """
    dist.setup(timeout=timedelta(10))

    # Register Hydra custom resolvers (needed for config resolution)
    register_hydra_resolvers()

    hydra_config_path = Path(hydra_config_path).resolve()
    hydra_config_dir = str(hydra_config_path.parent)
    hydra_config_name = hydra_config_path.stem

    # Load hydra config
    hydra_cfg = initialize_hydra_config_for_dir(
        config_dir=hydra_config_dir,
        config_name=hydra_config_name,
        overrides=[],
    )

    # Check if sweep mode is enabled
    if hasattr(hydra_cfg.mip, "sweep") and hydra_cfg.mip.sweep.get("enabled", False):
        run_mip_sweep(hydra_cfg)
    else:
        # mip_and_realize_models (distributed processing)
        # TODO: How to make it part of mnt.search() api, similarly to run_full_puzzletron() API
        mprint("Puzzletron Progress 7/8: running MIP and realizing models (multi-gpu)")
        mip_and_realize_models.launch_mip_and_realize_model(hydra_cfg)

    dist.cleanup()
    mprint("Puzzletron Progress 8/8: puzzletron pipeline completed (multi-gpu)")


def main():
    args = parse_args()

    if args.mip_only:
        run_mip_only(hydra_config_path=args.config)
    else:
        run_full_puzzletron(hydra_config_path=args.config)


if __name__ == "__main__":
    main()
