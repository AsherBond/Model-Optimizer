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

"""Calibration framework for sparse attention methods."""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..sparse_attention import SparseAttentionModule
from ..stats_manager import SparseAttentionStatsManager


class DynamicThresholdCalibrator:
    """Dynamic threshold calibrator using length-based linear relationship.

    Implements calibration algorithm:
    1. Find hyperparameter 'a' where threshold λ = a / context_length
    2. Use dataset with different lengths and test multiple thresholds
    3. For each sample, find optimal threshold closest to target sparsity
    4. Use linear regression to fit: threshold = a * (1/length)

    Calibrates separate scale factors for prefill and decode phases.
    """

    def __init__(
        self,
        target_sparse_ratio: dict[str, float],
        threshold_trials: list[float] | None = None,
    ):
        """Initialize dynamic threshold calibrator.

        Args:
            target_sparse_ratio: Target sparsity ratio per phase, e.g. {"prefill": 0.5, "decode": 0.3}
            threshold_trials: List of thresholds to try during calibration
        """
        self.target_sparse_ratio = target_sparse_ratio

        # Default threshold trials if not provided
        self.threshold_trials = threshold_trials or [
            1e-6,
            5e-6,
            1e-5,
            5e-5,
            1e-4,
            5e-4,
            1e-3,
            5e-3,
            1e-2,
            5e-2,
            1e-1,
            5e-1,
        ]

        # Statistics tracking per phase: {phase: {sample_idx: {length, threshold_sparsities}}}
        self.sparsity_results: dict[str, dict[int, dict]] = {"prefill": {}, "decode": {}}

    def calibrate(
        self,
        model: nn.Module,
        forward_loop: Callable,
    ) -> dict[str, Any]:
        """Two-phase calibration: separate prefill and decode phases.

        Args:
            model: The model with sparse attention modules
            forward_loop: Forward loop callable(model, max_new_tokens=decode_tokens) -> None
                          Prefill uses max_new_tokens=1, decode uses default (decode_tokens)

        Returns:
            Dict with scale_factor as {"prefill": float, "decode": float}
        """
        attention_modules = [m for m in model.modules() if isinstance(m, SparseAttentionModule)]

        if not attention_modules:
            raise ValueError("No sparse attention modules found for calibration")

        print("Starting two-phase dynamic threshold calibration")
        print(f"Target sparsity: {self.target_sparse_ratio}")
        print(f"Threshold trials: {len(self.threshold_trials)}")

        # ===== Phase 1: Prefill Calibration =====
        print("\n--- Phase 1: Prefill Calibration ---")
        self._set_skip_phases(attention_modules, skip_phases={"decode"})
        prefill_results: dict[int, dict] = {}
        num_samples = 0

        for threshold in tqdm(self.threshold_trials, desc="Prefill thresholds"):
            self._set_threshold(attention_modules, threshold)
            self._enable_calibration_mode(attention_modules)
            self._reset_calibration_stats(attention_modules)

            with torch.no_grad():
                forward_loop(model, max_new_tokens=1)

            stats = self._extract_calibration_stats(attention_modules)
            self._disable_calibration_mode(attention_modules)

            # Derive num_samples from first threshold run
            if num_samples == 0:
                num_samples = len(stats)

            for idx, stat in enumerate(stats):
                if idx not in prefill_results:
                    prefill_results[idx] = {
                        "length": stat["sample_length"],
                        "threshold_sparsities": {},
                    }
                prefill_results[idx]["threshold_sparsities"][threshold] = stat["sparsity"]

        print(f"Collected {len(prefill_results)} prefill samples")

        # ===== Phase 2: Decode Calibration =====
        print("\n--- Phase 2: Decode Calibration ---")
        self._set_skip_phases(attention_modules, skip_phases={"prefill"})
        decode_results: dict[int, dict] = {}
        decode_tokens = 0

        for threshold in tqdm(self.threshold_trials, desc="Decode thresholds"):
            self._set_threshold(attention_modules, threshold)
            self._enable_calibration_mode(attention_modules)
            self._reset_calibration_stats(attention_modules)

            with torch.no_grad():
                forward_loop(model)  # Uses default max_new_tokens (decode_tokens)

            raw_stats = self._extract_calibration_stats(attention_modules)
            self._disable_calibration_mode(attention_modules)

            # Derive decode_tokens from first threshold run
            if decode_tokens == 0 and num_samples > 0:
                decode_tokens = len(raw_stats) // num_samples

            # Average decode stats per sample
            stats = self._average_decode_stats(raw_stats, num_samples, decode_tokens)

            for idx, stat in enumerate(stats):
                if idx not in decode_results:
                    decode_results[idx] = {
                        "length": stat["sample_length"],
                        "threshold_sparsities": {},
                    }
                decode_results[idx]["threshold_sparsities"][threshold] = stat["sparsity"]

        print(f"Collected {len(decode_results)} decode samples (averaged)")

        # ===== Compute Scale Factors =====
        print("\n--- Computing Scale Factors ---")
        scale_factors: dict[str, float] = {}
        phase_results: dict[str, dict] = {}

        # Store results for _compute_phase_scale_factor
        self.sparsity_results = {"prefill": prefill_results, "decode": decode_results}

        for phase in ("prefill", "decode"):
            result = self._compute_phase_scale_factor(phase)
            if result:
                scale_factors[phase] = result["scale_factor"]
                phase_results[phase] = result

        if not scale_factors:
            warnings.warn("Calibration did not produce valid results for any phase")
            return {}

        # Print results
        print("\nCalibration Results:")
        for phase, result in phase_results.items():
            target = self.target_sparse_ratio[phase]
            print(f"\n  [{phase.upper()}] (target: {target:.2%})")
            print(
                f"    Scale factor: {result['scale_factor']:.6f} (std: {result['scale_factor_std']:.6f})"
            )
            print(f"    R-squared: {result['r_squared']:.4f}")
            print(f"    Achieved sparsity: {result['avg_achieved_sparsity']:.2%}")

        print("\nExample thresholds (λ = scale_factor / length):")
        for length in [1024, 2048, 4096, 8192]:
            parts = [f"{phase}: {sf / length:.2e}" for phase, sf in scale_factors.items()]
            print(f"  Length {length:5d}: {', '.join(parts)}")

        # Clear skip phases
        self._set_skip_phases(attention_modules, skip_phases=set())

        return {
            "scale_factor": scale_factors,
            "phase_results": phase_results,
            "target_sparsity": self.target_sparse_ratio,
            "calibration_type": "two_phase_dynamic",
        }

    def _compute_phase_scale_factor(self, phase: str) -> dict[str, Any] | None:
        """Compute scale factor for a single phase using linear regression.

        Args:
            phase: "prefill" or "decode"

        Returns:
            Dict with scale_factor, r_squared, etc., or None if insufficient data
        """
        results = self.sparsity_results.get(phase, {})
        if not results:
            warnings.warn(f"No samples collected for {phase} phase")
            return None

        target = self.target_sparse_ratio[phase]

        # Find optimal threshold for each sample
        optimal_pairs = []
        for sample_result in results.values():
            if not sample_result["threshold_sparsities"]:
                continue
            best_threshold, achieved_sparsity = min(
                sample_result["threshold_sparsities"].items(),
                key=lambda item: abs(item[1] - target),
            )
            optimal_pairs.append(
                {
                    "length": sample_result["length"],
                    "optimal_threshold": best_threshold,
                    "achieved_sparsity": achieved_sparsity,
                }
            )

        if not optimal_pairs:
            warnings.warn(f"No optimal threshold pairs found for {phase} phase")
            return None

        # Linear regression: threshold = a * (1/length)
        lengths = np.array([p["length"] for p in optimal_pairs])
        thresholds = np.array([p["optimal_threshold"] for p in optimal_pairs])

        x = 1.0 / lengths
        y = thresholds

        scale_factor = float(np.sum(x * y) / np.sum(x**2))
        scale_factor_std = float(np.std(y * lengths))

        # R-squared
        y_pred = scale_factor * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        avg_achieved_sparsity = float(np.mean([p["achieved_sparsity"] for p in optimal_pairs]))

        return {
            "scale_factor": scale_factor,
            "scale_factor_std": scale_factor_std,
            "r_squared": r_squared,
            "num_samples": len(optimal_pairs),
            "avg_achieved_sparsity": avg_achieved_sparsity,
            "optimal_pairs": optimal_pairs,
        }

    def _enable_calibration_mode(self, modules: list[nn.Module]):
        """Enable calibration mode on sparse attention modules."""
        for idx, module in enumerate(modules):
            # Create stats manager if needed
            if not module._stats_manager:
                module._stats_manager = SparseAttentionStatsManager(
                    module_name=f"sparse_attn_{idx}", enabled=True
                )
            else:
                # Re-enable if disabled
                module._stats_manager.enabled = True

            # Enable calibration mode with fresh stats
            module._stats_manager.set_calibration_mode(enabled=True, reset_history=True)
            module._sparse_method_instance.set_calibration_mode(True)

    def _disable_calibration_mode(self, modules: list[nn.Module]):
        """Disable calibration mode (but keep stats enabled if collect_stats=True)."""
        for module in modules:
            if module._stats_manager:
                module._stats_manager.set_calibration_mode(enabled=False)

            module._sparse_method_instance.set_calibration_mode(False)

    def _extract_calibration_stats(self, modules: list[nn.Module]) -> list[dict]:
        """Extract per-sample calibration statistics from modules.

        Args:
            modules: List of attention modules

        Returns:
            List of per-sample statistics across all modules
        """
        # Collect from all stats managers
        all_per_sample_stats = []

        for module in modules:
            # Skip modules without stats manager
            if not hasattr(module, "_stats_manager") or module._stats_manager is None:
                continue

            manager_stats = module._stats_manager.get_calibration_stats()
            if manager_stats:
                all_per_sample_stats.append(manager_stats)

        if not all_per_sample_stats:
            return []

        # Aggregate across modules by sample index
        num_samples = len(all_per_sample_stats[0])
        aggregated_stats = []

        for sample_idx in range(num_samples):
            sparsities = []
            sample_length = 0

            for module_stats in all_per_sample_stats:
                if sample_idx < len(module_stats):
                    sample_stat = module_stats[sample_idx]
                    sparsities.append(sample_stat.get("sparsity", 0.0))
                    if not sample_length and "sample_length" in sample_stat:
                        sample_length = sample_stat["sample_length"]

            avg_sparsity = np.mean(sparsities) if sparsities else 0.0

            # Get phase from first module's stats (all modules process same sample)
            phase = "unknown"
            for module_stats in all_per_sample_stats:
                if sample_idx < len(module_stats) and "phase" in module_stats[sample_idx]:
                    phase = module_stats[sample_idx]["phase"]
                    break

            aggregated_stats.append(
                {
                    "sparsity": avg_sparsity,
                    "sample_length": sample_length,
                    "phase": phase,
                }
            )

        return aggregated_stats

    def _set_threshold(self, modules: list[nn.Module], threshold: float):
        """Set threshold on sparse attention modules."""
        for module in modules:
            module._sparse_method_instance.threshold = threshold

    def _set_skip_phases(self, modules: list[nn.Module], skip_phases: set[str]):
        """Set phases to skip during stats collection."""
        for module in modules:
            if module._stats_manager:
                module._stats_manager.skip_phases = skip_phases

    def _reset_calibration_stats(self, modules: list[nn.Module]):
        """Reset calibration stats for a fresh collection run."""
        for module in modules:
            if module._stats_manager:
                module._stats_manager.per_sample_stats = []

    def _average_decode_stats(
        self, stats: list[dict], num_samples: int, decode_tokens: int
    ) -> list[dict]:
        """Average decode stats: every decode_tokens entries → 1 entry per sample.

        Args:
            stats: Flat list of decode statistics (num_samples * decode_tokens entries)
            num_samples: Number of calibration samples
            decode_tokens: Number of decode tokens generated per sample

        Returns:
            List of averaged statistics (num_samples entries)
        """
        if len(stats) != num_samples * decode_tokens:
            warnings.warn(
                f"Expected {num_samples * decode_tokens} decode stats, got {len(stats)}. "
                "Results may be inaccurate."
            )

        averaged = []
        for i in range(num_samples):
            start = i * decode_tokens
            end = start + decode_tokens

            if end > len(stats):
                break

            sample_stats = stats[start:end]
            avg_sparsity = np.mean([s["sparsity"] for s in sample_stats])
            sample_length = sample_stats[0]["sample_length"]

            averaged.append(
                {
                    "sparsity": avg_sparsity,
                    "sample_length": sample_length,
                    "phase": "decode",
                }
            )

        return averaged
