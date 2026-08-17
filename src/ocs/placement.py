"""Expert affinity tracking for OCS-aware expert placement.

Tracks which experts are frequently co-activated by the router, then
suggests expert-to-rank mappings that group co-activated experts together.
This minimizes OCS reconfiguration by keeping frequently communicating
expert pairs on the same rank (intra-rank transfer = no circuit needed)
or on ranks with stable circuits.

The tracker is sampling-based: it records routing decisions during training
and produces circuit placement plans for inference pre-configuration.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import torch


class ExpertAffinityTracker:
    """Tracks expert co-activation from router outputs.

    For top-K routing (K >= 2), records pairwise co-selection of experts.
    For top-1 routing, records per-expert usage frequency.

    The resulting affinity matrix can be used to:
      - Suggest expert-to-rank placement minimizing inter-rank communication
      - Estimate OCS circuit pressure (many distinct pairs = many circuits needed)
      - Pre-configure OCS circuits for inference based on training patterns

    Usage:
        tracker = ExpertAffinityTracker(num_experts)
        for step in warmup_steps:
            expert_ids, gate_weights, _ = moe.router(tokens)
            tracker.record_routing(expert_ids, gate_weights)
        suggested_placement = tracker.suggest_placement(experts_per_rank, world_size)
        # For OCS preset: export training affinity, use in inference
        plan = tracker.compute_circuit_plan(expert_to_rank, max_circuits=16)
    """

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.co_activation_counts = torch.zeros(num_experts, num_experts, dtype=torch.float64)
        self.expert_usage = torch.zeros(num_experts, dtype=torch.float64)
        self.total_samples = 0
        self._phase = "default"

    def record_routing(
        self,
        expert_ids: torch.Tensor,
        gate_weights: torch.Tensor,
        phase: Optional[str] = None,
        token_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Record one routing event.

        Args:
            expert_ids: expert assignments. Shape [T] for top-1, [T, K] for top-K.
            gate_weights: routing weights, shape [T, K].
            phase: optional phase label ("training", "inference", "warmup").
            token_weights: optional per-token weight multipliers, shape [T].
                When provided, co-activation counts are multiplied by these
                weights (e.g. guide-model centrality scores).
        """
        T = expert_ids.shape[0]

        # Per-token weight: default to 1.0 if not provided
        if token_weights is None:
            w = torch.ones(T, dtype=torch.float64)
        else:
            w = token_weights.to(dtype=torch.float64)

        if expert_ids.dim() == 1:
            for e in expert_ids.unique():
                mask = (expert_ids == e)
                count = (mask.float() * w).sum().item()
                self.co_activation_counts[e, e] += count
                self.expert_usage[e] += count
            self.total_samples += T
        else:
            K = expert_ids.shape[1]
            self.total_samples += T

            for i in range(T):
                wi = w[i].item()
                for a in range(K):
                    ea = int(expert_ids[i, a].item())
                    self.expert_usage[ea] += wi
                    for b in range(K):
                        eb = int(expert_ids[i, b].item())
                        self.co_activation_counts[ea, eb] += wi

    def get_affinity_scores(self) -> torch.Tensor:
        """Return normalized co-activation matrix [num_experts, num_experts].

        Values are in [0, 1], representing the probability that expert e_b
        is co-selected when expert e_a is selected.
        """
        if self.total_samples == 0:
            return torch.zeros(self.num_experts, self.num_experts)

        normalized = self.co_activation_counts.clone()
        for e in range(self.num_experts):
            if self.expert_usage[e] > 0:
                normalized[e] /= self.expert_usage[e]

        return normalized

    def export_affinity(self) -> Dict:
        """Export co-activation counts and usage as a serializable dict.

        Returns:
            Dict with keys: num_experts, total_samples, expert_usage (list),
            co_activation_counts (list of lists), and normalized co_activation
            matrix (flattened). Suitable for JSON serialization.
        """
        affinity = self.get_affinity_scores()
        return {
            "num_experts": self.num_experts,
            "total_samples": self.total_samples,
            "expert_usage": self.expert_usage.tolist(),
            "co_activation_raw": self.co_activation_counts.tolist(),
            "co_activation_norm": affinity.tolist(),
        }

    def export_affinity_json(self, path: str) -> None:
        """Save affinity data to a JSON file."""
        data = self.export_affinity()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_affinity_from_export(self, data: Dict) -> None:
        """Load affinity from a previously exported dict."""
        self.num_experts = data["num_experts"]
        self.total_samples = data["total_samples"]
        self.expert_usage = torch.tensor(data["expert_usage"], dtype=torch.float64)
        self.co_activation_counts = torch.tensor(
            data["co_activation_raw"], dtype=torch.float64,
        )

    def compute_circuit_plan(
        self,
        expert_to_rank: Optional[Dict[int, int]] = None,
        max_circuits: int = 16,
        experts_per_rank: int = 1,
        world_size: int = 4,
    ) -> List[Tuple[int, int, float]]:
        """Compute an OCS circuit placement plan from co-activation affinity.

        Maps expert co-activation to rank-pair communication pressure, then
        returns an ordered list of (src_rank, dst_rank, score) tuples sorted
        by descending affinity score — the recommended circuit placement order.

        Args:
            expert_to_rank: optional pre-built expert→rank mapping. If None,
                expert e maps to rank (e // experts_per_rank).
            max_circuits: cap on number of circuits in the plan (returns top-K).
            experts_per_rank: experts per GPU rank (used if expert_to_rank is None).
            world_size: number of ranks (used if expert_to_rank is None).

        Returns:
            List of (src_rank, dst_rank, affinity_score) sorted by score descending.
            The plan should be applied in order: establish circuits for the
            highest-scoring rank pairs first, up to max_circuits.
        """
        if expert_to_rank is None:
            expert_to_rank = {
                e: e // experts_per_rank for e in range(self.num_experts)
            }

        affinity = self.get_affinity_scores()

        rank_pair_scores: Dict[Tuple[int, int], float] = {}
        for ea in range(self.num_experts):
            ra = expert_to_rank.get(ea)
            if ra is None:
                continue
            for eb in range(self.num_experts):
                rb = expert_to_rank.get(eb)
                if rb is None:
                    continue
                if ra == rb:
                    continue
                score = affinity[ea, eb].item()
                if score > 0:
                    key = (ra, rb)
                    rank_pair_scores[key] = max(
                        rank_pair_scores.get(key, 0.0), score,
                    )

        sorted_pairs = sorted(
            rank_pair_scores.items(), key=lambda x: x[1], reverse=True,
        )
        plan = [
            (src, dst, score) for (src, dst), score in sorted_pairs[:max_circuits]
        ]
        return plan

    def suggest_placement(
        self,
        experts_per_rank: int,
        world_size: int,
    ) -> List[List[int]]:
        """Suggest expert-to-rank placement based on co-activation affinity.

        Uses a greedy clustering heuristic: for each rank, pick a seed expert
        (highest total affinity to unplaced experts), then fill remaining slots
        with the experts most co-activated with the seed.

        Returns:
            List of length world_size, each element is a list of expert IDs
            assigned to that rank. Total experts across all ranks equals
            world_size * experts_per_rank = num_experts.
        """
        affinity = self.get_affinity_scores()
        total_experts = world_size * experts_per_rank

        if self.total_samples == 0 or total_experts != self.num_experts:
            return [
                list(range(r * experts_per_rank, (r + 1) * experts_per_rank))
                for r in range(world_size)
            ]

        remaining = set(range(self.num_experts))
        placement: List[List[int]] = [[] for _ in range(world_size)]

        for rank in range(world_size):
            if not remaining:
                break
            slots = experts_per_rank

            if len(placement[rank]) == 0:
                best_seed = -1
                best_score = -1.0
                for cand in sorted(remaining):
                    score = affinity[cand, list(remaining)].sum().item()
                    if score > best_score:
                        best_score = score
                        best_seed = cand
                if best_seed >= 0:
                    placement[rank].append(best_seed)
                    remaining.remove(best_seed)
                    slots -= 1

            while slots > 0 and remaining:
                best_expert = -1
                best_score = -1.0
                for cand in sorted(remaining):
                    score = sum(
                        affinity[cand, e].item() + affinity[e, cand].item()
                        for e in placement[rank]
                    )
                    if score > best_score:
                        best_score = score
                        best_expert = cand
                if best_expert >= 0:
                    placement[rank].append(best_expert)
                    remaining.remove(best_expert)
                    slots -= 1
                else:
                    break

        for i, e in enumerate(sorted(remaining)):
            placement[i % world_size].append(e)

        return placement

    def get_expert_utilization(self) -> torch.Tensor:
        """Return normalized per-expert usage frequencies [num_experts]."""
        if self.total_samples == 0:
            return torch.zeros(self.num_experts)
        return self.expert_usage / self.expert_usage.sum()

    def reset(self) -> None:
        """Reset all counters. Useful for per-epoch or per-phase tracking."""
        self.co_activation_counts.zero_()
        self.expert_usage.zero_()
        self.total_samples = 0
