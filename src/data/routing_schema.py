"""
routing_schema.py — Unified routing data structures for MoE OCS research.

Canonical JSON format for routing traces captured from real MoE models
(Qwen-MoE, etc.). Used by the OCS simulator to replay real routing
patterns instead of synthetic ones.

Indexed by absolute token position (not forward-pass step) so that
cross-token and cross-layer analysis is trivial.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class RunMeta:
    """Metadata recorded once per inference run."""

    model_id: str
    model_type: str
    num_layers: int
    num_moe_layers: int
    num_experts: int
    top_k: int
    prompt_len: int
    generated_len: int
    total_tokens: int
    backend: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class LayerRoute:
    """Routing decision for a single layer at a single token position."""

    experts: list[int]
    weights: list[float]


@dataclass
class TokenRoute:
    """Routing decisions for one token position across all MoE layers."""

    token_pos: int
    token_id: int
    token_str: str
    phase: str
    layers: dict[str, LayerRoute]


@dataclass
class RoutingTrace:
    """Complete routing trace for a single inference run."""

    meta: RunMeta
    prompt_tokens: list[int]
    generated_tokens: list[int]
    routes: list[TokenRoute]

    def to_dict(self) -> dict:
        return {
            "meta": asdict(self.meta),
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "routes": [
                {
                    "token_pos": r.token_pos,
                    "token_id": r.token_id,
                    "token_str": r.token_str,
                    "phase": r.phase,
                    "layers": {
                        lid: asdict(lr) for lid, lr in r.layers.items()
                    },
                }
                for r in self.routes
            ],
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "RoutingTrace":
        with open(path) as f:
            raw = json.load(f)
        meta_raw = raw["meta"]
        meta = RunMeta(
            model_id=meta_raw["model_id"],
            model_type=meta_raw["model_type"],
            num_layers=meta_raw["num_layers"],
            num_moe_layers=meta_raw["num_moe_layers"],
            num_experts=meta_raw["num_experts"],
            top_k=meta_raw["top_k"],
            prompt_len=meta_raw["prompt_len"],
            generated_len=meta_raw["generated_len"],
            total_tokens=meta_raw["total_tokens"],
            backend=meta_raw["backend"],
            run_id=meta_raw.get("run_id", ""),
        )
        routes = [
            TokenRoute(
                token_pos=r["token_pos"],
                token_id=r["token_id"],
                token_str=r["token_str"],
                phase=r["phase"],
                layers={
                    lid: LayerRoute(**lr) for lid, lr in r["layers"].items()
                },
            )
            for r in raw["routes"]
        ]
        return cls(
            meta=meta,
            prompt_tokens=raw["prompt_tokens"],
            generated_tokens=raw["generated_tokens"],
            routes=routes,
        )

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_tokens + self.generated_tokens

    def total_routing_events(self) -> int:
        return sum(len(r.layers) for r in self.routes)

    def expert_load(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for route in self.routes:
            for lr in route.layers.values():
                for e in lr.experts:
                    counts[e] = counts.get(e, 0) + 1
        return dict(sorted(counts.items()))

    def per_layer_expert_load(self) -> dict[str, dict[int, int]]:
        result: dict[str, dict[int, int]] = {}
        for route in self.routes:
            for lid, lr in route.layers.items():
                if lid not in result:
                    result[lid] = {}
                for e in lr.experts:
                    result[lid][e] = result[lid].get(e, 0) + 1
        return result

    def rank_communication_matrix(self, experts_per_rank: int) -> dict[tuple[int, int], int]:
        """Build a communication heatmap: (src_rank, dst_rank) → count.

        For each token-layer combination, maps the selected experts to the
        ranks that own them. Accumulates how often each rank pair communicates.
        This is the primary statistic for OCS circuit placement.
        """
        if experts_per_rank <= 0:
            raise ValueError("experts_per_rank must be positive")

        def _rank_of(expert_id: int) -> int:
            return expert_id // experts_per_rank

        matrix: dict[tuple[int, int], int] = {}
        for route in self.routes:
            for lr in route.layers.values():
                src_experts = lr.experts
                for src_e in src_experts:
                    src_rank = _rank_of(src_e)
                    for dst_e in src_experts:
                        dst_rank = _rank_of(dst_e)
                        key = (src_rank, dst_rank)
                        matrix[key] = matrix.get(key, 0) + 1
        return matrix

    def per_layer_rank_targets(
        self, experts_per_rank: int
    ) -> dict[str, list[tuple[int, set[int]]]]:
        """Per-layer, per-token: which ranks are target ranks.

        Returns dict mapping layer_id → list of (token_pos, set[target_rank]).
        This drives the OCS circuit pre-establishment logic.
        """
        if experts_per_rank <= 0:
            raise ValueError("experts_per_rank must be positive")

        def _rank_of(expert_id: int) -> int:
            return expert_id // experts_per_rank

        result: dict[str, list[tuple[int, set[int]]]] = {}
        for route in self.routes:
            for lid, lr in route.layers.items():
                if lid not in result:
                    result[lid] = []
                target_ranks = {_rank_of(e) for e in lr.experts}
                result[lid].append((route.token_pos, target_ranks))
        return result
