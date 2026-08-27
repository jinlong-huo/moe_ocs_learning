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
import math
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


def _validate_placement(placement: dict, meta: "RunMeta", problems: list[str]) -> None:
    """Light consistency check for an optional placement manifest.

    The manifest records the cost-side projection attached to a trace:
    expert -> rank, rank -> physical (pod, node, local_rank), and the
    topology shape that produced the locations. It must be *consistent with*
    the routing meta (same expert space), but it never feeds back into
    routing — a malformed manifest is a capture bug, not a routing bug.
    """
    e2r = placement.get("expert_to_rank")
    if e2r is not None:
        if len(e2r) != meta.num_experts:
            problems.append(
                f"placement expert_to_rank length {len(e2r)} "
                f"!= meta.num_experts {meta.num_experts}"
            )
        else:
            if any(not (0 <= int(r) < len(e2r)) for r in e2r):
                problems.append("placement expert_to_rank contains out-of-range rank")
    r2l = placement.get("rank_to_location")
    ws = placement.get("world_size")
    if r2l is not None and ws is not None and len(r2l) != ws:
        problems.append(
            f"placement rank_to_location length {len(r2l)} != world_size {ws}"
        )
    topo = placement.get("topology")
    if topo is not None:
        for k in ("num_pods", "nodes_per_pod", "ranks_per_node"):
            if k not in topo:
                problems.append(f"placement.topology missing {k!r}")


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
    guide_affinity: Optional[list[list[float]]] = None
    placement: Optional[dict] = None

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
            "guide_affinity": self.guide_affinity
                if self.guide_affinity is not None else None,
            "placement": self.placement,
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
            raw = json.load(fp=f)
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
            guide_affinity=raw.get("guide_affinity"),
            placement=raw.get("placement"),
        )

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_tokens + self.generated_tokens

    def validate(self) -> None:
        """Verify internal consistency before the trace is saved.

        Catches capture bugs — out-of-range experts, corrupt token
        positions, wrong top-k arity, bad weights — so downstream stages
        (affinity, preconfig, OCS replay) fail loudly instead of silently
        consuming a poisoned trace. Raises ValueError listing all problems.
        """
        problems: list[str] = []
        num_experts = self.meta.num_experts
        top_k = self.meta.top_k
        num_layers = self.meta.num_layers
        total_tokens = self.meta.total_tokens

        if not self.routes:
            problems.append("no routes captured (empty trace)")

        positions = [r.token_pos for r in self.routes]
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            problems.append("token positions are not strictly increasing/unique")

        for r in self.routes:
            if not (0 <= r.token_pos < total_tokens):
                problems.append(
                    f"token_pos {r.token_pos} out of range [0, {total_tokens})"
                )
            for lid, lr in r.layers.items():
                try:
                    layer_idx = int(lid)
                except ValueError:
                    problems.append(f"layer id {lid!r} is not an integer")
                    continue
                if not (0 <= layer_idx < num_layers):
                    problems.append(
                        f"layer {layer_idx} out of range [0, {num_layers}) "
                        f"at pos {r.token_pos}"
                    )
                for e in lr.experts:
                    if not (0 <= e < num_experts):
                        problems.append(
                            f"expert {e} out of range [0, {num_experts}) "
                            f"at pos {r.token_pos} layer {lid}"
                        )
                if top_k > 0 and len(lr.experts) != top_k:
                    problems.append(
                        f"pos {r.token_pos} layer {lid}: {len(lr.experts)} "
                        f"experts != meta.top_k {top_k}"
                    )
                if lr.weights:
                    if len(lr.weights) != len(lr.experts):
                        problems.append(
                            f"pos {r.token_pos} layer {lid}: weights arity "
                            f"{len(lr.weights)} != experts arity {len(lr.experts)}"
                        )
                    # Top-k softmax masses sum to <= 1 (un-normalized) or ~1
                    # (renormalized). Low-precision backends (fp16/bf16 gates)
                    # can round slightly above 1, so allow a generous but
                    # still meaningful bound that catches real corruption.
                    s = sum(lr.weights)
                    if (
                        s <= 0.0
                        or s > 1.5
                        or any(w < 0 or w > 1.5 or not math.isfinite(w) for w in lr.weights)
                    ):
                        problems.append(
                            f"pos {r.token_pos} layer {lid}: weights sum "
                            f"{s:.4f} outside (0, 1.5] or non-finite/out-of-range weight"
                        )

        if self.meta.prompt_len + self.meta.generated_len != total_tokens:
            problems.append(
                "meta.prompt_len + meta.generated_len != meta.total_tokens"
            )
        if len(self.prompt_tokens) != self.meta.prompt_len:
            problems.append(
                f"{len(self.prompt_tokens)} prompt token ids != meta.prompt_len"
            )
        if len(self.generated_tokens) != self.meta.generated_len:
            problems.append(
                f"{len(self.generated_tokens)} generated token ids != meta.generated_len"
            )

        if self.placement is not None:
            _validate_placement(self.placement, self.meta, problems)

        if problems:
            raise ValueError(
                "RoutingTrace validation failed:\n  - "
                + "\n  - ".join(problems[:30])
            )

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
