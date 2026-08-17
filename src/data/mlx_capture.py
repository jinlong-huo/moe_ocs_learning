"""
mlx_capture.py — MLX MoE routing capture producing canonical RoutingTrace format.
"""

from __future__ import annotations

from collections import defaultdict


class RoutingCapture:
    """Collects per-token, per-layer expert routing decisions during MLX generation.

    Produces data compatible with ``moe_framework.schema.RoutingTrace``.
    A single ``state`` dict is shared between the generation loop and the
    patched MoE forward so that absolute token positions are always correct.

    Parameters
    ----------
    state : dict
        Mutable dict with keys ``seq_pos`` (int) and ``phase`` (str).
        Updated by the generation loop before/after each forward pass.
    """

    def __init__(self, state: dict):
        self._state = state
        self._routes: dict[int, dict] = {}  # abs_pos → {"phase": str, "layers": {layer_idx: {experts, weights}}}
        self._expert_load: dict[int, int] = defaultdict(int)

    # ── called from the patched MoE forward ─────────────────────────

    def log(self, layer_id: int, batch_token_experts: list, batch_token_weights: list):
        """Record routing for one MoE layer across all tokens in this forward pass.

        Parameters
        ----------
        layer_id : int
        batch_token_experts : list[list[int]]   shape [seq_len, top_k]
        batch_token_weights : list[list[float]]  shape [seq_len, top_k]
        """
        base_pos = self._state["seq_pos"]
        phase = self._state["phase"]
        seq_len = len(batch_token_experts)

        for t in range(seq_len):
            abs_pos = base_pos + t
            if abs_pos not in self._routes:
                self._routes[abs_pos] = {"phase": phase, "layers": {}}
            existing = self._routes[abs_pos]
            existing["layers"][layer_id] = {
                "experts": [int(e) for e in batch_token_experts[t]],
                "weights": [float(w) for w in batch_token_weights[t]],
            }
            for e in batch_token_experts[t]:
                self._expert_load[int(e)] += 1

    # ── called after generation ──────────────────────────────────────

    def build_trace(
        self,
        prompt_tokens: list[int],
        generated_tokens: list[int],
        tokenizer,
        model_id: str = "",
        model_type: str = "",
        num_layers: int = 0,
        num_experts: int = 0,
        top_k: int = 0,
        backend: str = "mlx",
    ):
        """Build a ``RoutingTrace`` ready for JSON serialization."""
        from src.data.routing_schema import RoutingTrace, TokenRoute, LayerRoute, RunMeta

        all_tokens = prompt_tokens + generated_tokens
        prompt_len = len(prompt_tokens)

        moe_layer_indices = sorted(set(
            lid for info in self._routes.values() for lid in info["layers"]
        ))

        route_objs: list[TokenRoute] = []
        for pos in sorted(self._routes.keys()):
            info = self._routes[pos]
            tid = all_tokens[pos] if 0 <= pos < len(all_tokens) else -1
            tok_str = tokenizer.decode([tid]) if tid >= 0 else ""
            phase = "prefill" if pos < prompt_len else "decode"
            layer_routes = {}
            for lid, lr in info["layers"].items():
                layer_routes[str(lid)] = LayerRoute(
                    experts=lr["experts"], weights=lr["weights"]
                )
            route_objs.append(TokenRoute(
                token_pos=pos,
                token_id=tid,
                token_str=tok_str,
                phase=phase,
                layers=layer_routes,
            ))

        meta = RunMeta(
            model_id=model_id,
            model_type=model_type,
            num_layers=num_layers,
            num_moe_layers=len(moe_layer_indices),
            num_experts=num_experts,
            top_k=top_k,
            prompt_len=prompt_len,
            generated_len=len(generated_tokens),
            total_tokens=len(all_tokens),
            backend=backend,
        )

        return RoutingTrace(
            meta=meta,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            routes=route_objs,
        )

    @property
    def expert_load(self) -> dict[int, int]:
        return dict(self._expert_load)

    @property
    def route_count(self) -> int:
        return sum(len(info["layers"]) for info in self._routes.values())
