"""
capture.py — Routing capture coordinator.

Uses ``output_router_logits=True`` (no monkey-patching) to collect
per-layer, per-token expert assignments during generation.

NOTE: This module requires PyTorch + HuggingFace transformers.
It is NOT required for the OCS simulation — only for capturing
new routing traces from real models.
"""

from __future__ import annotations

import torch
from typing import Optional

from src.data.routing_schema import RoutingTrace, TokenRoute, LayerRoute, RunMeta
from src.data.model_utils import ModelLayout


class RoutingCapture:
    """Coordinates routing capture across a single generation run."""

    def __init__(self, layout: ModelLayout, model_id: str, backend: str = "hf"):
        self.layout = layout
        self.model_id = model_id
        self.backend = backend
        self._routes: list[TokenRoute] = []
        self._seq_pos: int = 0
        self._phase: str = "prefill"
        self._captured_token_ids: dict[int, int] = {}

    def begin_prefill(self, prompt_len: int, prompt_ids: list[int]) -> None:
        self._phase = "prefill"
        for i, tid in enumerate(prompt_ids):
            self._captured_token_ids[i] = tid

    def begin_decode_step(self, token_id: int) -> None:
        self._phase = "decode"
        self._captured_token_ids[self._seq_pos] = token_id

    def consume(self, router_logits: tuple[torch.Tensor, ...],
                hidden_states: torch.Tensor) -> None:
        seq_len = hidden_states.shape[1]

        for pos_offset in range(seq_len):
            abs_pos = self._seq_pos + pos_offset
            tid = self._captured_token_ids.get(abs_pos, -1)
            token_route = TokenRoute(
                token_pos=abs_pos,
                token_id=tid,
                token_str="",
                phase=self._phase,
                layers={},
            )

            for layer_idx, logits in enumerate(router_logits):
                row = logits[pos_offset]
                probs = torch.softmax(row, dim=-1)
                _, topk_idx = torch.topk(probs, k=self.layout.top_k, dim=-1)
                topk_w = probs[topk_idx]

                real_layer = self.layout.moe_layer_indices[layer_idx]
                token_route.layers[str(real_layer)] = LayerRoute(
                    experts=topk_idx.cpu().tolist(),
                    weights=topk_w.cpu().tolist(),
                )

            self._routes.append(token_route)

        self._seq_pos += seq_len

    def finalize(self, tokenizer) -> RoutingTrace:
        pos_to_tid: dict[int, int] = {}
        for pos, tid in self._captured_token_ids.items():
            pos_to_tid[pos] = tid
        max_pos = max(pos_to_tid.keys()) if pos_to_tid else -1
        full_ids = [pos_to_tid.get(i, -1) for i in range(max_pos + 1)]

        for route in self._routes:
            pos = route.token_pos
            if 0 <= pos < len(full_ids):
                route.token_id = full_ids[pos]
                route.token_str = tokenizer.decode([full_ids[pos]])
            else:
                route.token_str = "[UNK]"

        prompt_positions: set[int] = set()
        decode_positions: set[int] = set()
        for r in self._routes:
            if r.phase == "prefill":
                prompt_positions.add(r.token_pos)
            else:
                decode_positions.add(r.token_pos)

        prompt_tokens = [pos_to_tid[p] for p in sorted(prompt_positions)]
        generated_tokens = [pos_to_tid[p] for p in sorted(decode_positions)]

        if not prompt_tokens:
            prompt_tokens = list(self._captured_token_ids.values())

        meta = RunMeta(
            model_id=self.model_id,
            model_type=self.layout.model_type,
            num_layers=self.layout.num_layers,
            num_moe_layers=self.layout.num_moe_layers,
            num_experts=self.layout.num_experts,
            top_k=self.layout.top_k,
            prompt_len=len(prompt_tokens),
            generated_len=len(generated_tokens),
            total_tokens=len(full_ids),
            backend=self.backend,
        )

        return RoutingTrace(
            meta=meta,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            routes=self._routes,
        )
