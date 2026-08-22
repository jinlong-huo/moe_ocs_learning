"""
capture.py — Multi-tenant routing capture for the vllm-metal engine.

The vllm-metal V1 engine batches tokens from several concurrent requests
into every forward pass (decode-first, then prefill chunks).  This module
attributes every token back to its (tenant, position-in-sequence) pair:

  * a contextvar ``_TOKEN_MAP`` is published around each engine forward by
    thin wrappers on the metal runner (``_start_paged_forward`` for the
    paged-attention path, ``_batched_decode`` / ``_sequential_decode`` /
    ``_prefill_single`` for the legacy non-paged path), and
  * the MLX sparse-MoE block ``__call__`` is patched (class-level) to log
    per-token, per-layer expert routing into the active tenant's trace.

The capture is pure instrumentation: the patched MoE call re-runs the gate
math for logging and then delegates to the original ``__call__``, so the
engine's computation is untouched.
"""

from __future__ import annotations

import contextvars
import time
from collections import Counter
from typing import Optional

# Per-step token attribution: flat list of (request_id, position) aligned
# with the token rows of the current forward's hidden states.
_TOKEN_MAP: contextvars.ContextVar = contextvars.ContextVar(
    "mt_token_map", default=None
)

# Legacy non-paged prefill: request ids pushed by the ``_handle_new_requests``
# wrapper, popped by the ``_prefill_single`` wrapper (the original call site
# does not pass the request id down).
_PREFILL_REQ_STACK: list[str] = []


class MultiTenantCapture:
    """Collects per-tenant, per-position, per-layer routing during serving.

    One instance lives for the whole session.  Request ids come from the
    engine (``LLMEngine.add_request``), so attribution is exact even when
    several tenants share one forward pass.
    """

    def __init__(self):
        # req_id → position → {"layers": {layer_idx: {"experts": [...], "weights": [...]}}}
        self._routes: dict[str, dict[int, dict]] = {}
        self._expert_load: dict[int, int] = Counter()
        self._moe_layers: list[int] = []
        self.top_k: Optional[int] = None

        self._steps: list[dict] = []
        self._step_counter = 0
        self._cur_step: Optional[dict] = None  # being accumulated

    # ── step bookkeeping (called from runner wrappers) ──────────────

    def begin_step(self) -> None:
        """Start a new engine-step record (called before each model forward).

        The previous step's duration is measured here — the wall time
        between consecutive forward submissions — so each record carries
        the real engine step period under multi-tenant load.
        """
        now = time.perf_counter()
        if self._cur_step is not None:
            self._cur_step["duration_s"] = max(now - self._cur_step["t_s"], 0.0)
            self._steps.append(self._cur_step)
            self._step_counter += 1
        self._cur_step = {
            "step": self._step_counter,
            "t_s": now,
            "tokens": {},
        }

    def set_token_map(self, token_map: list[tuple[str, int]]) -> None:
        """Publish the current forward's token attribution."""
        _TOKEN_MAP.set(token_map)
        if self._cur_step is not None:
            self._cur_step["tokens"] = dict(Counter(r for r, _ in token_map))

    def clear_token_map(self) -> None:
        _TOKEN_MAP.set(None)

    def finalize_steps(self) -> None:
        """Close the last step record and compute its duration."""
        if self._cur_step is not None:
            self._cur_step["duration_s"] = max(
                time.perf_counter() - self._cur_step["t_s"], 0.0
            )
            self._steps.append(self._cur_step)
            self._cur_step = None

    # ── called from the patched MLX MoE forward ─────────────────────

    def log(self, layer_idx: int, batch_experts: list, batch_weights: list) -> None:
        """Record routing for one MoE layer over the current forward's tokens."""
        token_map = _TOKEN_MAP.get()
        if not token_map:
            return

        if layer_idx not in self._moe_layers:
            self._moe_layers.append(layer_idx)

        for t, (req_id, pos) in enumerate(token_map):
            if t >= len(batch_experts):
                break
            experts = [int(e) for e in batch_experts[t]]
            weights = [float(w) for w in batch_weights[t]]
            routes = self._routes.setdefault(req_id, {})
            pos_entry = routes.setdefault(pos, {"layers": {}})
            pos_entry["layers"][layer_idx] = {
                "experts": experts,
                "weights": weights,
            }
            for e in experts:
                self._expert_load[e] += 1

    # ── called after the session ────────────────────────────────────

    def request_ids(self) -> list[str]:
        return sorted(self._routes.keys())

    def get_request_routes(self, req_id: str) -> dict[int, dict]:
        return self._routes.get(req_id, {})

    def build_tenant_trace(
        self,
        req_id: str,
        prompt_tokens: list[int],
        generated_tokens: list[int],
        tokenizer,
        model_id: str,
        layout: dict,
        backend: str = "vllm",
    ):
        """Build the canonical ``RoutingTrace`` for one tenant."""
        from src.data.routing_schema import (
            RoutingTrace,
            TokenRoute,
            LayerRoute,
            RunMeta,
        )

        all_tokens = prompt_tokens + generated_tokens
        prompt_len = len(prompt_tokens)
        pos_routes = self._routes.get(req_id, {})

        route_objs: list[TokenRoute] = []
        for pos in sorted(pos_routes.keys()):
            info = pos_routes[pos]
            tid = all_tokens[pos] if 0 <= pos < len(all_tokens) else -1
            tok_str = ""
            if tid >= 0:
                try:
                    tok_str = str(tokenizer.decode([tid]))
                except Exception:
                    tok_str = ""
            phase = "prefill" if pos < prompt_len else "decode"
            layer_routes = {
                str(lid): LayerRoute(
                    experts=lr["experts"], weights=lr["weights"]
                )
                for lid, lr in info["layers"].items()
            }
            route_objs.append(TokenRoute(
                token_pos=pos,
                token_id=tid,
                token_str=tok_str,
                phase=phase,
                layers=layer_routes,
            ))

        meta = RunMeta(
            model_id=model_id,
            model_type=layout.get("model_type", "unknown"),
            num_layers=int(layout.get("num_layers", 0)),
            num_moe_layers=len(self._moe_layers),
            num_experts=int(layout.get("num_experts", 0)),
            top_k=int(layout.get("top_k", 0) or self.top_k or 0),
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
        return dict(sorted(self._expert_load.items()))

    @property
    def moe_layer_indices(self) -> list[int]:
        return sorted(self._moe_layers)

    @property
    def steps(self) -> list[dict]:
        return self._steps

    @property
    def route_count(self) -> int:
        return sum(
            len(pos_entry["layers"])
            for req_routes in self._routes.values()
            for pos_entry in req_routes.values()
        )


# ═══════════════════════════════════════════════════════════════════
# Engine/model discovery
# ═══════════════════════════════════════════════════════════════════

def _is_mlx_module(obj) -> bool:
    try:
        import mlx.nn as mlx_nn

        return isinstance(obj, mlx_nn.Module)
    except ImportError:
        return False


def locate_metal_runner(engine):
    """Find the ``MetalModelRunner`` instance reachable from a vLLM engine."""
    return _bfs(engine, lambda n: type(n).__name__ == "MetalModelRunner")


def locate_mlx_model(engine):
    """Find the raw MLX causal-LM model reachable from a vLLM engine."""
    return _bfs(
        engine,
        lambda n: _is_mlx_module(n) and n.__class__.__module__.startswith("mlx"),
    )


def _bfs(root, predicate, max_nodes: int = 4000):
    seen: set[int] = set()
    stack = [root]
    while stack and len(seen) < max_nodes:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if predicate(node):
            return node
        for name in (
            "model",
            "model_executor",
            "engine_core",
            "driver_worker",
            "worker",
            "workers",
            "model_runner",
            "scheduler",
            "lm_head",
            "language_model",
            "layers",
        ):
            try:
                child = node[name] if isinstance(node, (list, tuple)) else getattr(node, name, None)
            except (AttributeError, IndexError, KeyError, TypeError):
                continue
            if child is not None and id(child) not in seen:
                stack.append(child)
    return None


# ═══════════════════════════════════════════════════════════════════
# Metal runner wrappers (per-step token attribution)
# ═══════════════════════════════════════════════════════════════════

def _build_paged_token_map(runner, prefill_reqs, decode_reqs, scheduler_output):
    """Reconstruct the exact token order of the paged forward.

    Mirrors ``MetalModelRunner._start_paged_forward``: decode segments
    first (in ``decode_reqs`` order), then prefill chunks (in
    ``prefill_reqs`` order).  Speculative-decode draft rows widen a decode
    segment; without a draft model each segment is exactly one token.
    """
    token_map: list[tuple[str, int]] = []

    spec_tokens = None
    try:
        controller = runner._spec_decode_controller
        spec_tokens = controller.active_spec_decode_tokens(scheduler_output)
    except Exception:
        spec_tokens = None

    for req_id, state in decode_reqs:
        drafts = tuple(spec_tokens.get(req_id, ())) if spec_tokens else ()
        n_query = 1 + len(drafts)
        token_ids = state.token_ids
        cache_start_pos = len(token_ids) - 1 if token_ids else 0
        for t in range(n_query):
            token_map.append((req_id, cache_start_pos + t))

    for pr in prefill_reqs:
        for t in range(len(pr.token_ids)):
            token_map.append((pr.req_id, pr.start_pos + t))

    return token_map


def _wrap_paged_forward(runner, capture: MultiTenantCapture):
    orig = runner._start_paged_forward

    def wrapped(batch, prefill_reqs, decode_reqs, scheduler_output):
        token_map = _build_paged_token_map(
            runner, prefill_reqs, decode_reqs, scheduler_output
        )
        capture.begin_step()
        capture.set_token_map(token_map)
        try:
            return orig(batch, prefill_reqs, decode_reqs, scheduler_output)
        finally:
            capture.clear_token_map()

    runner._start_paged_forward = wrapped


def _wrap_decode_batch(runner, capture: MultiTenantCapture, name: str):
    """Wrap ``_batched_decode`` / ``_sequential_decode`` (non-paged path).

    Each request contributes exactly one token at position len-1.
    """
    orig = getattr(runner, name)

    def wrapped(decode_reqs):
        token_map = []
        for req_id, state in decode_reqs:
            pos = len(state.token_ids) - 1 if state.token_ids else 0
            token_map.append((req_id, pos))
        capture.begin_step()
        capture.set_token_map(token_map)
        try:
            return orig(decode_reqs)
        finally:
            capture.clear_token_map()

    setattr(runner, name, wrapped)


def _wrap_handle_new_requests(runner, capture: MultiTenantCapture):
    """Push request ids for legacy per-request prefills (non-paged path)."""
    orig = runner._handle_new_requests

    def wrapped(batch, new_reqs, scheduler_output):
        if runner._paged_attention_runtime is None:
            for nr in new_reqs:
                if nr.prompt_token_ids:
                    _PREFILL_REQ_STACK.append(nr.req_id)
        try:
            return orig(batch, new_reqs, scheduler_output)
        finally:
            _PREFILL_REQ_STACK.clear()

    runner._handle_new_requests = wrapped


def _wrap_prefill_single(runner, capture: MultiTenantCapture):
    orig = runner._prefill_single

    def wrapped(token_ids, *args, **kwargs):
        req_id = _PREFILL_REQ_STACK.pop(0) if _PREFILL_REQ_STACK else None
        token_map = [(req_id, pos) for pos in range(len(token_ids))] if req_id else []
        capture.begin_step()
        capture.set_token_map(token_map)
        try:
            return orig(token_ids, *args, **kwargs)
        finally:
            capture.clear_token_map()

    runner._prefill_single = wrapped


# ═══════════════════════════════════════════════════════════════════
# MLX MoE block patch (routing capture only)
# ═══════════════════════════════════════════════════════════════════

_MOE_ORIG_CALL = None


def _make_metal_moe_call(capture: MultiTenantCapture):
    """Replacement ``__call__`` that logs routing, then delegates."""
    import mlx.core as mx

    def patched_call(self, x):
        layer_idx = getattr(self, "_layer_idx", -1)

        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = getattr(self, "top_k", 1) or 1
        inds = mx.stop_gradient(
            mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
        )
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if getattr(self, "norm_topk_prob", False):
            scores = scores / scores.sum(axis=-1, keepdims=True)

        if inds.ndim == 3:
            flat_inds, flat_scores = inds[0], scores[0]
        else:
            flat_inds, flat_scores = inds, scores

        capture.log(layer_idx, flat_inds.tolist(), flat_scores.tolist())

        return _MOE_ORIG_CALL(self, x)

    return patched_call


def install_hooks(engine, capture: MultiTenantCapture) -> int:
    """Install runner wrappers + the MLX MoE patch.

    Returns the number of patched MoE layers.
    """
    global _MOE_ORIG_CALL

    runner = locate_metal_runner(engine)
    model = locate_mlx_model(engine)

    patched = 0
    if runner is not None:
        if hasattr(runner, "_start_paged_forward"):
            _wrap_paged_forward(runner, capture)
        if hasattr(runner, "_handle_new_requests"):
            _wrap_handle_new_requests(runner, capture)
        if hasattr(runner, "_prefill_single"):
            _wrap_prefill_single(runner, capture)
        for name in ("_batched_decode", "_sequential_decode"):
            if hasattr(runner, name):
                _wrap_decode_batch(runner, capture, name)

    if model is not None:
        layers = _find_mlx_layers(model)
        moe_cls = None
        for layer_idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not hasattr(mlp, "switch_mlp"):
                continue
            if moe_cls is None:
                moe_cls = type(mlp)
            mlp._layer_idx = layer_idx
            if capture.top_k is None:
                capture.top_k = int(getattr(mlp, "top_k", 1) or 1)
            patched += 1
        if moe_cls is not None and patched > 0:
            _MOE_ORIG_CALL = moe_cls.__call__
            moe_cls.__call__ = _make_metal_moe_call(capture)

    print(f"[hook] Multi-tenant hooks: runner={'ok' if runner else 'missing'}, "
          f"model={'ok' if model else 'missing'}, MoE layers={patched}")
    return patched


def restore_hooks(model=None) -> None:
    """Restore the original MLX MoE ``__call__`` if it was patched."""
    global _MOE_ORIG_CALL
    if _MOE_ORIG_CALL is None or model is None:
        return
    for layer in _find_mlx_layers(model):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp"):
            type(mlp).__call__ = _MOE_ORIG_CALL
            break
    _MOE_ORIG_CALL = None


def _find_mlx_layers(model) -> list:
    """Locate the MLX decoder layers across wrapper nesting."""
    for cand in (
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "model", None), "model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ):
        if cand is None:
            continue
        layers = getattr(cand, "layers", None)
        if layers:
            return layers
    return []
