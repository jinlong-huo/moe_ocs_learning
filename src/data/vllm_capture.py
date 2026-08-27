"""
vllm_capture.py — vLLM MoE routing capture + steering.

Delegates prefill + decode to vLLM's engine (no manual generation loop),
instruments the MoE router gate to (a) optionally steer expert selection and
(b) log per-token, per-layer expert routing decisions, then builds the
canonical ``RoutingTrace`` (same format as the MLX and HF backends).

NOTE: This module requires PyTorch + vLLM. vLLM is CUDA-only; on Apple
Silicon the installed CPU build is used for smoke testing only. It is NOT
required for the OCS simulation — only for capturing routing traces from a
vLLM serving run.
"""

from __future__ import annotations

import contextvars

import torch
import torch.nn as nn

from src.data.routing_schema import RoutingTrace, TokenRoute, LayerRoute, RunMeta

# Absolute token positions for the *current* decoder-layer forward. Set by the
# patched decoder layer, read by the patched MoE gate (token i of the MoE
# block's flattened hidden_states maps 1:1 to positions[i]).
_positions_var: contextvars.ContextVar = contextvars.ContextVar(
    "vllm_positions", default=None
)


# ═══════════════════════════════════════════════════════════════════
# Steering config
# ═══════════════════════════════════════════════════════════════════

class VllmSteering:
    """Router-steering config applied to ``router_logits`` inside the gate hook.

    Mirrors ``RouterSteering`` (src/data/routing_interventions.py) but is a
    plain config object — the actual patching lives in
    ``install_vllm_hooks`` and is active for the whole ``LLM.generate`` call.
    """

    def __init__(self):
        self._interventions: dict[int, dict] = {}

    def force_expert(self, layer: int, expert_id: int, exclusive: bool = False) -> None:
        cfg = self._interventions.setdefault(layer, {})
        cfg.setdefault("force", {})[expert_id] = exclusive

    def bias_expert(self, layer: int, expert_id: int, bias: float) -> None:
        cfg = self._interventions.setdefault(layer, {})
        cfg.setdefault("biases", {})[expert_id] = (
            cfg.setdefault("biases", {}).get(expert_id, 0.0) + bias
        )

    def ablate_expert(self, layer: int, expert_id: int) -> None:
        cfg = self._interventions.setdefault(layer, {})
        cfg.setdefault("ablate", set()).add(expert_id)

    @property
    def active_layers(self) -> list[int]:
        return sorted(self._interventions.keys())

    def has(self, layer: int) -> bool:
        return layer in self._interventions

    def _apply(self, layer_idx: int, router_logits: torch.Tensor) -> torch.Tensor:
        """Return ``router_logits`` with force/bias/ablate applied (dtype-preserving)."""
        cfg = self._interventions.get(layer_idx)
        if not cfg:
            return router_logits

        orig_dtype = router_logits.dtype
        logits = router_logits.to(torch.float32)

        for eid, bias in cfg.get("biases", {}).items():
            logits[:, eid] = logits[:, eid] + bias

        for eid in cfg.get("ablate", set()):
            logits[:, eid] = float("-inf")

        force = cfg.get("force", {})
        if force:
            for eid, exclusive in force.items():
                if exclusive:
                    logits[:, :] = float("-inf")
                logits[:, eid] = 100.0

        return logits.to(orig_dtype)


# ═══════════════════════════════════════════════════════════════════
# Capture
# ═══════════════════════════════════════════════════════════════════

class VllmRoutingCapture:
    """Collects per-token, per-layer expert routing decisions during vLLM run.

    Keyed by absolute token position (vLLM supplies ``positions`` directly, so
    no rolling ``seq_pos`` bookkeeping is needed).
    """

    def __init__(self):
        self._routes: dict[int, dict] = {}  # abs_pos → {"layers": {layer_idx: {experts, weights}}}
        self._expert_load: dict[int, int] = {}
        self._moe_layers: list[int] = []
        self.top_k: int | None = None

    # ── called from the patched MoE gate ───────────────────────────

    def log(
        self,
        layer_idx: int,
        positions: torch.Tensor,
        expert_ids: torch.Tensor,
        weight_vals: torch.Tensor,
    ) -> None:
        pos_list = positions.tolist() if torch.is_tensor(positions) else list(positions)
        ids = (
            expert_ids.detach().cpu().tolist()
            if torch.is_tensor(expert_ids)
            else [list(r) for r in expert_ids]
        )
        ws = (
            weight_vals.detach().cpu().tolist()
            if torch.is_tensor(weight_vals)
            else [list(r) for r in weight_vals]
        )

        for i, pos in enumerate(pos_list):
            abs_pos = int(pos)
            if abs_pos not in self._routes:
                self._routes[abs_pos] = {"layers": {}}
            self._routes[abs_pos]["layers"][layer_idx] = {
                "experts": [int(e) for e in ids[i]],
                "weights": [float(w) for w in ws[i]],
            }
            for e in ids[i]:
                self._expert_load[int(e)] = self._expert_load.get(int(e), 0) + 1

    def _register_moe_layer(self, layer_idx: int) -> None:
        if layer_idx not in self._moe_layers:
            self._moe_layers.append(layer_idx)

    # ── called after generation ────────────────────────────────────

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
        backend: str = "vllm",
    ) -> RoutingTrace:
        all_tokens = prompt_tokens + generated_tokens
        prompt_len = len(prompt_tokens)

        moe_layer_indices = sorted(self._moe_layers)

        route_objs: list[TokenRoute] = []
        for pos in sorted(self._routes.keys()): # layers 0 - 40; length of layers is determined by the number of moe length; len 51 last token position = prompt_len + generate_len without counting the last token "EOS"
            info = self._routes[pos]
            tid = all_tokens[pos] if 0 <= pos < len(all_tokens) else -1
            tok_str = _decode(tokenizer, [tid]) if tid >= 0 else ""
            phase = "prefill" if pos < prompt_len else "decode"
            layer_routes = {
                str(lid): LayerRoute(experts=lr["experts"], weights=lr["weights"])
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
        return dict(sorted(self._expert_load.items()))

    @property
    def route_count(self) -> int:
        return sum(len(info["layers"]) for info in self._routes.values())

    @property
    def moe_layer_indices(self) -> list[int]:
        return sorted(self._moe_layers)


def _decode(tokenizer, ids: list[int]) -> str:
    """Best-effort token decode across vLLM tokenizer-group / plain tokenizers."""
    for src in (tokenizer, getattr(tokenizer, "tokenizer", None),
                getattr(tokenizer, "get_lora_tokenizer", None)):
        if src is None:
            continue
        if callable(src) and not hasattr(src, "decode"):
            src = src()
        decode = getattr(src, "decode", None)
        if callable(decode):
            try:
                return str(decode(list(ids)))
            except Exception:
                continue
    return ""


# ═══════════════════════════════════════════════════════════════════
# Hooks
# ═══════════════════════════════════════════════════════════════════

def _is_moe_block(mlp) -> bool:
    return (
        hasattr(mlp, "gate")
        and hasattr(mlp, "experts")
        and hasattr(getattr(mlp, "experts"), "top_k")
    )


def _get_top_k(block) -> int:
    experts = getattr(block, "experts", None)
    return (
        getattr(experts, "top_k", None)
        or getattr(block, "top_k", None)
        or getattr(block, "num_experts_per_tok", None)
        or 1
    )


def _make_patched_gate_forward(block, capture: VllmRoutingCapture, steering: VllmSteering | None):
    """Patch ``block.gate.forward`` to steer + capture router logits in-place.

    The MoE block's own forward already does ``router_logits, _ = self.gate(x)``
    then ``self.experts(x, router_logits)`` — so steering the logits the gate
    returns is sufficient to change real expert selection, and the logits are
    also exactly what we need for the routing trace.
    """
    orig = block.gate.forward  # bound method, captured before reassignment
    layer_idx = block._layer_idx
    top_k = _get_top_k(block)

    def patched(hidden_states, *args, **kwargs):
        out = orig(hidden_states, *args, **kwargs)
        router_logits = out[0] if isinstance(out, (tuple, list)) else out

        # ── Steer ──
        if steering is not None and steering.has(layer_idx):
            router_logits = steering._apply(layer_idx, router_logits)

        # ── Capture: softmax-over-all → top-k. Matches the HF capture in
        #    routing_capture.py (weights are the raw top-k softmax mass, NOT
        #    renormalized to sum 1). Expert ids are identical to the model's
        #    own selection since softmax is monotonic.
        positions = _positions_var.get()
        if positions is not None:
            probs = torch.softmax(router_logits.to(torch.float32), dim=-1)
            topk_w, topk_idx = torch.topk(probs, k=top_k, dim=-1)
            capture.log(layer_idx, positions, topk_idx, topk_w)

        # ── Return steered logits in the original tuple shape ──
        if isinstance(out, (tuple, list)):
            return (router_logits, *out[1:])
        return router_logits

    return patched


def _make_patched_layer_forward(layer):
    """Patch a decoder layer's forward to publish its ``positions`` tensor."""
    orig = layer.forward  # bound method

    def patched(*args, **kwargs):
        positions = kwargs.get("positions", args[0] if args else None)
        token = _positions_var.set(positions)
        try:
            return orig(*args, **kwargs)
        finally:
            _positions_var.reset(token)

    return patched


def install_vllm_hooks(
    model: nn.Module,
    capture: VllmRoutingCapture,
    steering: VllmSteering | None = None,
) -> int:
    """Tag MoE blocks and patch their gates + enclosing decoder layers.

    Returns the number of patched MoE layers.
    """
    from src.data.model_utils import _find_decoder_layers

    layers = _find_decoder_layers(model)
    patched = 0

    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not _is_moe_block(mlp):
            continue

        mlp._layer_idx = layer_idx
        mlp.gate.forward = _make_patched_gate_forward(mlp, capture, steering)
        layer.forward = _make_patched_layer_forward(layer)

        if capture.top_k is None:
            capture.top_k = _get_top_k(mlp)
        capture._register_moe_layer(layer_idx)
        patched += 1

    print(f"[hook] Installed routing hooks on {patched} MoE layers")
    return patched


# ═══════════════════════════════════════════════════════════════════
# Engine introspection
# ═══════════════════════════════════════════════════════════════════

def locate_model(llm) -> nn.Module | None:
    """Return the causal-LM model from an ``LLM`` instance.

    Reaches through vLLM's executor/worker/model_runner structure, trying the
    known attribute chains for both the v0 and v1 engines. With
    ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` (required for hook installation) the
    V1 executor runs in-process, so the model is reachable directly.

    On the vllm-metal backend the returned object is the raw MLX model
    (``mlx.nn.Module``, e.g. ``mlx_lm.models.qwen3_5_moe.Model``), not a torch
    module.
    """
    engine = getattr(llm, "llm_engine", None)
    executor = getattr(engine, "model_executor", None)

    if executor is not None:
        for chain in (
            ("driver_worker", "model_runner", "model"),
            ("driver_worker", "worker", "model_runner", "model"),
            ("workers", 0, "model_runner", "model"),
        ):
            obj = executor
            for attr in chain:
                try:
                    obj = obj[attr] if isinstance(attr, int) else getattr(obj, attr)
                except (AttributeError, IndexError, KeyError, TypeError):
                    obj = None
                    break
            if isinstance(obj, nn.Module) or _is_mlx_module(obj):
                return obj

    # Fallback: bounded BFS over the engine for a *ForCausalLM module.
    return _bfs_find_model(llm)


def _is_mlx_module(obj) -> bool:
    try:
        import mlx.nn as mlx_nn

        return isinstance(obj, mlx_nn.Module)
    except ImportError:
        return False


def _bfs_find_model(root, max_nodes: int = 2000) -> nn.Module | None:
    seen: set[int] = set()
    stack = [root]
    while stack and len(seen) < max_nodes:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, nn.Module) and node.__class__.__name__.endswith("ForCausalLM"):
            return node
        if _is_mlx_module(node) and node.__class__.__module__.startswith("mlx_lm"):
            return node
        for name in ("model", "model_executor", "driver_worker", "worker",
                     "model_runner", "lm_head", "layers"):
            child = getattr(node, name, None)
            if child is not None and id(child) not in seen:
                stack.append(child)
    return None


def get_vllm_layout(llm, capture: VllmRoutingCapture) -> dict:
    """Extract model metadata for the routing trace from the vLLM engine."""
    hf_cfg = llm.llm_engine.vllm_config.model_config.hf_config
    text_cfg = getattr(hf_cfg, "text_config", None)

    def _cfg(attr: str, default=0):
        v = getattr(hf_cfg, attr, None)
        if v in (None, 0) and text_cfg is not None:
            v = getattr(text_cfg, attr, None)
        return v or default

    num_experts = int(
        _cfg("num_experts") or _cfg("num_local_experts") or 0
    )
    top_k = (
        _cfg("num_experts_per_tok")
        or _cfg("top_k")
        or capture.top_k
        or 0
    )

    return {
        "model_type": _cfg("model_type", "unknown"),
        "num_layers": int(_cfg("num_hidden_layers")),
        "num_experts": num_experts,
        "top_k": int(top_k),
    }


# ═══════════════════════════════════════════════════════════════════
# vllm-metal hooks (MLX-backed engine)
# ═══════════════════════════════════════════════════════════════════
#
# vllm-metal executes the raw MLX model (mlx_lm), not torch modules, so the
# torch hooks above cannot fire. These hooks monkey-patch the MLX MoE block
# instead — the same technique as the MLX backend in moe_run.py.
#
# Positions: with a single request and in-order execution (no prefix-cache
# reuse, chunked prefill only for large prompts), every token passes through
# the MoE exactly once, so a monotonic token counter equals the absolute
# sequence position.

_METAL_OFFSET = [0]  # absolute position of the first token of the current call
_METAL_IN_CALL = [False]  # reentrancy guard for nested model wrappers
_METAL_ORIG_MOE_CALL = None


def _patch_metal_text_model(text_model) -> None:
    """Patch the MLX text model class so MoE hooks see correct absolute positions.

    Special methods (``__call__``) resolve on the class, so the patch must be
    class-level. A reentrancy guard makes nested wrapper calls count once.
    """
    cls = type(text_model)
    if getattr(cls, "_moe_pos_patched", False):
        return
    orig = cls.__call__

    def patched_call(self, *args, **kwargs):
        global _METAL_OFFSET
        if _METAL_IN_CALL[0]:
            return orig(self, *args, **kwargs)
        _METAL_IN_CALL[0] = True
        _METAL_OFFSET.append(_METAL_OFFSET[0])
        try:
            return orig(self, *args, **kwargs)
        finally:
            _METAL_OFFSET[0] += _metal_batch_len(args, kwargs)
            _METAL_IN_CALL[0] = False

    cls.__call__ = patched_call
    cls._moe_pos_patched = True


def _metal_batch_len(args, kwargs) -> int:
    v = args[0] if args else None
    if v is None:
        v = kwargs.get("inputs") or kwargs.get("input_embeddings")
    if v is None:
        return 0
    if hasattr(v, "size"):
        return int(v.size)
    return int(len(v))


def _apply_steering_mx(steering, layer_idx: int, gates):
    """Steer MLX router logits (mirror of ``VllmSteering._apply`` for torch)."""
    import mlx.core as mx

    cfg = steering._interventions.get(layer_idx)
    if not cfg:
        return gates

    logits = gates.astype(mx.float32)

    for eid, bias in cfg.get("biases", {}).items():
        logits[..., eid] = logits[..., eid] + bias

    for eid in cfg.get("ablate", set()):
        logits[..., eid] = float("-inf")

    force = cfg.get("force", {})
    if force:
        for eid, exclusive in force.items():
            if exclusive:
                logits = mx.full_like(logits, float("-inf"))
            logits[..., eid] = 100.0

    return logits.astype(gates.dtype)


def _make_metal_moe_call(capture: VllmRoutingCapture, steering):
    """Return a replacement ``__call__`` for the MLX sparse-MoE block."""
    import mlx.core as mx

    def patched_call(self, x):
        layer_idx = self._layer_idx

        gates = self.gate(x)
        if steering is not None and steering.has(layer_idx):
            gates = _apply_steering_mx(steering, layer_idx, gates)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if getattr(self, "norm_topk_prob", False):
            # qwen2_moe-style blocks lack norm_topk_prob — skip renormalization.
            scores = scores / scores.sum(axis=-1, keepdims=True)

        offset = _METAL_OFFSET[-1]
        if inds.ndim == 3:
            # vllm-metal keeps a batch dim: [B=1, T, k]
            flat_inds = inds[0]
            flat_scores = scores[0]
            num_tokens = flat_inds.shape[0]
        else:
            flat_inds = inds
            flat_scores = scores
            num_tokens = flat_inds.shape[0]
        capture.log(
            layer_idx,
            [offset + t for t in range(num_tokens)],
            flat_inds.tolist(),
            flat_scores.tolist(),
        )

        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

        return y + shared_y

    return patched_call


def install_vllm_metal_hooks(
    model,
    capture: VllmRoutingCapture,
    steering: VllmSteering | None = None,
) -> int:
    """Tag MLX MoE blocks and patch their ``__call__`` for routing capture.

    Returns the number of patched MoE layers.
    """
    global _METAL_ORIG_MOE_CALL

    text_model = getattr(model, "model", None) or getattr(model, "language_model", None)
    layers = getattr(text_model, "layers", None)
    if layers is None:
        layers = getattr(getattr(text_model, "model", None), "layers", None)
    if layers is None:
        print("[hook] [metal] Could not find MLX text model layers")
        return 0

    patched = 0
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
        capture._register_moe_layer(layer_idx)
        patched += 1

    if moe_cls is not None and patched > 0:
        _METAL_ORIG_MOE_CALL = moe_cls.__call__
        moe_cls.__call__ = _make_metal_moe_call(capture, steering)

        # Position tracking: vllm-metal calls ``model.language_model``, which
        # delegates to the inner text model. Patch every candidate wrapper so
        # the token counter advances exactly once per forward (reentrancy
        # guard inside the patched call).
        seen: set[int] = set()
        candidates = (
            model,
            getattr(model, "model", None),
            getattr(model, "language_model", None),
            getattr(getattr(model, "model", None), "model", None),
            getattr(getattr(model, "language_model", None), "model", None),
        )
        for cand in candidates:
            if cand is None or id(cand) in seen:
                continue
            seen.add(id(cand))
            if hasattr(cand, "layers") or hasattr(getattr(cand, "model", None), "layers"):
                _patch_metal_text_model(cand)

    print(f"[hook] [metal] Installed routing hooks on {patched} MoE layers")
    return patched


def restore_vllm_metal_hooks(model) -> None:
    """Restore the original MLX MoE ``__call__`` (class-level)."""
    global _METAL_ORIG_MOE_CALL
    if _METAL_ORIG_MOE_CALL is None:
        return
    text_model = getattr(model, "model", None) or getattr(model, "language_model", None)
    layers = getattr(text_model, "layers", None) or []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp"):
            type(mlp).__call__ = _METAL_ORIG_MOE_CALL
            break
    _METAL_ORIG_MOE_CALL = None
