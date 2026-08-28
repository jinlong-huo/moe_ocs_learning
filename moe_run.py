"""
moe_run.py — MoE inference with per-layer, per-token routing capture (MLX backend).

Loads a Qwen-MoE model via mlx_lm, instruments MoE blocks to log gate
decisions, runs step-by-step generation, and saves a canonical
RoutingTrace JSON (shared schema across all capture backends).

Usage:
    python moe_run.py
        Uses Qwen3.6-35B-A3B-4bit (MoE, top-4) by default.

    python moe_run.py --model ./models/Qwen3.6-35B-A3B-4bit

Output:
    logs/routing.json — schema-compatible routing trace (same format as HF backend)
"""

import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache


def _load_with_fallback(model_path: str, adapter_path: str | None = None):
    """Load via mlx_lm; shim unknown model_type to a known module on failure.

    E.g. Hy3 (model_type ``hy_v3``) maps to ``mlx_lm.models.hunyuan`` in
    mlx_lm versions that predate a dedicated hy_v3 module.
    """
    try:
        return load(model_path, adapter_path=adapter_path)
    except Exception as exc:
        msg = str(exc)
        if "not supported" not in msg and "No module named" not in msg:
            raise
        import mlx_lm.utils as _u

        _u.MODEL_REMAPPING.setdefault("hy_v3", "hunyuan")
        _u.MODEL_REMAPPING.setdefault("hy_v3_moe", "hunyuan")
        print("[load] model_type unsupported — shimmed via MODEL_REMAPPING")
        return load(model_path, adapter_path=adapter_path)


# ═══════════════════════════════════════════════════════════════════
# Guide model — dense generalised model for affinity graph prior
# ═══════════════════════════════════════════════════════════════════
_GUIDE_HIDDEN = None

# ═══════════════════════════════════════════════════════════════════ 
# Monkey-patch: instrument MoE block forward pass
# ═══════════════════════════════════════════════════════════════════

_MOE_ORIG_CALL = None


def _get_layers(model):
    """Get decoder layers from model, handling both flat and nested structures."""
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    raise AttributeError("Cannot find decoder layers on model")


def _make_patched_call(capture):
    """Return a replacement ``__call__`` that logs routing then delegates."""

    def patched_call(self, x):
        layer_idx = self._layer_idx

        # ── Run original gate logic ──
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.stop_gradient(
            mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
        )
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if getattr(self, "norm_topk_prob", False):
            scores = scores / scores.sum(axis=-1, keepdims=True)

        # ── Log routing (batch=1 in MLX generation, strip dim) ──
        capture.log(
            layer_id=layer_idx,
            batch_token_experts=inds.tolist()[0],   # [seq_len, top_k]
            batch_token_weights=scores.tolist()[0],  # [seq_len, top_k]
        )

        # ── Route through SwitchGLU experts + shared expert ──
        if getattr(self, "use_shared_mlp", False):
            # Hy3 / hunyuan path (mirrors mlx_lm hunyuan.MoeBlock.__call__)
            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None].astype(mx.float32)).sum(axis=-2).astype(y.dtype)
            y = y + self.shared_mlp(x)
        else:
            # Qwen path (unchanged)
            y = self.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2)
            y = y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return y

    return patched_call


def install_routing_hooks(model, capture):
    """Tag each MoE block with its layer index and install class-level hook."""
    global _MOE_ORIG_CALL

    patched = 0
    moe_cls = None
    layers = _get_layers(model)

    for layer_idx, layer in enumerate(layers):
        moe_block = getattr(layer, "mlp", None)
        if moe_block is None or not hasattr(moe_block, "switch_mlp"):
            continue

        if moe_cls is None:
            moe_cls = type(moe_block)
            _MOE_ORIG_CALL = moe_cls.__call__

        moe_block._layer_idx = layer_idx
        patched += 1

    if moe_cls is not None and patched > 0:
        moe_cls.__call__ = _make_patched_call(capture)

    print(f"[hook] Installed routing hooks on {patched} MoE layers")
    return patched


def _patch_guide_for_hidden_states(model):
    """Patch the guide model so its next forward captures last-layer hidden states."""
    global _GUIDE_HIDDEN
    if hasattr(model, "lm_head"):
        _orig = model.lm_head
        def _capture(x):
            global _GUIDE_HIDDEN
            _GUIDE_HIDDEN = x
            return _orig(x)
        model.lm_head = _capture
        return True
    layers = _get_layers(model)
    if layers is not None and len(layers) > 0:
        last = layers[-1]
        for attr in ("norm", "final_layer_norm", "ln_f", "output_norm"):
            norm = getattr(last, attr, None) or getattr(model, attr, None)
            if norm is not None:
                _orig_norm = norm
                def _capture_norm(x):
                    global _GUIDE_HIDDEN
                    out = _orig_norm(x)
                    _GUIDE_HIDDEN = out
                    return out
                if hasattr(last, attr):
                    setattr(last, attr, _capture_norm)
                else:
                    setattr(model, attr, _capture_norm)
                return True
    return False


def _compute_guide_affinity():
    """Compute token cosine-similarity matrix from captured guide hidden states."""
    global _GUIDE_HIDDEN
    if _GUIDE_HIDDEN is None:
        return None
    h = _GUIDE_HIDDEN
    if h.ndim == 3:
        h = h[0]
    norms = mx.linalg.norm(h, axis=-1, keepdims=True)
    norms = mx.where(norms == 0, mx.ones_like(norms), norms)
    h_norm = h / norms
    affinity = h_norm @ h_norm.T
    return affinity.tolist()


# ═══════════════════════════════════════════════════════════════════
# Model metadata extraction
# ═══════════════════════════════════════════════════════════════════

def _extract_model_meta(model, model_path: str) -> dict:
    """Extract model metadata from an MLX model for the routing trace."""
    layers = _get_layers(model)
    total_layers = len(layers)

    num_experts = 0
    top_k = 0
    moe_count = 0
    model_type = "unknown"

    for layer in layers:
        moe_block = getattr(layer, "mlp", None)
        if moe_block is not None and hasattr(moe_block, "switch_mlp"):
            moe_count += 1
            if num_experts == 0:
                num_experts = getattr(moe_block, "num_experts", 0) or 0
                top_k = getattr(moe_block, "top_k", 0) or 0

    # Fallback: infer from config if available
    config = getattr(model, "config", None)
    if config is not None:
        if num_experts == 0:
            num_experts = getattr(config, "num_experts", 0) or 0
        if top_k == 0:
            top_k = getattr(config, "num_experts_per_tok", 0) or getattr(config, "top_k", 0) or 0
        model_type = getattr(config, "model_type", model_type)

    # Fallback: guess from path
    if model_type == "unknown":
        path_lower = model_path.lower()
        if "qwen3" in path_lower or "qwen3.5" in path_lower:
            model_type = "qwen3_moe"
        elif "qwen2" in path_lower:
            model_type = "qwen2_moe"

    model_id = Path(model_path).name

    return {
        "model_id": model_id,
        "model_type": model_type,
        "num_layers": total_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "moe_count": moe_count,
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MoE inference with routing trace capture (MLX)"
    )
    parser.add_argument(
        "--model", type=str,
        default="./models/Qwen3.6-35B-A3B-4bit",
        help="Path to mlx-format model directory",
    )
    parser.add_argument(
        "--guide-model", type=str,
        default="./models/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
        help="Path to dense generalised model for affinity graph prior",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="Explain why Mixture of Experts models need routing, in one paragraph.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter weights")
    parser.add_argument("--no-chat", action="store_true",
                        help="Disable chat template")
    parser.add_argument("--system-prompt", type=str,
                        default="You are a helpful assistant.")
    args = parser.parse_args()

    # ── Load model ──
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[error] Model not found: {model_path}")
        return 1

    model, tokenizer = _load_with_fallback(str(model_path), adapter_path=args.adapter_path)
    print(f"[load] Model: {args.model}")
    if args.adapter_path:
        print(f"[load] Adapter: {args.adapter_path}")

    # ── Extract metadata ──
    meta = _extract_model_meta(model, args.model)
    print(f"[model] {meta['num_layers']} decoder layers, "
          f"{meta['moe_count']} MoE layers, "
          f"{meta['num_experts']} experts, top_k={meta['top_k']}")

    # ── Shared state between generation loop and hooks ──
    state = {"seq_pos": 0, "phase": "prefill"}

    # ── Install routing hooks ──
    from src.data.mlx_capture import RoutingCapture

    capture = RoutingCapture(state)
    install_routing_hooks(model, capture)

    # ── Tokenize ──
    use_chat = (
        not args.no_chat
        and hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    )
    if use_chat:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_tokens = tokenizer.encode(prompt_text)
        print("[input] Chat template applied")
    else:
        prompt_tokens = tokenizer.encode(args.prompt)

    print(f"[input] Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"[input] Prompt tokens: {len(prompt_tokens)}")

    # ── Set up KV cache ──
    prompt_cache = make_prompt_cache(model)
    prompt_array = mx.array([prompt_tokens])
    generated_tokens: list[int] = []

    # ── Prefill forward pass ──
    state["phase"] = "prefill"
    logits = model(prompt_array, cache=prompt_cache)
    logits = logits[:, -1, :]
    state["seq_pos"] += len(prompt_tokens)

    # ── Generation loop ──
    token = prompt_array
    for step in range(args.max_tokens):
        state["phase"] = "decode"

        # Greedy decoding (--temp 0 -> argmax) for deterministic traces
        if args.temp <= 0:
            next_token_id = int(mx.argmax(logits, axis=-1).item())
            next_token = mx.array([next_token_id])
        else:
            next_token = mx.random.categorical(logits / args.temp)
            next_token_id = int(next_token.item())
        generated_tokens.append(next_token_id)

        decoded = tokenizer.decode([next_token_id])
        print(decoded, end="", flush=True)

        if next_token_id == tokenizer.eos_token_id:
            print()
            break

        # Forward pass for the new token
        token = next_token
        logits = model(token[None], cache=prompt_cache)
        logits = logits[:, -1, :]
        state["seq_pos"] += 1
    else:
        print()

    print(f"\n[summary] Generated {len(generated_tokens)} tokens")
    print(f"[summary] Routing events logged: {capture.route_count}")

    # ── Compute guide affinity from dense generalised model ──
    guide_affinity = None
    if args.guide_model:
        guide_path = Path(args.guide_model)
        if guide_path.exists():
            print(f"[guide] Loading generalised model: {guide_path.name}")
            guide_model, guide_tokenizer = load(str(guide_path))
            patched_ok = _patch_guide_for_hidden_states(guide_model)
            if patched_ok:
                if use_chat and hasattr(guide_tokenizer, "apply_chat_template") \
                        and guide_tokenizer.chat_template is not None:
                    guide_prompt = guide_tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False,
                    )
                    guide_ids = guide_tokenizer.encode(guide_prompt)
                else:
                    guide_ids = guide_tokenizer.encode(args.prompt)
                try:
                    _ = guide_model(mx.array([guide_ids]))
                    guide_affinity = _compute_guide_affinity()
                    if guide_affinity is not None:
                        aff_rows = len(guide_affinity)
                        aff_cols = len(guide_affinity[0]) if guide_affinity else 0
                        print(f"[guide] Affinity matrix: {aff_rows} x {aff_cols}")
                    else:
                        print("[guide] Warning: could not extract hidden states")
                except Exception as e:
                    print(f"[guide] Warning: affinity computation failed: {e}")
            else:
                print("[guide] Warning: could not patch guide model for hidden states")
        else:
            print(f"[guide] Guide model not found: {guide_path} -- skipping")

    # ── Build canonical RoutingTrace and save ──
    trace = capture.build_trace(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        tokenizer=tokenizer,
        model_id=meta["model_id"],
        model_type=meta["model_type"],
        num_layers=meta["num_layers"],
        num_experts=meta["num_experts"],
        top_k=meta["top_k"],
        backend="mlx",
    )

    trace.guide_affinity = guide_affinity

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    trace.validate()
    trace.save(str(log_path / "routing.json"))
    print(f"[save] Routing trace: {log_path / 'routing.json'}")

    # ── Quick summary ──
    expert_load = trace.expert_load()
    if expert_load:
        print(f"\n[routing] Expert load distribution (top 10):")
        for expert_id, count in sorted(expert_load.items(), key=lambda x: -x[1])[:10]:
            print(f"  expert {expert_id:3d}: {count:5d} tokens")

    # ── Cleanup: restore original MoE call ──
    if _MOE_ORIG_CALL is not None:
        moe_cls = None
        for layer in _get_layers(model):
            moe_block = getattr(layer, "mlp", None)
            if moe_block is not None and hasattr(moe_block, "switch_mlp"):
                moe_cls = type(moe_block)
                break
        if moe_cls is not None:
            moe_cls.__call__ = _MOE_ORIG_CALL

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
