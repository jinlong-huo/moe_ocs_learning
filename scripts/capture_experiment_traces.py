#!/usr/bin/env python3
"""Batch capture: run a prompt taxonomy through a real MoE model and save traces.

Loads the model ONCE, then runs each prompt in the taxonomy sequentially with
temperature=0 (greedy decoding) for deterministic, reproducible routing traces.

Usage:
    python scripts/capture_experiment_traces.py
    python scripts/capture_experiment_traces.py --model ./models/Qwen3.6-35B-A3B-4bit
    python scripts/capture_experiment_traces.py --groups ml_moe_1,ml_moe_2,history_1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

# ── Import capture machinery from moe_run ──────────────────────────
# Add project root to path so we can import from both src/ and moe_run
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))


def _load_prompt_taxonomy(config_path: str) -> dict:
    """Load prompt taxonomy from YAML config."""
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# Model helpers (adapted from moe_run.py)
# ═══════════════════════════════════════════════════════════════════════


def _get_layers(model):
    """Get decoder layers from model, handling nested structures."""
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    raise AttributeError("Cannot find decoder layers on model")


def _extract_model_meta(model, model_path: str) -> dict:
    """Extract model metadata for the routing trace."""
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

    config = getattr(model, "config", None)
    if config is not None:
        if num_experts == 0:
            num_experts = getattr(config, "num_experts", 0) or 0
        if top_k == 0:
            top_k = (
                getattr(config, "num_experts_per_tok", 0)
                or getattr(config, "top_k", 0)
                or 0
            )
        model_type = getattr(config, "model_type", model_type)

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


# ═══════════════════════════════════════════════════════════════════════
# MoE hooking
# ═══════════════════════════════════════════════════════════════════════

_MOE_ORIG_CALL = None


def _make_patched_call(capture):
    """Return a replacement ``__call__`` that logs routing then delegates."""

    def patched_call(self, x):
        layer_idx = self._layer_idx

        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if getattr(self, "norm_topk_prob", False):
            scores = scores / scores.sum(axis=-1, keepdims=True)

        capture.log(
            layer_id=layer_idx,
            batch_token_experts=inds.tolist()[0],
            batch_token_weights=scores.tolist()[0],
        )

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

    return patched


def restore_moe_call(model):
    """Restore original MoE forward pass."""
    global _MOE_ORIG_CALL
    if _MOE_ORIG_CALL is None:
        return
    for layer in _get_layers(model):
        moe_block = getattr(layer, "mlp", None)
        if moe_block is not None and hasattr(moe_block, "switch_mlp"):
            type(moe_block).__call__ = _MOE_ORIG_CALL
            break


# ═══════════════════════════════════════════════════════════════════════
# Single-prompt capture
# ═══════════════════════════════════════════════════════════════════════


def run_one_prompt(
    model,
    tokenizer,
    meta: dict,
    prompt_text: str,
    temp: float,
    max_tokens: int,
    system_prompt: str = "You are a helpful assistant.",
) -> dict:
    """Run a single prompt through the model and return the routing trace.

    Returns:
        dict with keys: trace (RoutingTrace), prompt_tokens, generated_tokens,
        num_prompt_tokens, num_generated_tokens, routing_events.
    """
    from src.data.mlx_capture import RoutingCapture

    state = {"seq_pos": 0, "phase": "prefill"}
    capture = RoutingCapture(state)

    # Install hooks
    n_patched = install_routing_hooks(model, capture)

    # Tokenize with chat template
    use_chat = (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    )
    if use_chat:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = tokenizer.encode(full_prompt)
    else:
        prompt_tokens = tokenizer.encode(prompt_text)

    # Prefill
    prompt_cache = make_prompt_cache(model)
    prompt_array = mx.array([prompt_tokens])
    generated_tokens: list[int] = []

    state["phase"] = "prefill"
    logits = model(prompt_array, cache=prompt_cache)
    logits = logits[:, -1, :]
    state["seq_pos"] += len(prompt_tokens)

    # Generate
    token = prompt_array
    for _step in range(max_tokens):
        state["phase"] = "decode"

        if temp <= 0.0:
            # Greedy: always pick argmax
            next_token_id = int(logits.argmax(axis=-1).item())
        else:
            next_token = mx.random.categorical(logits / temp)
            next_token_id = int(next_token.item())

        generated_tokens.append(next_token_id)

        if next_token_id == tokenizer.eos_token_id:
            break

        token = mx.array([next_token_id])
        logits = model(token[None], cache=prompt_cache)
        logits = logits[:, -1, :]
        state["seq_pos"] += 1

    # Build trace
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

    # Restore original forward
    restore_moe_call(model)

    return {
        "trace": trace,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "num_prompt_tokens": len(prompt_tokens),
        "num_generated_tokens": len(generated_tokens),
        "routing_events": capture.route_count,
        "num_moe_layers_patched": n_patched,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Batch capture routing traces from a prompt taxonomy"
    )
    parser.add_argument(
        "--config",
        default="configs/experiment_prompts.yaml",
        help="Path to prompt taxonomy YAML",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path (overrides config capture.model)",
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="Comma-separated group IDs to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (overrides config capture.output_dir)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=None,
        help="Temperature (overrides config; 0.0 = greedy)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens per prompt (overrides config)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip prompts whose trace file already exists",
    )
    args = parser.parse_args()

    # ── Load config ──
    cfg = _load_prompt_taxonomy(args.config)
    groups = cfg["groups"]
    experiments = cfg.get("experiments", [])
    cap_cfg = cfg.get("capture", {})

    model_path = args.model or cap_cfg.get("model")
    temp = args.temp if args.temp is not None else cap_cfg.get("temp", 0.0)
    max_tokens = args.max_tokens or cap_cfg.get("max_tokens", 256)
    output_dir = Path(args.output_dir or cap_cfg.get("output_dir", "outputs/experiment_traces"))

    # Filter groups if requested
    if args.groups:
        requested = set(args.groups.split(","))
        groups = [g for g in groups if g["id"] in requested]
        if not groups:
            print(f"[error] No groups match: {args.groups}")
            sys.exit(1)

    print("=" * 65)
    print("BATCH CAPTURE: Prompt Taxonomy → Routing Traces")
    print("=" * 65)
    print(f"Model:      {model_path}")
    print(f"Temp:       {temp} {'(greedy)' if temp <= 0 else ''}")
    print(f"Max tokens: {max_tokens}")
    print(f"Groups:     {len(groups)}")
    print(f"Output:     {output_dir}")
    print(f"Experiments:{len(experiments)} defined")
    print()

    # ── Validate model path ──
    if not Path(model_path).exists():
        print(f"[error] Model not found: {model_path}")
        sys.exit(1)

    # ── Load model once ──
    print(f"[load] Loading model from {model_path} ...")
    t0 = time.time()
    model, tokenizer = load(str(model_path))
    print(f"[load] Loaded in {time.time() - t0:.1f}s")

    meta = _extract_model_meta(model, model_path)
    print(f"[model] {meta['num_layers']} layers, {meta['moe_count']} MoE layers, "
          f"{meta['num_experts']} experts, top_k={meta['top_k']}")
    print()

    # ── Run each prompt ──
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, group in enumerate(groups):
        group_id = group["id"]
        domain = group["domain"]
        role = group.get("role", "test")
        prompt = group["prompt"]

        out_path = output_dir / group_id / "routing.json"
        if args.skip_existing and out_path.exists():
            print(f"[{i+1:2d}/{len(groups)}] SKIP {group_id} ({domain}/{role}) — trace exists")
            # Load existing trace for manifest
            from src.data.routing_schema import RoutingTrace
            existing_trace = RoutingTrace.load(str(out_path))
            results.append({
                "group_id": group_id,
                "domain": domain,
                "role": role,
                "trace_path": str(out_path),
                "num_prompt_tokens": existing_trace.meta.prompt_len,
                "num_generated_tokens": existing_trace.meta.generated_len,
                "routing_events": existing_trace.total_routing_events(),
            })
            continue

        print(f"[{i+1:2d}/{len(groups)}] {group_id} ({domain}/{role})")
        print(f"       prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        t_start = time.time()
        result = run_one_prompt(
            model=model,
            tokenizer=tokenizer,
            meta=meta,
            prompt_text=prompt,
            temp=temp,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - t_start

        trace = result["trace"]
        trace_dir = output_dir / group_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace.validate()
        trace.save(str(trace_dir / "routing.json"))

        # Quick stats
        expert_load = trace.expert_load()
        top5 = sorted(expert_load.items(), key=lambda x: -x[1])[:5]
        top5_str = "  ".join(f"e{e:2d}:{c:4d}" for e, c in top5)

        print(f"       tokens:  {result['num_prompt_tokens']} prompt + "
              f"{result['num_generated_tokens']} generated = "
              f"{result['num_prompt_tokens'] + result['num_generated_tokens']} total")
        print(f"       events:  {result['routing_events']} routing events, "
              f"{result['num_moe_layers_patched']} MoE layers patched")
        print(f"       time:    {elapsed:.1f}s ({result['num_generated_tokens'] / max(elapsed, 0.01):.1f} tok/s)")
        print(f"       experts: {top5_str}")
        print()

        results.append({
            "group_id": group_id,
            "domain": domain,
            "role": role,
            "trace_path": str(trace_dir / "routing.json"),
            "num_prompt_tokens": result["num_prompt_tokens"],
            "num_generated_tokens": result["num_generated_tokens"],
            "routing_events": result["routing_events"],
            "capture_time_s": round(elapsed, 1),
        })

    # ── Export manifest ──
    manifest = {
        "model": model_path,
        "model_meta": meta,
        "temp": temp,
        "max_tokens": max_tokens,
        "num_groups": len(results),
        "experiments": experiments,
        "results": results,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Summary ──
    print("=" * 65)
    print("CAPTURE COMPLETE")
    print("=" * 65)
    domains = {}
    for r in results:
        d = r["domain"]
        domains.setdefault(d, []).append(r["group_id"])
    print(f"Groups captured: {len(results)}")
    for domain, ids in sorted(domains.items()):
        print(f"  {domain}: {', '.join(ids)}")
    print(f"\nManifest: {manifest_path}")
    print()
    print("Next steps:")
    print(f"  python scripts/analyze_cross_prompt_affinity.py --manifest {manifest_path}")
    print(f"  python scripts/run_affinity_loop.py --manifest {manifest_path}")


if __name__ == "__main__":
    main()
