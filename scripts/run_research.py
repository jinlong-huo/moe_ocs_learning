#!/usr/bin/env python3
"""
run_research.py — MoE research CLI using HuggingFace Transformers.

Usage:
    python run_research.py run \\
        --model Qwen/Qwen3.6-35B-A3B \\
        --prompt "Explain MoE routing." \\
        --max-tokens 64 --device mps

    python run_research.py analyze logs/routing.json

    python run_research.py compare trace_a.json trace_b.json

    python run_research.py intervene --force-expert 12 5 ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MoE research toolkit (HF Transformers)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run MoE inference with routing capture")
    p_run.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p_run.add_argument("--prompt", default="Explain why Mixture of Experts models need routing, in one paragraph.")
    p_run.add_argument("--max-tokens", type=int, default=256)
    p_run.add_argument("--temp", type=float, default=0.6)
    p_run.add_argument("--device", default="auto")
    p_run.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    p_run.add_argument("--output", default="logs/routing.json")
    p_run.add_argument("--no-chat", action="store_true")
    p_run.add_argument("--system-prompt", default="You are a helpful assistant.")

    # ── analyze ────────────────────────────────────────────────────
    p_ana = sub.add_parser("analyze", help="Analyze a saved routing trace")
    p_ana.add_argument("trace", help="Path to routing.json")
    p_ana.add_argument("--metric", default="all",
                       choices=["load", "entropy", "specialization", "coactivation", "all"])

    # ── compare ────────────────────────────────────────────────────
    p_cmp = sub.add_parser("compare", help="Compare two routing traces")
    p_cmp.add_argument("trace_a")
    p_cmp.add_argument("trace_b")

    # ── intervene ──────────────────────────────────────────────────
    p_int = sub.add_parser("intervene", help="Run inference with routing intervention")
    p_int.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p_int.add_argument("--prompt", default="Explain MoE routing.")
    p_int.add_argument("--max-tokens", type=int, default=128)
    p_int.add_argument("--temp", type=float, default=0.6)
    p_int.add_argument("--device", default="auto")
    p_int.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    p_int.add_argument("--output", default="logs/intervened_routing.json")
    p_int.add_argument("--force-expert", nargs=2, action="append", default=[],
                       metavar=("LAYER", "EXPERT_ID"),
                       help="Force expert selection (repeatable)")
    p_int.add_argument("--force-expert-exclusive", nargs=2, action="append", default=[],
                       metavar=("LAYER", "EXPERT_ID"),
                       help="Force ONLY this expert, suppress all others (repeatable)")
    p_int.add_argument("--bias-expert", nargs=3, action="append", default=[],
                       metavar=("LAYER", "EXPERT_ID", "BIAS"),
                       help="Add bias to router logits (repeatable)")

    # ── ablate ─────────────────────────────────────────────────────
    p_abl = sub.add_parser("ablate", help="Run inference with expert ablation")
    p_abl.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p_abl.add_argument("--prompt", default="Explain MoE routing.")
    p_abl.add_argument("--max-tokens", type=int, default=128)
    p_abl.add_argument("--temp", type=float, default=0.6)
    p_abl.add_argument("--device", default="auto")
    p_abl.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    p_abl.add_argument("--output", default="logs/ablated_routing.json")
    p_abl.add_argument("--ablate-expert", nargs=2, action="append", default=[],
                       metavar=("LAYER", "EXPERT_ID"),
                       help="Zero out expert output (repeatable)")

    return parser


# ═══════════════════════════════════════════════════════════════════
# Subcommand: run
# ═══════════════════════════════════════════════════════════════════

def _cmd_run(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.data.model_utils import (
        apply_mps_patches, enable_router_logits, ModelLayout,
    )
    from src.data.routing_capture import RoutingCapture

    # ── MPS workarounds ──────────────────────────────────────────
    patched = apply_mps_patches()
    if patched:
        print("[mps] Applied MPS compatibility patches")

    # ── Determine device / dtype ──────────────────────────────────
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device_map = "mps"
        elif torch.cuda.is_available():
            device_map = "cuda"
        else:
            device_map = "cpu"
    else:
        device_map = args.device

    dtype_map = {"float16": torch.float16, "float32": torch.float32,
                 "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    # ── Load model & tokenizer ────────────────────────────────────
    model_id = args.model
    print(f"[load] Model: {model_id}")
    print(f"[load] Device: {device_map}, dtype: {args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    # ── Enable router logits & detect layout ──────────────────────
    if not enable_router_logits(model):
        print("[warn] Could not enable output_router_logits — routing capture may be empty")

    layout = ModelLayout.from_model(model, model.config)
    print(f"[layout] model_type={layout.model_type}, "
          f"layers={layout.num_layers}, MoE layers={layout.num_moe_layers}, "
          f"experts={layout.num_experts}, top_k={layout.top_k}")

    # ── Tokenize ──────────────────────────────────────────────────
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
        prompt_ids = tokenizer.encode(prompt_text)
        print("[input] Chat template applied")
    else:
        prompt_ids = tokenizer.encode(args.prompt)

    print(f"[input] Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"[input] Prompt tokens: {len(prompt_ids)}")

    # ── Set up capture & generation ───────────────────────────────
    capture = RoutingCapture(layout, model_id=model_id, backend="hf")
    device = next(model.parameters()).device
    eos_ids = tokenizer.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    eos_ids = [e for e in eos_ids if e is not None]

    input_ids = torch.tensor([prompt_ids], device=device)
    past_key_values = None
    generated_ids: list[int] = []

    # ── Prefill ───────────────────────────────────────────────────
    capture.begin_prefill(len(prompt_ids), prompt_ids)
    with torch.no_grad():
        outputs = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            output_router_logits=True,
        )
    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values
    if outputs.router_logits is not None:
        capture.consume(outputs.router_logits, input_ids)

    # ── Decode loop ───────────────────────────────────────────────
    for step in range(args.max_tokens):
        logits_scaled = logits / args.temp
        probs = torch.softmax(logits_scaled, dim=-1)
        next_token_id = int(torch.multinomial(probs, num_samples=1).item())
        generated_ids.append(next_token_id)

        # print token
        decoded = tokenizer.decode([next_token_id])
        print(decoded, end="", flush=True)

        if next_token_id in eos_ids:
            print()
            break

        capture.begin_decode_step(next_token_id)
        next_input = torch.tensor([[next_token_id]], device=device)
        with torch.no_grad():
            outputs = model(
                next_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_router_logits=True,
            )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        if outputs.router_logits is not None:
            capture.consume(outputs.router_logits, next_input)
    else:
        print()

    # ── Finalize & save ───────────────────────────────────────────
    trace = capture.finalize(tokenizer)
    trace.validate()
    out_path = trace.save(args.output)
    print(f"\n[save] Routing trace → {out_path}")
    print(f"[stats] Generated {len(generated_ids)} tokens, "
          f"{trace.total_routing_events()} routing events, "
          f"{len(trace.routes)} token positions")

    # ── Quick summary ─────────────────────────────────────────────
    load = trace.expert_load()
    if load:
        print(f"\n[load] Expert load distribution (top 10):")
        for eid, count in sorted(load.items(), key=lambda x: -x[1])[:10]:
            bar = "█" * min(count // 10, 50)
            print(f"  expert {eid:3d}: {count:5d} {bar}")

    return 0


# ═══════════════════════════════════════════════════════════════════
# Subcommand: analyze
# ═══════════════════════════════════════════════════════════════════

def _cmd_analyze(args: argparse.Namespace) -> int:
    from src.data.routing_schema import RoutingTrace

    trace = RoutingTrace.load(args.trace)
    print(f"[trace] {trace.meta.model_id}  "
          f"prompt={trace.meta.prompt_len} gen={trace.meta.generated_len} "
          f"layers={trace.meta.num_moe_layers} experts={trace.meta.num_experts} "
          f"top_k={trace.meta.top_k}")

    metrics = ["load", "entropy", "specialization", "coactivation"] if args.metric == "all" else [args.metric]

    for metric in metrics:
        if metric == "load":
            _show_load(trace)
        elif metric == "entropy":
            _show_entropy(trace)
        elif metric == "specialization":
            _show_specialization(trace)
        elif metric == "coactivation":
            _show_coactivation(trace)

    return 0


def _show_load(trace) -> None:
    import math
    from collections import Counter

    load = trace.expert_load()
    total = sum(load.values())
    n_experts = trace.meta.num_experts
    counts = [load.get(i, 0) for i in range(n_experts)]
    mean = total / n_experts if n_experts else 0

    # Gini coefficient
    sorted_c = sorted(counts)
    cumsum = 0.0
    gini_sum = 0.0
    for i, c in enumerate(sorted_c):
        cumsum += c
        gini_sum += (i + 1) * c
    gini = (2 * gini_sum / (n_experts * total) - (n_experts + 1) / n_experts) if total > 0 else 0.0

    # Entropy
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log(p)
    max_entropy = math.log(n_experts)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    print(f"\n── Load Balance ──")
    print(f"  Total routing events: {total}")
    print(f"  Mean per expert:      {mean:.1f}")
    print(f"  Max per expert:       {max(counts)}")
    print(f"  Min per expert:       {min(c for c in counts if c > 0)}")
    print(f"  Dead experts (0 load): {sum(1 for c in counts if c == 0)} / {n_experts}")
    print(f"  Gini coefficient:     {gini:.4f}  (0=perfectly equal, 1=one expert)")
    print(f"  Normalized entropy:   {normalized_entropy:.4f}  (1=uniform)")


def _show_entropy(trace) -> None:
    import math

    per_layer = trace.per_layer_expert_load()
    print(f"\n── Per-Layer Entropy ──")
    print(f"  {'Layer':>6}  {'Entropy':>8}  {'Normalized':>10}  {'Active Experts':>15}")
    n_experts = trace.meta.num_experts
    max_ent = math.log(n_experts)
    for lid in sorted(per_layer.keys(), key=int):
        load = per_layer[lid]
        total = sum(load.values())
        entropy = 0.0
        for c in load.values():
            if c > 0:
                p = c / total
                entropy -= p * math.log(p)
        norm = entropy / max_ent if max_ent > 0 else 0.0
        active = sum(1 for c in load.values() if c > 0)
        print(f"  {lid:>6}  {entropy:8.4f}  {norm:10.4f}  {active:>4} / {n_experts}")


def _show_specialization(trace) -> None:
    # Per expert: which tokens does it see most?
    expert_tokens: dict[int, dict[str, int]] = {}
    for route in trace.routes:
        for lr in route.layers.values():
            for e in lr.experts:
                if e not in expert_tokens:
                    expert_tokens[e] = {}
                tok = route.token_str
                expert_tokens[e][tok] = expert_tokens[e].get(tok, 0) + 1

    load = trace.expert_load()
    top_experts = sorted(load.items(), key=lambda x: -x[1])[:10]

    print(f"\n── Expert Specialization (top 10 experts by load) ──")
    for eid, count in top_experts:
        tokens = expert_tokens.get(eid, {})
        top_tokens = sorted(tokens.items(), key=lambda x: -x[1])[:5]
        token_strs = ", ".join(f"{t!r}({c})" for t, c in top_tokens)
        print(f"  expert {eid:3d} ({count:5d}): {token_strs}")


def _show_coactivation(trace) -> None:
    n_experts = trace.meta.num_experts
    coact = [[0] * n_experts for _ in range(n_experts)]
    for route in trace.routes:
        for lr in route.layers.values():
            for i, e1 in enumerate(lr.experts):
                for e2 in lr.experts[i + 1:]:
                    coact[e1][e2] += 1
                    coact[e2][e1] += 1

    # Find top coactivation pairs
    pairs = []
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            if coact[i][j] > 0:
                pairs.append((i, j, coact[i][j]))
    pairs.sort(key=lambda x: -x[2])

    print(f"\n── Expert Coactivation (top 15 pairs) ──")
    for e1, e2, count in pairs[:15]:
        print(f"  expert {e1:3d} + expert {e2:3d}: {count:5d}")


# ═══════════════════════════════════════════════════════════════════
# Subcommand: compare
# ═══════════════════════════════════════════════════════════════════

def _cmd_compare(args: argparse.Namespace) -> int:
    from src.data.routing_schema import RoutingTrace

    trace_a = RoutingTrace.load(args.trace_a)
    trace_b = RoutingTrace.load(args.trace_b)

    print(f"[trace A] {trace_a.meta.model_id}  run_id={trace_a.meta.run_id}")
    print(f"[trace B] {trace_b.meta.model_id}  run_id={trace_b.meta.run_id}")
    print()

    load_a = trace_a.expert_load()
    load_b = trace_b.expert_load()
    all_experts = sorted(set(load_a) | set(load_b))

    print(f"  {'Expert':>6}  {'Trace A':>8}  {'Trace B':>8}  {'Delta':>8}")
    for e in all_experts[:20]:
        ca = load_a.get(e, 0)
        cb = load_b.get(e, 0)
        delta = cb - ca
        print(f"  {e:>6}  {ca:>8}  {cb:>8}  {delta:>+8}")

    return 0


# ═══════════════════════════════════════════════════════════════════
# Shared: intervened generation
# ═══════════════════════════════════════════════════════════════════

def _run_intervened_generation(
    args: argparse.Namespace,
    steering: "RouterSteering",
    label: str,
) -> int:
    """Shared generation loop used by ``intervene`` and ``ablate``.

    Loads the model, enables router-logits, wraps generation inside the
    provided ``RouterSteering`` context, and saves the trace.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.data.model_utils import (
        apply_mps_patches, enable_router_logits, ModelLayout,
    )
    from src.data.routing_capture import RoutingCapture

    # ── MPS workarounds ──────────────────────────────────────────
    patched = apply_mps_patches()
    if patched:
        print("[mps] Applied MPS compatibility patches")

    # ── Device / dtype ───────────────────────────────────────────
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device_map = "mps"
        elif torch.cuda.is_available():
            device_map = "cuda"
        else:
            device_map = "cpu"
    else:
        device_map = args.device

    dtype_map = {"float16": torch.float16, "float32": torch.float32,
                 "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    # ── Load model & tokenizer ───────────────────────────────────
    model_id = args.model
    print(f"[load] Model: {model_id}")
    print(f"[load] Device: {device_map}, dtype: {args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    # ── Enable router logits & detect layout ─────────────────────
    if not enable_router_logits(model):
        print("[warn] Could not enable output_router_logits")

    layout = ModelLayout.from_model(model, model.config)
    print(f"[layout] model_type={layout.model_type}, "
          f"layers={layout.num_layers}, MoE layers={layout.num_moe_layers}, "
          f"experts={layout.num_experts}, top_k={layout.top_k}")

    # Re-attach steering to the *loaded* model instance
    steering.model = model
    steering.layout = layout

    # ── Tokenize ─────────────────────────────────────────────────
    use_chat = (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    )
    if use_chat:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt_text)
    else:
        prompt_ids = tokenizer.encode(args.prompt)

    print(f"[input] Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print(f"[input] Prompt tokens: {len(prompt_ids)}")

    # ── Set up capture ───────────────────────────────────────────
    capture = RoutingCapture(layout, model_id=model_id, backend="hf")
    device = next(model.parameters()).device
    eos_ids = tokenizer.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    eos_ids = [e for e in eos_ids if e is not None]

    print(f"[steering] Active layers: {steering.active_layers}")
    print(f"[steering] Entering {label} context...")

    input_ids = torch.tensor([prompt_ids], device=device)
    past_key_values = None
    generated_ids: list[int] = []

    with steering:
        # ── Prefill ───────────────────────────────────────────────
        capture.begin_prefill(len(prompt_ids), prompt_ids)
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_router_logits=True,
            )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        if outputs.router_logits is not None:
            capture.consume(outputs.router_logits, input_ids)

        # ── Decode loop ───────────────────────────────────────────
        for step in range(args.max_tokens):
            logits_scaled = logits / args.temp
            probs = torch.softmax(logits_scaled, dim=-1)
            next_token_id = int(torch.multinomial(probs, num_samples=1).item())
            generated_ids.append(next_token_id)

            decoded = tokenizer.decode([next_token_id])
            print(decoded, end="", flush=True)

            if next_token_id in eos_ids:
                print()
                break

            capture.begin_decode_step(next_token_id)
            next_input = torch.tensor([[next_token_id]], device=device)
            with torch.no_grad():
                outputs = model(
                    next_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_router_logits=True,
                )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            if outputs.router_logits is not None:
                capture.consume(outputs.router_logits, next_input)
        else:
            print()

    print(f"[steering] Exited {label} context — gates restored")

    # ── Finalize & save ───────────────────────────────────────────
    trace = capture.finalize(tokenizer)
    trace.validate()
    out_path = trace.save(args.output)
    print(f"\n[save] Intervened trace → {out_path}")
    print(f"[stats] Generated {len(generated_ids)} tokens, "
          f"{trace.total_routing_events()} routing events, "
          f"{len(trace.routes)} token positions")

    # ── Quick summary ─────────────────────────────────────────────
    load = trace.expert_load()
    if load:
        print(f"\n[load] Expert load distribution (top 10):")
        for eid, count in sorted(load.items(), key=lambda x: -x[1])[:10]:
            bar = "█" * min(count // 10, 50)
            print(f"  expert {eid:3d}: {count:5d} {bar}")

    return 0


# ═══════════════════════════════════════════════════════════════════
# Subcommand: intervene
# ═══════════════════════════════════════════════════════════════════

def _cmd_intervene(args: argparse.Namespace) -> int:
    from src.data.routing_interventions import RouterSteering

    # Build steering config from CLI args
    steering = RouterSteering(None, None)  # model/layout attached later

    for layer_str, expert_str in args.force_expert:
        layer, expert = int(layer_str), int(expert_str)
        steering.force_expert(layer=layer, expert_id=expert, exclusive=False)
        print(f"[config] force-expert: layer={layer}, expert={expert}")

    for layer_str, expert_str in args.force_expert_exclusive:
        layer, expert = int(layer_str), int(expert_str)
        steering.force_expert(layer=layer, expert_id=expert, exclusive=True)
        print(f"[config] force-expert (exclusive): layer={layer}, expert={expert}")

    for layer_str, expert_str, bias_str in args.bias_expert:
        layer, expert, bias = int(layer_str), int(expert_str), float(bias_str)
        steering.bias_expert(layer=layer, expert_id=expert, bias=bias)
        print(f"[config] bias-expert: layer={layer}, expert={expert}, bias={bias}")

    if not steering.active_layers:
        print("[error] No interventions configured.")
        print("  Use --force-expert, --force-expert-exclusive, or --bias-expert")
        return 1

    return _run_intervened_generation(args, steering, label="intervene")


# ═══════════════════════════════════════════════════════════════════
# Subcommand: ablate
# ═══════════════════════════════════════════════════════════════════

def _cmd_ablate(args: argparse.Namespace) -> int:
    from src.data.routing_interventions import RouterSteering

    steering = RouterSteering(None, None)

    for layer_str, expert_str in args.ablate_expert:
        layer, expert = int(layer_str), int(expert_str)
        steering.ablate_expert(layer=layer, expert_id=expert)
        print(f"[config] ablate-expert: layer={layer}, expert={expert}")

    if not steering.active_layers:
        print("[error] No ablations configured. Use --ablate-expert LAYER EXPERT_ID")
        return 1

    return _run_intervened_generation(args, steering, label="ablate")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

_COMMANDS = {
    "run": _cmd_run,
    "analyze": _cmd_analyze,
    "compare": _cmd_compare,
    "intervene": _cmd_intervene,
    "ablate": _cmd_ablate,
}


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
