#!/usr/bin/env python3
"""
run_vllm.py — MoE research CLI using vLLM.

Delegates prefill + decode to vLLM's engine (no manual generation loop) and
captures per-token, per-layer expert routing into the canonical RoutingTrace,
plus router steering (force / bias / ablate) for intervention studies.

Usage:
    python run_vllm.py run \\
        --model Qwen/Qwen3.6-35B-A3B \\
        --prompt "Explain MoE routing." \\
        --max-tokens 64

    # With the dense guide-model affinity prior (same as moe_run.py):
    python run_vllm.py run --model Qwen/Qwen3.6-35B-A3B \\
        --guide-model Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled \\
        --max-tokens 64

    python run_vllm.py intervene --force-expert 0 5 --max-tokens 32

    python run_vllm.py ablate --ablate-expert 3 12 --max-tokens 32

Analyze / compare reuse the HF CLI (same trace schema):
    python run_research.py analyze logs/routing_vllm.json

NOTE: vLLM is CUDA-only upstream. On Apple Silicon the vllm-metal plugin
(https://github.com/vllm-project/vllm-metal) runs MLX-format models through
the Metal GPU backend; routing capture then patches the MLX MoE blocks
(``install_vllm_metal_hooks``). Hooks require the in-process engine core
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``, set automatically). The host IP is
pinned to loopback (``VLLM_HOST_IP=127.0.0.1`` + ``MASTER_ADDR``) — the
macOS hostname can resolve to a non-loopback LAN IP and crash PyTorch's
TCPStore.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable so ``from src.data...`` resolves regardless of
# the CWD used to invoke this script.
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MoE research toolkit (vLLM)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run vLLM MoE inference with routing capture")
    _add_common_args(p_run)
    p_run.add_argument("--output", default=None,
                       help="Trace output path (default: logs/routing_vllm_<model-slug>.json "
                            "— stamped per model so different models never overwrite each other)")

    # ── intervene ──────────────────────────────────────────────────
    p_int = sub.add_parser("intervene", help="vLLM inference with routing intervention")
    _add_common_args(p_int)
    p_int.add_argument("--output", default="logs/intervened_vllm_routing.json")
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
    p_abl = sub.add_parser("ablate", help="vLLM inference with expert ablation")
    _add_common_args(p_abl)
    p_abl.add_argument("--output", default="logs/ablated_vllm_routing.json")
    p_abl.add_argument("--ablate-expert", nargs=2, action="append", default=[],
                       metavar=("LAYER", "EXPERT_ID"),
                       help="Zero out expert output (repeatable)")

    return parser


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--prompt",
                   default="Explain why Mixture of Experts models need routing, in one paragraph.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-chat", action="store_true", help="Disable chat template")
    p.add_argument("--system-prompt", default="You are a helpful assistant.")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--enforce-eager", action="store_true",
                   help="Disable CUDA-graph capture (required for routing hooks to fire)")
    p.add_argument("--guide-model", type=str, default=None,
                   help="HF-format dense generalised model for affinity graph prior "
                        "(captures last-layer hidden states, same as moe_run.py)")


# ═══════════════════════════════════════════════════════════════════
# Shared: vLLM generation
# ═══════════════════════════════════════════════════════════════════

def _get_eos_ids(llm) -> list[int] | None:
    hf_cfg = llm.llm_engine.vllm_config.model_config.hf_config
    eos = getattr(hf_cfg, "eos_token_id", None)
    if eos is None:
        return None
    if isinstance(eos, (list, tuple)):
        ids = [int(e) for e in eos if e is not None]
        return ids or None
    return [int(eos)]


def _build_sampling_params(args, llm):
    from vllm import SamplingParams

    kwargs = dict(temperature=args.temp, max_tokens=args.max_tokens, seed=args.seed)
    eos = _get_eos_ids(llm)
    if eos:
        kwargs["stop_token_ids"] = eos
    return SamplingParams(**kwargs)


def _run_vllm_generation(args: argparse.Namespace, steering) -> int:
    """Run one live vLLM inference with routing capture and save the trace.

    Delegates to ``src.data.live_capture.capture_live`` (realtime capture,
    no pre-recorded traces). The output path defaults to a model-stamped
    file so captures from different models never overwrite each other.
    """
    from src.data.live_capture import capture_live

    try:
        trace = capture_live(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temp=args.temp,
            seed=args.seed,
            system_prompt=args.system_prompt,
            no_chat=args.no_chat,
            enforce_eager=args.enforce_eager,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            trust_remote_code=args.trust_remote_code,
            steering=steering,
        )
    except Exception as e:
        print(f"[error] live capture failed: {e}")
        return 1

    if steering is not None and steering.active_layers:
        print(f"[steering] Active layers: {steering.active_layers}")

    # Model-stamped default output: logs/routing_vllm_<model-slug>.json
    if args.output:
        out_path = Path(args.output)
    else:
        slug = Path(args.model).name.replace("/", "_")
        out_path = Path("logs") / f"routing_vllm_{slug}.json"

    trace.guide_affinity = _load_guide_affinity(args, None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace.save(str(out_path))
    print(f"[save] Routing trace → {out_path}")
    print(f"[stats] {trace.total_routing_events()} routing events, "
          f"{len(trace.routes)} token positions, "
          f"{trace.meta.num_moe_layers} MoE layers")

    load = trace.expert_load()
    if load:
        print(f"\n[load] Expert load distribution (top 10):")
        for eid, count in sorted(load.items(), key=lambda x: -x[1])[:10]:
            print(f"  expert {eid:3d}: {count:5d} tokens")

    return 0


# ═══════════════════════════════════════════════════════════════════
# Guide model — dense generalised model for affinity graph prior
# ═══════════════════════════════════════════════════════════════════

_GUIDE_HIDDEN = None


def _patch_guide_for_hidden_states(model) -> bool:
    """Hook the guide model so its next forward captures last-layer hidden states."""
    import torch

    global _GUIDE_HIDDEN
    _GUIDE_HIDDEN = None

    # Prefer capturing the input to lm_head (final hidden states before projection).
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:

        def _capture(module, inp, out):
            global _GUIDE_HIDDEN
            _GUIDE_HIDDEN = inp[0].detach().to(torch.float32)

        lm_head.register_forward_hook(_capture)
        return True

    # Fallback: hook the last decoder layer's output norm.
    from src.data.model_utils import _find_decoder_layers

    try:
        layers = _find_decoder_layers(model)
    except ValueError:
        return False
    if not layers:
        return False
    last = layers[-1]
    for attr in ("norm", "final_layer_norm", "ln_f", "output_norm"):
        norm = getattr(last, attr, None) or getattr(model, attr, None)
        if norm is not None:

            def _capture_norm(module, inp, out):
                global _GUIDE_HIDDEN
                _GUIDE_HIDDEN = out.detach().to(torch.float32)

            norm.register_forward_hook(_capture_norm)
            return True
    return False


def _compute_guide_affinity() -> list | None:
    """Compute token cosine-similarity matrix from captured guide hidden states."""
    import torch

    global _GUIDE_HIDDEN
    h = _GUIDE_HIDDEN
    if h is None:
        return None
    if h.ndim == 3:
        h = h[0]
    norms = torch.linalg.norm(h, dim=-1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    h_norm = h / norms
    return (h_norm @ h_norm.T).tolist()


def _load_guide_affinity(args: argparse.Namespace, messages) -> list | None:
    """Load the dense guide model and return its affinity matrix (or None)."""
    if not args.guide_model:
        return None
    guide_path = Path(args.guide_model)
    if not guide_path.exists():
        print(f"[guide] Guide model not found: {guide_path} -- skipping")
        return None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"[guide] Loading generalised model: {guide_path.name} ({device})")

        guide_model = AutoModelForCausalLM.from_pretrained(
            str(guide_path), torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device).eval()
        guide_tokenizer = AutoTokenizer.from_pretrained(str(guide_path))

        if not _patch_guide_for_hidden_states(guide_model):
            print("[guide] Warning: could not patch guide model for hidden states")
            return None

        if messages is not None and guide_tokenizer.chat_template is not None:
            guide_prompt = guide_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            guide_ids = guide_tokenizer.encode(guide_prompt)
        else:
            guide_ids = guide_tokenizer.encode(args.prompt)

        with torch.no_grad():
            _ = guide_model(torch.as_tensor([guide_ids], device=device))

        affinity = _compute_guide_affinity()
        if affinity is None:
            print("[guide] Warning: could not extract hidden states")
            return None
        print(f"[guide] Affinity matrix: {len(affinity)} x {len(affinity[0])}")
        return affinity
    except Exception as e:
        print(f"[guide] Warning: guide affinity computation failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# Subcommands
# ═══════════════════════════════════════════════════════════════════

def _cmd_run(args: argparse.Namespace) -> int:
    return _run_vllm_generation(args, steering=None)


def _cmd_intervene(args: argparse.Namespace) -> int:
    from src.data.vllm_capture import VllmSteering

    steering = VllmSteering()

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

    return _run_vllm_generation(args, steering=steering)


def _cmd_ablate(args: argparse.Namespace) -> int:
    from src.data.vllm_capture import VllmSteering

    steering = VllmSteering()

    for layer_str, expert_str in args.ablate_expert:
        layer, expert = int(layer_str), int(expert_str)
        steering.ablate_expert(layer=layer, expert_id=expert)
        print(f"[config] ablate-expert: layer={layer}, expert={expert}")

    if not steering.active_layers:
        print("[error] No ablations configured. Use --ablate-expert LAYER EXPERT_ID")
        return 1

    return _run_vllm_generation(args, steering=steering)


_COMMANDS = {
    "run": _cmd_run,
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
