#!/usr/bin/env python3
"""
vllm_serve.py — Multi-tenant MoE serving with routing capture (vLLM + vllm-metal).

Mimics a real serving deployment on Apple Silicon: the vLLM V1 engine
(vllm-metal backend) processes several tenants whose requests arrive on a
traffic schedule (Poisson / periodic / burst / uniform).  Every engine
step is recorded with its tenant composition, and each tenant gets a
canonical RoutingTrace — so expert-dispatch contention and per-tenant
delay (TTFT / ITL) are measurable, which a single-stream trace cannot
show.

Usage:
    # 6 tenants, Poisson arrivals at 1 req/s, concurrent serving
    python vllm_serve.py run \\
        --model ./models/Qwen3.6-35B-A3B-4bit \\
        --tenants 6 --schedule poisson --rate 1.0 --max-tokens 128
        
    # ~/.venv-vllm-metal/bin/python vllm_serve.py run --model ./models/Qwen3.6-35B-A3B-4bit --tenants 6 --schedule poisson --rate 1.0 --max-tokens 128
    
    # Zero-contention baseline for the same workload
    python vllm_serve.py run \\
        --model ./models/Qwen3.6-35B-A3B-4bit \\
        --tenants 6 --schedule poisson --rate 1.0 --mode sequential

    # Analyze a session (optionally compare against the baseline run)
    python vllm_serve.py analyze logs/multi_tenant/run_poisson_6t \\
        --baseline logs/multi_tenant/run_poisson_6t_sequential --plot

Requires the vllm-metal environment (e.g. ~/.venv-vllm-metal/bin/python).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable so ``src.serving`` resolves regardless
# of the CWD used to invoke this script.
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-tenant MoE serving with routing capture (vLLM)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run a multi-tenant serving session")
    p_run.add_argument("--model", type=str,
                       default="./models/Qwen3.6-35B-A3B-4bit",
                       help="Model path (HF or MLX-format dir)")
    p_run.add_argument("--tenants", type=int, default=4,
                       help="Number of concurrent tenant requests")
    p_run.add_argument("--schedule", default="poisson",
                       choices=["poisson", "periodic", "burst", "uniform"],
                       help="Arrival process")
    p_run.add_argument("--rate", type=float, default=1.0,
                       help="Mean arrival rate (req/s)")
    p_run.add_argument("--max-tokens", type=int, default=128)
    p_run.add_argument("--temp", type=float, default=0.6)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--mode", default="concurrent",
                       choices=["concurrent", "sequential"],
                       help="sequential = one tenant at a time (baseline)")
    p_run.add_argument("--prompts-file", type=str, default=None,
                       help="JSONL prompt pool (keys: prompt/text)")
    p_run.add_argument("--family", default="mixed",
                       choices=["identical", "similar", "mixed"],
                       help="Prompt family: identical = same prompt for all "
                            "tenants; similar = base prompt with a controlled "
                            "edit-distance gradient")
    p_run.add_argument("--base-prompt", type=str, default=None,
                       help="Base prompt for identical/similar families "
                            "(may contain {slot} placeholders)")
    p_run.add_argument("--slot-step", type=int, default=1,
                       help="For similar family: extra template slots changed "
                            "per tenant index step")
    p_run.add_argument("--greedy", action="store_true",
                       help="Greedy decoding (temperature=0) — deterministic "
                            "routing, isolates prompt-driven vs sampling drift")
    p_run.add_argument("--seed-mode", default="same",
                       choices=["same", "distinct"],
                       help="same = identical sampling seeds for all tenants; "
                            "distinct = per-tenant seeds (stochastic divergence)")
    p_run.add_argument("--prefix-cache", default="auto",
                       choices=["auto", "on", "off"],
                       help="Prefix caching: off = every tenant recomputes and "
                            "logs the full prefill routing; on = later tenants "
                            "reuse KV blocks (shared-KV behavior)")
    p_run.add_argument("--max-model-len", type=int, default=4096)
    p_run.add_argument("--no-chat", action="store_true",
                       help="Disable chat template")
    p_run.add_argument("--system-prompt", default="You are a helpful assistant.")
    p_run.add_argument("--output-dir", default="logs/multi_tenant")
    p_run.add_argument("--no-eager", action="store_true",
                       help="Disable enforce_eager (default: enabled)")

    # ── analyze ────────────────────────────────────────────────────
    p_ana = sub.add_parser("analyze", help="Analyze a saved serving session")
    p_ana.add_argument("session_dir", help="Session directory (contains session.json)")
    p_ana.add_argument("--baseline", default=None,
                       help="Sequential baseline session dir for slowdown comparison")
    p_ana.add_argument("--plot", action="store_true",
                       help="Save a timeline plot")

    # ── affinity ───────────────────────────────────────────────────
    p_aff = sub.add_parser(
        "affinity",
        help="Cross-tenant routing affinity (identical/similar prompt families)",
    )
    p_aff.add_argument("session_dir", help="Session directory (contains session.json)")
    p_aff.add_argument("--plot", action="store_true",
                       help="Save affinity heatmaps")

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    from src.serving.engine import (
        ServeConfig,
        build_llm,
        get_vllm_layout,
        run_concurrent_session,
        run_sequential_baseline,
    )
    from src.serving.capture import MultiTenantCapture, install_hooks
    from src.serving.workload import build_workload

    cfg = ServeConfig(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=not args.no_eager,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        seed=args.seed,
        num_tenants=args.tenants,
        schedule=args.schedule,
        rate=args.rate,
        mode=args.mode,
        prompts_file=args.prompts_file,
        no_chat=args.no_chat,
        system_prompt=args.system_prompt,
        output_dir=args.output_dir,
        family=args.family,
        base_prompt=args.base_prompt,
        slot_step=args.slot_step,
        greedy=args.greedy,
        seed_mode=args.seed_mode,
        prefix_caching=None if args.prefix_cache == "auto" else (args.prefix_cache == "on"),
    )

    workload = build_workload(
        num_tenants=cfg.num_tenants,
        schedule=cfg.schedule,
        rate=cfg.rate,
        seed=cfg.seed,
        prompts_file=cfg.prompts_file,
        family=cfg.family,
        base_prompt=cfg.base_prompt,
        slot_step=cfg.slot_step,
    )
    
    print(f"[workload] {cfg.num_tenants} tenants, {cfg.schedule} arrivals, "
          f"rate={cfg.rate} req/s, family={workload.family}, "
          f"greedy={cfg.greedy}, seed_mode={cfg.seed_mode}")
    for i, (tid, t, p) in enumerate(zip(workload.tenant_ids, workload.arrivals_s, workload.prompts)):
        sc = workload.slots_changed[i] if workload.slots_changed else 0
        tag = f"  Δslots={sc}" if workload.family == "similar" else ""
        print(f"  {tid:>9s} @ t={t:6.2f}s{tag}  {p[:60]}")

    llm = build_llm(cfg)
    capture = MultiTenantCapture()
    patched = install_hooks(llm.llm_engine, capture)
    if patched == 0:
        print("[error] Could not instrument MoE layers — no routing will be captured")
        return 1

    layout = get_vllm_layout(llm)
    layout["top_k"] = layout.get("top_k") or capture.top_k
    print(f"[layout] model_type={layout['model_type']}, layers={layout['num_layers']}, "
          f"experts={layout['num_experts']}, top_k={layout['top_k']}")

    if cfg.mode == "sequential":
        out_dir = run_sequential_baseline(cfg, llm, capture, layout, workload)
    else:
        out_dir = run_concurrent_session(cfg, llm, capture, layout, workload)

    print(f"\n[save] Session → {out_dir}")
    print(f"[stats] {capture.route_count} routing events, "
          f"{len(capture.steps)} engine steps, "
          f"{len(capture.request_ids())} tenants captured")

    from src.serving.analyze import analyze_session

    analyze_session(out_dir, baseline_dir=None, plot=True)
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from src.serving.analyze import analyze_session

    report = analyze_session(
        args.session_dir, baseline_dir=args.baseline, plot=args.plot
    )

    print("\n[report] Summary:")
    s = report["summary"]
    for key in (
        "num_tenants", "peak_concurrency", "total_time_s", "mean_ttft_s",
        "mean_tpot_s", "aggregate_throughput_tok_s", "multi_tenant_steps",
        "mean_expert_collision_ratio",
    ):
        print(f"  {key:<30s} {s[key]}")
    print("\n[report] Delay vs concurrency:")
    for conc, v in report["delay_vs_concurrency"].items():
        print(f"  concurrency={conc:<3s} mean_ttft={v['mean_ttft_s']:.3f}s  (n={v['n']})")
    if report.get("baseline_comparison"):
        print("\n[report] vs sequential baseline (TTFT slowdown):")
        for row in report["baseline_comparison"]:
            print(f"  tenant {row['tenant_idx']:>3d}: solo={row['ttft_solo_s']:.3f}s "
                  f"shared={row['ttft_shared_s']:.3f}s "
                  f"slowdown={row['slowdown']}x")
    return 0


def _cmd_affinity(args: argparse.Namespace) -> int:
    from src.serving.affinity import affinity_report

    report = affinity_report(args.session_dir, plot=args.plot)
    print("\n[affinity] Pairwise metrics (pos-overlap | same-token | JS div | corr):")
    for p in report["pairs"]:
        print(f"  {p['a']} <-> {p['b']}:  pos={p['topk_overlap']:.3f}  "
              f"tok={p['same_token_overlap']:.3f}  jsd={p['js_divergence']:.4f}  "
              f"corr={p['affinity_correlation']:.3f}")
    if report["edit_distance_curve"]:
        print("\n[affinity] Routing similarity vs prompt edit distance:")
        for d, v in report["edit_distance_curve"].items():
            print(f"  Δslots={d}:  pairs={v['n_pairs']}  pos_overlap={v['topk_overlap']:.3f}  "
                  f"tok_overlap={v['same_token_overlap']:.3f}  jsd={v['js_divergence']:.4f}  "
                  f"corr={v['affinity_correlation']:.3f}  plan_hit={v['plan_hit_rate']:.3f}")
    return 0


_COMMANDS = {
    "run": _cmd_run,
    "analyze": _cmd_analyze,
    "affinity": _cmd_affinity,
}


def main() -> int:
    args = _build_parser().parse_args()
    handler = _COMMANDS.get(args.command)
    if handler is None:
        return 1
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
