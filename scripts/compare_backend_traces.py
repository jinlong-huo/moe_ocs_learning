#!/usr/bin/env python3
"""Phase 2 — Cross-backend routing invariance: MLX vs vLLM-metal.

Same 4-bit weights, same prompt, greedy decoding on both engines.
Verifies the routing-independence assumption at the hardware/engine level:

  * prefill routing must be identical up to the quantized-GEMM noise floor
  * distribution-level metrics (JS divergence, affinity correlation,
    plan hit-rate) must pass the invariance contract

Contract (Phase 0):
  prompt_token_ids identical ............ hard requirement
  prefill js_divergence      ≤ 0.01
  prefill affinity_corr      ≥ 0.99
  prefill plan_hit_rate      = 1.0
  cell top-k overlap         ≥ 0.84  (measured Metal noise floor)
  cell top-(k-1) overlap     ≥ 0.95  (noise-robust variant)

Usage:
    .venv/bin/python scripts/compare_backend_traces.py \
        --a logs/phase2/mlx/routing.json \
        --b logs/phase2/run_uniform_1t/traces/tenant-000.json \
        --output logs/phase2/invariance_report.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.data.routing_schema import RoutingTrace  # noqa: E402
from src.serving.affinity import pairwise_metrics  # noqa: E402

CONTRACT = {
    "prefill_js_divergence_max": 0.01,
    "prefill_affinity_corr_min": 0.99,
    "prefill_plan_hit_rate_min": 1.0,
    "topk_overlap_noise_floor": 0.84,
    "topk_minus1_overlap_min": 0.95,
}


def prefill_trace(trace: RoutingTrace) -> RoutingTrace:
    n = trace.meta.prompt_len
    return dataclasses.replace(
        trace, routes=[r for r in trace.routes if r.token_pos < n]
    )


def compute(trace_a: RoutingTrace, trace_b: RoutingTrace, layers: list[str],
            top_k: int, label: str) -> dict:
    m_full = pairwise_metrics(trace_a, trace_b, trace_a.meta.num_experts,
                              layers, top_k)
    m_km1 = pairwise_metrics(trace_a, trace_b, trace_a.meta.num_experts,
                             layers, top_k, k_compare=top_k - 1)
    m_prefill = pairwise_metrics(prefill_trace(trace_a), prefill_trace(trace_b),
                                 trace_a.meta.num_experts, layers, top_k)
    return {
        "scope": label,
        "full_trace": {
            "cells_common": m_full["cells_common"],
            "topk_overlap": m_full["topk_overlap"],
            "same_token_overlap": m_full["same_token_overlap"],
            "js_divergence": m_full["js_divergence"],
            "affinity_correlation": m_full["affinity_correlation"],
            "plan_hit_rate": m_full["plan_hit_rate"],
        },
        "topk_minus1": {"topk_overlap": m_km1["topk_overlap"]},
        "prefill_only": {
            "cells_common": m_prefill["cells_common"],
            "topk_overlap": m_prefill["topk_overlap"],
            "js_divergence": m_prefill["js_divergence"],
            "affinity_correlation": m_prefill["affinity_correlation"],
            "plan_hit_rate": m_prefill["plan_hit_rate"],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="First trace (e.g. MLX)")
    ap.add_argument("--b", required=True, help="Second trace (e.g. vLLM)")
    ap.add_argument("--output", default="logs/phase2/invariance_report.json")
    args = ap.parse_args()

    a = RoutingTrace.load(args.a)
    b = RoutingTrace.load(args.b)
    assert a.meta.num_experts == b.meta.num_experts
    assert a.meta.top_k == b.meta.top_k

    print(f"[phase2] A: backend={a.meta.backend} prompt_len={a.meta.prompt_len} "
          f"gen={a.meta.generated_len} cells={sum(len(r.layers) for r in a.routes)}")
    print(f"[phase2] B: backend={b.meta.backend} prompt_len={b.meta.prompt_len} "
          f"gen={b.meta.generated_len} cells={sum(len(r.layers) for r in b.routes)}")

    prompts_equal = a.prompt_tokens == b.prompt_tokens
    print(f"[phase2] prompt_token_ids identical: {prompts_equal}")
    if not prompts_equal:
        n = min(len(a.prompt_tokens), len(b.prompt_tokens))
        diffs = [i for i in range(n) if a.prompt_tokens[i] != b.prompt_tokens[i]]
        print(f"[phase2]   first token diffs: {diffs[:10]}")

    layers = sorted({lid for r in a.routes for lid in r.layers})
    res = compute(a, b, layers, a.meta.top_k, "mlx_vs_vllm")

    pre = res["prefill_only"]
    checks = {
        "prompt_tokens_identical": prompts_equal,
        "prefill_js_divergence_pass": pre["js_divergence"] <= CONTRACT["prefill_js_divergence_max"],
        "prefill_affinity_corr_pass": pre["affinity_correlation"] >= CONTRACT["prefill_affinity_corr_min"],
        "prefill_plan_hit_rate_pass": pre["plan_hit_rate"] >= CONTRACT["prefill_plan_hit_rate_min"],
        "topk_overlap_at_or_above_noise_floor": res["full_trace"]["topk_overlap"] >= CONTRACT["topk_overlap_noise_floor"],
        "topk_minus1_overlap_pass": res["topk_minus1"]["topk_overlap"] >= CONTRACT["topk_minus1_overlap_min"],
    }
    verdict = all(checks.values())

    report = {
        "experiment": "phase2_cross_backend_invariance",
        "trace_a": args.a,
        "trace_b": args.b,
        "contract": CONTRACT,
        **res,
        "checks": checks,
        "verdict": {
            "routing_hardware_independent_up_to_noise_floor": verdict,
            "note": "distribution-level invariance must hold exactly; "
                    "cell-level is bounded by the Metal quantized-GEMM noise floor",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[phase2] scope         topk_ovlp  JS_div   corr     plan_hit")
    for scope, d in (("full", res["full_trace"]), ("prefill", pre)):
        print(f"[phase2] {scope:<10s} {d['topk_overlap']:<10.5f} {d['js_divergence']:<8.6f} "
              f"{d['affinity_correlation']:<8.5f} {d['plan_hit_rate']}")
    print(f"[phase2] top-(k-1) cell overlap: {res['topk_minus1']['topk_overlap']}")
    print(f"[phase2] checks: {checks}")
    print(f"[phase2] VERDICT: routing_hardware_independent_up_to_noise_floor={verdict}")
    print(f"[phase2] report → {out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
