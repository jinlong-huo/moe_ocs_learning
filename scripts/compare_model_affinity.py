#!/usr/bin/env python3
"""Phase 3 — Model-dependence control: same prompt, same hardware, different models.

Complements the invariance check. Routing = f(input, weights), so with
input and hardware held fixed, *changing the model must change the
affinity graph*.  This establishes that OCS preset plans are
model-specific: a plan derived from one model cannot serve another.

The two models have different expert spaces (60 vs 256), top-k (4 vs 8),
layers (24 vs 40), and tokenizers — so cross-model comparison is at the
*per-layer distribution shape* level, not cell level.

⚠️ Per-layer only (C2 in docs/research_assessment.md). Expert ids are
per-layer namespaces, so layer-POOLED statistics — pooled load entropy,
pooled top-5 share, cross-layer JS, pooled off-diagonal affinity —
average unrelated distributions and saturate. Those metrics were
discarded, not recalibrated. Everything reported here is computed within
a single layer and then summarised across layers.

For model separation on real workloads the authoritative evidence is the
workload chain (Q2 category decoding, `verify_live_invariance.py`) and
per-expert category-KL (`src/eval/specialization.py`), which is the C4
replacement for the discarded entropy comparison.

Usage:
    .venv/bin/python scripts/compare_model_affinity.py \
        --small logs/phase2/mlx/routing.json \
        --large logs/phase3/large/routing.json \
        --output logs/phase3/model_diversity_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.data.routing_schema import RoutingTrace  # noqa: E402
from src.serving.affinity import (  # noqa: E402
    _route_cells,
    expert_distribution,
)


def model_profile(trace: RoutingTrace, label: str) -> dict:
    """Per-layer load statistics, summarised across layers.

    All distribution stats are computed WITHIN one layer (a single expert-id
    namespace) and then averaged — never pooled across layers.
    """
    cells = _route_cells(trace)
    num_experts = trace.meta.num_experts

    layers = sorted({int(lid) for _, _, lid, _, _ in cells})
    per_layer_ids: dict[int, list[int]] = {
        lid: [e for _, _, l, experts, _ in cells if l == lid for e in experts]
        for lid in layers
    }

    gini_vals, maxload_vals, top5_vals = [], [], []
    for lid in layers:
        dist = expert_distribution([per_layer_ids[lid]], num_experts)
        nonzero = dist[dist > 0]
        if nonzero.size == 0:
            continue

        # Gini concentration within this layer's namespace
        lorenz = np.cumsum(np.sort(dist))
        gini_vals.append(float(1 - 2 * np.mean(lorenz[:-1])) if num_experts > 1 else 0.0)

        # Peak load vs uniform within this layer
        maxload_vals.append(float(dist.max() * num_experts))

        # Top-5 share within this layer (namespace-consistent)
        top5_vals.append(float(np.sort(dist)[::-1][:5].sum()))

    used = len({e for ids in per_layer_ids.values() for e in ids})

    return {
        "label": label,
        "model_id": trace.meta.model_id,
        "num_layers": trace.meta.num_layers,
        "num_experts": num_experts,
        "top_k": trace.meta.top_k,
        "prompt_len": trace.meta.prompt_len,
        "generated_len": trace.meta.generated_len,
        "cells": len(cells),
        "used_experts_union": used,
        "used_fraction_union": round(used / num_experts, 4),
        "layer_gini_mean": round(float(np.mean(gini_vals)), 4),
        "layer_maxload_vs_uniform_mean": round(float(np.mean(maxload_vals)), 4),
        "layer_maxload_vs_uniform_max": round(float(np.max(maxload_vals)), 4),
        "layer_top5_share_mean": round(float(np.mean(top5_vals)), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--output", default="logs/phase3/model_diversity_report.json")
    args = ap.parse_args()

    small = RoutingTrace.load(args.small)
    large = RoutingTrace.load(args.large)

    p_small = model_profile(small, "small")
    p_large = model_profile(large, "large")

    # divergence between the two routing shapes (per-layer effect sizes)
    maxload_ratio = (p_small["layer_maxload_vs_uniform_mean"]
                     / max(p_large["layer_maxload_vs_uniform_mean"], 1e-12))
    gini_abs_diff = abs(p_small["layer_gini_mean"] - p_large["layer_gini_mean"])
    top5_rel = abs(p_small["layer_top5_share_mean"] - p_large["layer_top5_share_mean"]) \
        / max(p_small["layer_top5_share_mean"], p_large["layer_top5_share_mean"], 1e-12)

    effect_sizes = {
        "layer_maxload_ratio_small_over_large": round(maxload_ratio, 2),
        "layer_gini_abs_diff": round(gini_abs_diff, 4),
        "layer_top5_share_rel_diff": round(top5_rel, 4),
    }
    # The per-layer load profile is where models genuinely separate (measured:
    # max expert load 1.64x vs 10.54x uniform for Qwen1.5 vs Qwen3.6, C4).
    diverged = maxload_ratio > 1.5 or gini_abs_diff > 0.10

    print(f"[phase3] {'model':<10s} {'experts':>7s} {'top_k':>5s} {'layers':>6s} "
          f"{'prompt':>6s} {'used%':>6s} {'gini':>6s} {'maxload':>8s} {'top5%':>6s}"
          f"   (per-layer means)")
    for p in (p_small, p_large):
        print(f"[phase3] {p['label']:<10s} {p['num_experts']:>7d} {p['top_k']:>5d} "
              f"{p['num_layers']:>6d} {p['prompt_len']:>6d} "
              f"{p['used_fraction_union']:>6.3f} {p['layer_gini_mean']:>6.4f} "
              f"{p['layer_maxload_vs_uniform_mean']:>8.4f} "
              f"{p['layer_top5_share_mean']:>6.4f}")

    report = {
        "experiment": "phase3_model_dependence_control",
        "hold_fixed": "same prompt, same hardware/backend (MLX), greedy",
        "varied": "model weights",
        "note": "expert spaces, top-k and tokenizers differ between models — "
                "comparison is at the PER-LAYER distribution-shape level "
                "(layer-pooled statistics are saturated by construction, C2). "
                "Authoritative model separation on real workloads: Q2 category "
                "decoding + per-expert category-KL in the workload chain.",
        "small_model": p_small,
        "large_model": p_large,
        "effect_sizes": effect_sizes,
        "verdict": {
            "routing_diverges_across_models": diverged,
            "implication": "OCS preset plans are model-specific — must be "
                           "re-derived per model (weights)",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[phase3] VERDICT: routing_diverges_across_models={diverged}")
    print(f"[phase3] report → {out}")
    return 0 if diverged else 1


if __name__ == "__main__":
    raise SystemExit(main())
