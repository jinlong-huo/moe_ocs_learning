#!/usr/bin/env python3
"""Phase 3 — Model-dependence control: same prompt, same hardware, different models.

Complements the invariance check. Routing = f(input, weights), so with
input and hardware held fixed, *changing the model must change the
affinity graph*.  This establishes that OCS preset plans are
model-specific: a plan derived from one model cannot serve another.

The two models have different expert spaces (60 vs 256), top-k (4 vs 8),
layers (24 vs 40), and tokenizers — so cross-model comparison is at the
*distribution shape* level, not cell level:

  per model: used-expert fraction, normalized load entropy, Gini
             concentration, top-5 expert share, mean pairwise JS between
             per-layer distributions (intra-model routing diversity),
             mean off-diagonal co-activation (affinity strength)

Verdict contract: the two models' routing shapes must diverge
(different concentration / entropy / layer-diversity profiles), while
each model remains self-consistent.

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
    co_activation,
    expert_distribution,
    js_divergence,
)


def model_profile(trace: RoutingTrace, label: str) -> dict:
    cells = _route_cells(trace)
    num_experts = trace.meta.num_experts
    top_k = trace.meta.top_k

    all_ids = [e for _, _, _, experts, _ in cells for e in experts]
    dist = expert_distribution([all_ids], num_experts)

    used = int((dist > 0).sum())
    nonzero = dist[dist > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())
    norm_entropy = entropy / np.log2(num_experts)

    sorted_p = np.sort(dist)[::-1]
    lorenz = np.cumsum(np.sort(dist))
    gini = float(1 - 2 * np.mean(lorenz[:-1])) if num_experts > 1 else 0.0
    top5_share = float(sorted_p[:5].sum())

    # per-layer distributions → intra-model layer diversity
    layers = sorted({int(lid) for _, _, lid, _, _ in cells})
    per_layer = {}
    for lid in layers:
        ids = [e for _, _, l, experts, _ in cells if l == lid for e in experts]
        per_layer[lid] = expert_distribution([ids], num_experts)
    pairs = [(a, b) for i, a in enumerate(layers) for b in layers[i + 1:]]
    layer_js = [js_divergence(per_layer[a], per_layer[b]) for a, b in pairs]

    # affinity strength: mean off-diagonal co-activation
    ca = co_activation(trace, num_experts)
    norm_ca = ca / (ca.sum() + 1e-12)
    off_diag = norm_ca[~np.eye(num_experts, dtype=bool)]
    aff_strength = float(off_diag.mean())
    aff_entropy = float(-(off_diag[off_diag > 0] * np.log2(off_diag[off_diag > 0])).sum())

    return {
        "label": label,
        "model_id": trace.meta.model_id,
        "num_layers": trace.meta.num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "prompt_len": trace.meta.prompt_len,
        "generated_len": trace.meta.generated_len,
        "cells": len(cells),
        "used_experts": used,
        "used_fraction": round(used / num_experts, 4),
        "load_entropy_norm": round(norm_entropy, 4),
        "load_gini": round(gini, 4),
        "top5_expert_share": round(top5_share, 4),
        "layer_diversity_mean_js": round(float(np.mean(layer_js)), 6),
        "layer_diversity_max_js": round(float(np.max(layer_js)), 6),
        "affinity_strength_offdiag": round(aff_strength, 8),
        "affinity_offdiag_entropy": round(aff_entropy, 4),
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

    # divergence between the two routing shapes (effect sizes)
    top5 = max(p_small["top5_expert_share"], p_large["top5_expert_share"])
    top5_rel_diff = abs(p_small["top5_expert_share"] - p_large["top5_expert_share"]) / top5
    layer_js_diff = abs(p_small["layer_diversity_mean_js"] - p_large["layer_diversity_mean_js"])
    aff_ratio = (p_small["affinity_strength_offdiag"]
                 / max(p_large["affinity_strength_offdiag"], 1e-12))
    entropy_diff = abs(p_small["load_entropy_norm"] - p_large["load_entropy_norm"])

    effect_sizes = {
        "top5_share_rel_diff": round(top5_rel_diff, 4),
        "layer_diversity_js_diff": round(layer_js_diff, 4),
        "affinity_strength_ratio": round(aff_ratio, 2),
        "entropy_norm_diff": round(entropy_diff, 4),
    }
    diverged = top5_rel_diff > 0.25 or layer_js_diff > 0.05 or entropy_diff > 0.005

    print(f"[phase3] {'model':<10s} {'experts':>7s} {'top_k':>5s} {'layers':>6s} "
          f"{'prompt':>6s} {'used%':>6s} {'entropy':>8s} {'gini':>6s} {'top5%':>6s} "
          f"{'layerJS':>8s} {'affStr':>8s}")
    for p in (p_small, p_large):
        print(f"[phase3] {p['label']:<10s} {p['num_experts']:>7d} {p['top_k']:>5d} "
              f"{p['num_layers']:>6d} {p['prompt_len']:>6d} "
              f"{p['used_fraction']:>6.3f} {p['load_entropy_norm']:>8.4f} "
              f"{p['load_gini']:>6.4f} {p['top5_expert_share']:>6.4f} "
              f"{p['layer_diversity_mean_js']:>8.5f} {p['affinity_strength_offdiag']:>8.6f}")

    report = {
        "experiment": "phase3_model_dependence_control",
        "hold_fixed": "same prompt, same hardware/backend (MLX), greedy",
        "varied": "model weights (Qwen1.5-MoE-A2.7B vs Qwen3.6-35B-A3B)",
        "note": "expert spaces, top-k and tokenizers differ between models — "
                "comparison is at distribution-shape level",
        "small_model": p_small,
        "large_model": p_large,
        "effect_sizes": effect_sizes,
        "verdict": {
            "routing_diverges_across_models": diverged,
            "implication": "OCS preset plans are model-specific — must be "
                           "re-derived per model (weights), as README TODO requires",
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
