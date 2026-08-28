#!/usr/bin/env python3
"""routing_map.py — visual correctness check for captured RoutingTraces.

Two modes:

1. Paired cross-backend map (--a A.json --b B.json):
   side-by-side top-1 expert maps (token x layer) for the two engines plus a
   per-cell top-k overlap map, so "routing stays still across backends" can be
   inspected by eye and quantified per layer.

2. Single-trace map (--trace T.json):
   top-1 expert map for one capture_workload trace with the prefill/decode
   boundary, plus a per-layer active-expert profile and inline sanity checks
   (expert-id range, top_k, weight sums).

Usage:
    .venv/bin/python scripts/routing_map.py \
        --a logs/phase2/mlx/routing.json \
        --b logs/phase2/run_uniform_1t/traces/tenant-000.json \
        --out logs/phase2/routing_map.png

    .venv/bin/python scripts/routing_map.py \
        --trace logs/workload/qwen15/traces/cat.code_debugging.00.json \
        --out logs/workload/qwen15/routing_map.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.data.routing_schema import RoutingTrace  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# matrix builders
# ═══════════════════════════════════════════════════════════════════

def build_cell_index(trace: RoutingTrace) -> dict[tuple[int, str], tuple[list, list]]:
    """(token_pos, layer) -> (experts, weights)."""
    idx = {}
    for r in trace.routes:
        for lid, lr in r.layers.items():
            idx[(r.token_pos, int(lid))] = (lr.experts, lr.weights)
    return idx


def sanity_checks(trace: RoutingTrace, name: str) -> list[str]:
    issues = []
    ne, tk = trace.meta.num_experts, trace.meta.top_k
    n_cells = bad_id = bad_k = 0
    w_sums = []
    for r in trace.routes:
        for lr in r.layers.values():
            n_cells += 1
            if len(lr.experts) != tk:
                bad_k += 1
            if any(not (0 <= e < ne) for e in lr.experts):
                bad_id += 1
            if lr.weights:
                w_sums.append(sum(lr.weights))
    ws = np.array(w_sums) if w_sums else np.array([0.0])
    print(f"[{name}] cells={n_cells} experts-in-range_ok={bad_id == 0} "
          f"top_k=={tk}_ok={bad_k == 0}")
    print(f"[{name}] weight-sum mean={ws.mean():.4f} min={ws.min():.4f} "
          f"max={ws.max():.4f} (renorm only if norm_topk_prob)")
    if bad_id:
        issues.append(f"{bad_id} cells with expert id outside [0,{ne})")
    if bad_k:
        issues.append(f"{bad_k} cells with top_k != {tk}")
    return issues


def paired_matrices(a: RoutingTrace, b: RoutingTrace):
    ia, ib = build_cell_index(a), build_cell_index(b)
    layers = sorted({lid for (_, lid) in ia.keys() & ib.keys()})
    positions = sorted({p for (p, _) in ia.keys() & ib.keys()})
    L, T = len(layers), len(positions)
    li = {l: i for i, l in enumerate(layers)}

    top1_a = np.full((L, T), np.nan)
    top1_b = np.full((L, T), np.nan)
    ovlp = np.full((L, T), np.nan)
    k = a.meta.top_k
    for p_i, p in enumerate(positions):
        for l in layers:
            ea, wa = ia.get((p, l), ([], []))
            eb, wb = ib.get((p, l), ([], []))
            if not ea or not eb:
                continue
            top1_a[li[l], p_i] = ea[0]
            top1_b[li[l], p_i] = eb[0]
            ovlp[li[l], p_i] = len(set(ea) & set(eb)) / max(k, 1)
    return (np.array(layers), np.array(positions), top1_a, top1_b, ovlp, k)


def expert_map_ax(ax, mat, positions, layers, ne, title, prompt_len=None,
                  vmin=0, vmax=None, cmap="viridis", cbar_label="expert id"):
    im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                   vmin=vmin, vmax=vmax if vmax is not None else ne - 1, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("MoE layer")
    ax.set_xticks(np.arange(0, len(positions), 8))
    ax.set_xticklabels([str(p) for p in positions[::8]], fontsize=7)
    ax.set_yticks(np.arange(0, len(layers), 4))
    ax.set_yticklabels([str(l) for l in layers[::4]], fontsize=7)
    if prompt_len is not None and prompt_len in list(positions):
        pi = list(positions).index(prompt_len)
        ax.axvline(pi - 0.5, color="red", lw=1.2, ls="--")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label=cbar_label)


# ═══════════════════════════════════════════════════════════════════
# mode 1 — paired cross-backend map
# ═══════════════════════════════════════════════════════════════════

def render_pair(path_a: str, path_b: str, out: Path) -> int:
    a, b = RoutingTrace.load(path_a), RoutingTrace.load(path_b)
    assert a.meta.num_experts == b.meta.num_experts and a.meta.top_k == b.meta.top_k
    sanity_checks(a, "A:" + a.meta.backend)
    sanity_checks(b, "B:" + b.meta.backend)

    same_prompt = a.prompt_tokens == b.prompt_tokens
    print(f"[pair] prompt_tokens identical: {same_prompt}")
    layers, positions, top1_a, top1_b, ovlp, k = paired_matrices(a, b)
    pre = a.meta.prompt_len
    is_pre = np.array([p < pre for p in positions])

    pre_mask = ovlp[:, is_pre]
    dec_mask = ovlp[:, ~is_pre]
    agree_pre = float(np.nanmean(pre_mask))
    agree_dec = float(np.nanmean(dec_mask))
    exact_pre = float(np.mean(pre_mask == 1.0))
    exact_all = float(np.nanmean(ovlp == 1.0))
    print(f"[pair] mean top-k overlap  prefill={agree_pre:.4f}  decode={agree_dec:.4f}")
    print(f"[pair] exact-set rate      prefill={exact_pre:.4f}  all={exact_all:.4f}")

    per_layer_agree = np.nanmean(ovlp, axis=1)
    worst = np.argsort(per_layer_agree)[:5]
    print("[pair] worst layers by agreement:",
          [(int(layers[i]), round(float(per_layer_agree[i]), 3)) for i in worst])

    fig, axes = plt.subplots(3, 1, figsize=(15, 9.5),
                             gridspec_kw={"height_ratios": [1, 1, 1]})
    common = dict(positions=positions, layers=layers, ne=a.meta.num_experts,
                  prompt_len=pre)
    expert_map_ax(axes[0], top1_a, title=f"top-1 expert — backend A: {a.meta.backend}",
                  **common)
    expert_map_ax(axes[1], top1_b, title=f"top-1 expert — backend B: {b.meta.backend}",
                  **common)
    expert_map_ax(axes[2], ovlp, title=f"top-{k} set overlap (1 = identical expert set)",
                  ne=a.meta.num_experts, vmin=0, vmax=1, cmap="magma_r",
                  cbar_label="overlap", positions=positions, layers=layers,
                  prompt_len=pre)
    axes[2].set_xlabel("token position  (red dashed = prefill | decode boundary)")

    fig.suptitle(f"Cross-backend routing map — {a.meta.model_id} "
                 f"(prompt identical: {same_prompt}) | "
                 f"prefill overlap {agree_pre:.3f}, decode {agree_dec:.3f}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[pair] map -> {out}")
    return 0


# ═══════════════════════════════════════════════════════════════════
# mode 2 — single-trace map (capture_workload output)
# ═══════════════════════════════════════════════════════════════════

def render_single(path: str, out: Path) -> int:
    t = RoutingTrace.load(path)
    sanity_checks(t, t.meta.backend)

    layers = sorted({int(lid) for r in t.routes for lid in r.layers})
    positions = [r.token_pos for r in t.routes]
    L, T, ne = len(layers), len(positions), t.meta.num_experts
    li = {l: i for i, l in enumerate(layers)}
    top1 = np.full((L, T), np.nan)
    for c, r in enumerate(t.routes):
        for lid, lr in r.layers.items():
            if lr.experts:
                top1[li[int(lid)], c] = lr.experts[0]

    usage = np.zeros((L, ne))
    for c, r in enumerate(t.routes):
        for lid, lr in r.layers.items():
            for e in lr.experts:
                usage[li[int(lid)], e] += 1
    active = (usage > 0).sum(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(15, 7),
                             gridspec_kw={"height_ratios": [2.2, 1]})
    expert_map_ax(axes[0], top1, positions=positions, layers=layers, ne=ne,
                  title=f"top-1 expert map — {t.meta.model_id} ({t.meta.backend})",
                  prompt_len=t.meta.prompt_len)
    axes[0].set_xlabel("")

    axes[1].bar(np.arange(L), active, color="steelblue")
    axes[1].set_xlabel(f"token position 0..{T-1}  →  per-MoE-layer active experts "
                       f"(red dashed = prefill | decode)")
    axes[1].set_ylabel("# distinct experts used")
    axes[1].set_xticks(np.arange(0, T, 8))
    axes[1].set_xticklabels([str(p) for p in positions[::8]], fontsize=7)
    axes[1].axvline(t.meta.prompt_len - 0.5, color="red", lw=1.2, ls="--")
    axes[1].set_xlim(-0.5, L - 0.5)
    axes[1].set_title(f"active experts per layer: min={active.min()} "
                      f"max={active.max()} of {ne}", fontsize=9)

    fig.suptitle(f"Routing map — {path} | prompt_len={t.meta.prompt_len} "
                 f"gen={t.meta.generated_len} cells={sum(len(r.layers) for r in t.routes)}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[single] map -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--trace")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.a and args.b:
        return render_pair(args.a, args.b, Path(args.out))
    if args.trace:
        return render_single(args.trace, Path(args.out))
    ap.error("provide --a/--b (paired mode) or --trace (single mode)")


if __name__ == "__main__":
    raise SystemExit(main())
