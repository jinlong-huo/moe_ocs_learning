#!/usr/bin/env python3
"""Visualize MoE expert co-activation affinity and train/inference discrepancy.

Generates a self-contained HTML report with:
  1. Rank-pair communication heatmaps (train, inference, delta)
  2. Affinity correlation scatter (train vs infer per rank-pair)
  3. Per-layer JS divergence (routing distribution shift)
  4. Expert load distribution comparison
  5. Preset circuit coverage analysis (if plan file provided)

Usage:
  # Compare two traces (train vs inference):
  python scripts/viz_affinity.py \
      --train-trace data/routing_traces/routing_pretrained.json \
      --infer-trace data/routing_traces/routing.json \
      --preset-plan outputs/preset_plan.json

  # Single trace (affinity exploration):
  python scripts/viz_affinity.py \
      --train-trace data/routing_traces/routing.json

  # Three-way: pretrained vs finetuned + plan overlay:
  python scripts/viz_affinity.py \
      --train-trace data/routing_traces/routing_pretrained.json \
      --infer-trace data/routing_traces/routing_finetuned.json \
      --preset-plan outputs/preset_plan.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.routing_schema import RoutingTrace
from src.ocs.placement import ExpertAffinityTracker
from src.eval.affinity_consistency import js_divergence, affinity_correlation, estimated_hit_rate


# ═══════════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════════

def build_rank_comm_matrix(
    trace: RoutingTrace,
    experts_per_rank: int,
) -> np.ndarray:
    """Build rank-pair communication frequency matrix from a routing trace.

    For each token-layer, maps co-selected experts to their owner ranks and
    accumulates (src_rank, dst_rank) communication counts.

    Returns [num_ranks, num_ranks] float64 matrix normalized by total.
    """
    num_experts = trace.meta.num_experts
    num_ranks = (num_experts + experts_per_rank - 1) // experts_per_rank

    matrix = np.zeros((num_ranks, num_ranks), dtype=np.float64)

    for route in trace.routes:
        for _layer_id, layer_data in route.layers.items():
            experts = layer_data.experts
            weights = layer_data.weights if layer_data.weights else [1.0] * len(experts)

            for i, ea in enumerate(experts):
                ra = ea // experts_per_rank
                for j, eb in enumerate(experts):
                    rb = eb // experts_per_rank
                    if ra != rb:
                        # Weight by gate scores for a more nuanced signal
                        w = min(weights[i], weights[j]) if weights else 1.0
                        matrix[ra, rb] += w

    # Normalize
    total = matrix.sum()
    if total > 0:
        matrix /= total

    return matrix


def build_expert_load(trace: RoutingTrace) -> np.ndarray:
    """Build per-expert selection frequency distribution.

    Returns [num_experts] float64 array summing to 1.
    """
    num_experts = trace.meta.num_experts
    counts = np.zeros(num_experts, dtype=np.float64)

    for route in trace.routes:
        for _layer_id, layer_data in route.layers.items():
            for e in layer_data.experts:
                if 0 <= e < num_experts:
                    counts[e] += 1.0

    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def build_per_layer_expert_dist(trace: RoutingTrace) -> dict[int, np.ndarray]:
    """Build per-layer expert selection distribution.

    Returns {layer_idx: np.ndarray[num_experts]}.
    """
    num_experts = trace.meta.num_experts
    per_layer: dict[int, dict[int, float]] = {}

    for route in trace.routes:
        for lid_str, layer_data in route.layers.items():
            lid = int(lid_str)
            if lid not in per_layer:
                per_layer[lid] = {e: 0.0 for e in range(num_experts)}
            for e in layer_data.experts:
                if 0 <= e < num_experts:
                    per_layer[lid][e] += 1.0

    result = {}
    for lid, counts in sorted(per_layer.items()):
        arr = np.array([counts[e] for e in range(num_experts)], dtype=np.float64)
        total = arr.sum()
        if total > 0:
            arr /= total
        result[lid] = arr

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.labelcolor": "#c9d1d9",
    "font.size": 10,
})


def _fig_to_b64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _rank_labels(num_ranks: int, experts_per_rank: int) -> list[str]:
    """Human-readable rank labels showing expert range."""
    return [f"R{r}\n(e{r*experts_per_rank}-e{r*experts_per_rank+experts_per_rank-1})"
            for r in range(num_ranks)]


# ═══════════════════════════════════════════════════════════════════════════
# Chart generators — each returns a base64 PNG string
# ═══════════════════════════════════════════════════════════════════════════

def chart_rank_heatmap(
    matrix: np.ndarray,
    title: str,
    experts_per_rank: int,
    preset_pairs: set | None = None,
    cmap_name: str = "YlOrRd",
) -> str:
    """Rank-pair communication heatmap. Optionally overlays preset circuit markers."""
    num_ranks = matrix.shape[0]
    labels = _rank_labels(num_ranks, experts_per_rank)

    fig, ax = plt.subplots(figsize=(max(7, num_ranks * 1.2), max(6, num_ranks * 1.1)))
    cmap = plt.colormaps[cmap_name].copy()
    cmap.set_under("#161b22")

    im = ax.imshow(matrix, cmap=cmap, aspect="equal", norm=mcolors.LogNorm(
        vmin=max(matrix[matrix > 0].min() if matrix.any() > 0 else 1e-6, 1e-6),
        vmax=matrix.max(),
    ))

    # Annotate cells
    for i in range(num_ranks):
        for j in range(num_ranks):
            val = matrix[i, j]
            if val > 0:
                text_color = "white" if val > matrix.max() * 0.6 else "#c9d1d9"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")

    # Overlay preset circuit markers
    if preset_pairs:
        for src, dst in preset_pairs:
            if 0 <= src < num_ranks and 0 <= dst < num_ranks:
                rect = plt.Rectangle((dst - 0.5, src - 0.5), 1, 1,
                                     fill=False, edgecolor="#3fb950", linewidth=2.5,
                                     linestyle="--")
                ax.add_patch(rect)

    ax.set_xticks(range(num_ranks))
    ax.set_yticks(range(num_ranks))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#58a6ff", pad=15)
    ax.set_xlabel("Destination Rank", fontsize=10)
    ax.set_ylabel("Source Rank", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Communication Frequency (log scale)", color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    cbar.outline.set_edgecolor("#30363d")

    if preset_pairs:
        legend_elements = [Patch(facecolor="none", edgecolor="#3fb950", linewidth=2.5,
                                 linestyle="--", label="Preset Circuit")]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8,
                  bbox_to_anchor=(0, -0.12))

    return _fig_to_b64(fig)


def chart_delta_heatmap(
    train_matrix: np.ndarray,
    infer_matrix: np.ndarray,
    experts_per_rank: int,
) -> str:
    """Heatmap showing train - infer discrepancy per rank pair."""
    delta = train_matrix - infer_matrix
    num_ranks = delta.shape[0]
    labels = _rank_labels(num_ranks, experts_per_rank)

    # Symmetric around zero: blue = train overestimates, red = train underestimates
    abs_max = max(abs(delta.min()), abs(delta.max()), 1e-6)

    fig, ax = plt.subplots(figsize=(max(7, num_ranks * 1.2), max(6, num_ranks * 1.1)))
    im = ax.imshow(delta, cmap="RdBu_r", aspect="equal", vmin=-abs_max, vmax=abs_max)

    for i in range(num_ranks):
        for j in range(num_ranks):
            val = delta[i, j]
            if abs(val) > abs_max * 0.05:
                text_color = "white" if abs(val) > abs_max * 0.7 else "#c9d1d9"
                ax.text(j, i, f"{val:+.4f}", ha="center", va="center",
                        fontsize=7, color=text_color, fontweight="bold")

    ax.set_xticks(range(num_ranks))
    ax.set_yticks(range(num_ranks))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Affinity Delta (Train − Inference)", fontsize=13, fontweight="bold",
                 color="#58a6ff", pad=15)
    ax.set_xlabel("Destination Rank", fontsize=10)
    ax.set_ylabel("Source Rank", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Δ Communication Frequency", color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    cbar.outline.set_edgecolor("#30363d")

    # Annotation
    ax.text(0.5, -0.18, "Blue = train over-estimates (preset may waste circuits)\n"
            "Red = train under-estimates (preset may miss circuits)",
            transform=ax.transAxes, ha="center", fontsize=8, color="#8b949e",
            va="top")

    return _fig_to_b64(fig)


def chart_correlation_scatter(
    train_matrix: np.ndarray,
    infer_matrix: np.ndarray,
) -> str:
    """Scatter plot: each point is a rank pair, showing train vs infer affinity."""
    num_ranks = train_matrix.shape[0]
    xs, ys = [], []
    for i in range(num_ranks):
        for j in range(num_ranks):
            if i != j:
                xs.append(train_matrix[i, j])
                ys.append(infer_matrix[i, j])

    xs = np.array(xs)
    ys = np.array(ys)

    # Filter zero-zero pairs
    mask = (xs > 0) | (ys > 0)
    xs_f = xs[mask]
    ys_f = ys[mask]

    if len(xs_f) < 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "Not enough data for correlation scatter",
                ha="center", va="center", transform=ax.transAxes, color="#8b949e")
        return _fig_to_b64(fig)

    corr = float(np.corrcoef(xs_f, ys_f)[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xs_f, ys_f, alpha=0.7, s=40, c="#58a6ff", edgecolors="none")

    # Diagonal line (perfect correlation)
    lim_max = max(xs_f.max(), ys_f.max()) * 1.1
    ax.plot([0, lim_max], [0, lim_max], "--", color="#30363d", linewidth=1.5, alpha=0.7,
            label=f"Perfect correlation (y=x)")

    # Best-fit line
    if len(xs_f) > 1:
        m, b = np.polyfit(xs_f, ys_f, 1)
        ax.plot([0, lim_max], [b, m * lim_max + b], "-", color="#f85149", linewidth=1.5,
                alpha=0.8, label=f"Best fit (y={m:.2f}x+{b:.4f})")

    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_xlabel("Training Affinity", fontsize=10)
    ax.set_ylabel("Inference Affinity", fontsize=10)
    ax.set_title(f"Affinity Correlation (r = {corr:.4f})", fontsize=13,
                 fontweight="bold", color="#58a6ff")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")

    # Annotate r value
    ax.text(0.95, 0.05, f"Pearson r = {corr:.4f}\n"
            f"n = {len(xs_f)} rank-pairs",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#8b949e",
            bbox=dict(facecolor="#161b22", edgecolor="#30363d", pad=8, alpha=0.9))

    return _fig_to_b64(fig)


def chart_per_layer_js_divergence(
    train_trace: RoutingTrace,
    infer_trace: RoutingTrace,
) -> str:
    """Per-layer Jensen-Shannon divergence bar chart."""
    train_per_layer = build_per_layer_expert_dist(train_trace)
    infer_per_layer = build_per_layer_expert_dist(infer_trace)

    # Find common layers
    common_layers = sorted(set(train_per_layer.keys()) & set(infer_per_layer.keys()))
    if not common_layers:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No common layers between traces",
                ha="center", va="center", transform=ax.transAxes, color="#8b949e")
        return _fig_to_b64(fig)

    js_vals = []
    for lid in common_layers:
        js = js_divergence(train_per_layer[lid], infer_per_layer[lid])
        js_vals.append(js)

    # Color by severity
    colors = []
    for v in js_vals:
        if v < 0.1:
            colors.append("#3fb950")   # green: high consistency
        elif v < 0.3:
            colors.append("#d29922")   # yellow: moderate
        else:
            colors.append("#f85149")   # red: high divergence

    fig, ax = plt.subplots(figsize=(max(8, len(common_layers) * 0.25), 5))
    bars = ax.bar(range(len(common_layers)), js_vals, color=colors, edgecolor="#0d1117",
                  linewidth=0.5)

    # Threshold lines
    ax.axhline(y=0.1, color="#3fb950", linestyle="--", linewidth=1, alpha=0.5, label="Low divergence (0.1)")
    ax.axhline(y=0.3, color="#d29922", linestyle="--", linewidth=1, alpha=0.5, label="Moderate (0.3)")

    mean_js = np.mean(js_vals)
    ax.axhline(y=mean_js, color="#58a6ff", linestyle="-", linewidth=1.5, alpha=0.7,
               label=f"Mean JS = {mean_js:.4f}")

    ax.set_xticks(range(0, len(common_layers), max(1, len(common_layers) // 20)))
    ax.set_xticklabels([str(common_layers[i]) for i in
                        range(0, len(common_layers), max(1, len(common_layers) // 20))],
                       fontsize=8)
    ax.set_xlabel("Layer Index", fontsize=10)
    ax.set_ylabel("JS Divergence", fontsize=10)
    ax.set_title(f"Per-Layer Routing Distribution Shift ({len(common_layers)} layers)",
                 fontsize=13, fontweight="bold", color="#58a6ff")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.15, axis="y")
    ax.set_ylim(0, max(1.0, max(js_vals) * 1.1))

    # Top-5 most divergent layers annotation
    ranked = sorted(enumerate(js_vals), key=lambda x: -x[1])[:5]
    text_lines = ["Most divergent layers:"]
    for idx, v in ranked:
        text_lines.append(f"  Layer {common_layers[idx]}: JS={v:.4f}")
    ax.text(0.98, 0.95, "\n".join(text_lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#8b949e",
            bbox=dict(facecolor="#161b22", edgecolor="#30363d", pad=6, alpha=0.9))

    return _fig_to_b64(fig)


def chart_expert_load_comparison(
    train_load: np.ndarray,
    infer_load: np.ndarray,
    top_n: int = 30,
) -> str:
    """Grouped bar chart comparing top-N expert loads between train and inference."""
    num_experts = len(train_load)

    # Find top-N experts by average load
    avg_load = (train_load + infer_load) / 2
    top_indices = np.argsort(-avg_load)[:top_n]

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.35), 6))
    x = np.arange(top_n)
    width = 0.35

    bars1 = ax.bar(x - width/2, train_load[top_indices] * 100, width,
                   label="Training", color="#58a6ff", edgecolor="#0d1117", linewidth=0.5)
    bars2 = ax.bar(x + width/2, infer_load[top_indices] * 100, width,
                   label="Inference", color="#f0883e", edgecolor="#0d1117", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"E{i}" for i in top_indices], fontsize=7, rotation=90)
    ax.set_xlabel("Expert ID", fontsize=10)
    ax.set_ylabel("Selection Frequency (%)", fontsize=10)
    ax.set_title(f"Expert Load Distribution (Top {top_n} of {num_experts})",
                 fontsize=13, fontweight="bold", color="#58a6ff")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.15, axis="y")

    # JS divergence for expert load
    js_global = js_divergence(train_load, infer_load)
    ax.text(0.98, 0.95, f"JS divergence = {js_global:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#8b949e",
            bbox=dict(facecolor="#161b22", edgecolor="#30363d", pad=6, alpha=0.9))

    return _fig_to_b64(fig)


def chart_preset_coverage(
    train_matrix: np.ndarray,
    infer_matrix: np.ndarray | None,
    preset_pairs: set,
    experts_per_rank: int,
) -> str:
    """Bar chart: rank-pair communication frequency, highlighting preset coverage."""
    num_ranks = train_matrix.shape[0]

    # Flatten rank pairs (excluding diagonal)
    pairs = []
    for i in range(num_ranks):
        for j in range(num_ranks):
            if i != j:
                train_val = train_matrix[i, j]
                infer_val = infer_matrix[i, j] if infer_matrix is not None else train_val
                in_preset = (i, j) in preset_pairs
                pairs.append((i, j, train_val, infer_val, in_preset))

    # Sort by inference frequency
    pairs.sort(key=lambda x: -x[3])
    n_show = min(30, len(pairs))

    fig, ax = plt.subplots(figsize=(max(10, n_show * 0.38), 6))
    x = np.arange(n_show)
    width = 0.35

    train_vals = [p[2] * 100 for p in pairs[:n_show]]
    infer_vals = [p[3] * 100 for p in pairs[:n_show]]
    colors = ["#3fb950" if p[4] else "#f85149" for p in pairs[:n_show]]

    ax.bar(x - width/2, train_vals, width, label="Training", color="#58a6ff",
           edgecolor="#0d1117", linewidth=0.5)
    bars_infer = ax.bar(x + width/2, infer_vals, width, label="Inference",
                        color=colors, edgecolor="#0d1117", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"R{r[0]}→R{r[1]}" for r in pairs[:n_show]], fontsize=7, rotation=90)
    ax.set_xlabel("Rank Pair (src → dst)", fontsize=10)
    ax.set_ylabel("Communication Frequency (%)", fontsize=10)
    ax.set_title(f"Preset Circuit Coverage (top {n_show} rank pairs)",
                 fontsize=13, fontweight="bold", color="#58a6ff")

    # Legend
    legend_elements = [
        Patch(facecolor="#58a6ff", label="Training"),
        Patch(facecolor="#3fb950", label="Inference — Covered by Preset"),
        Patch(facecolor="#f85149", label="Inference — NOT in Preset"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.15, axis="y")

    # Coverage stats
    covered = sum(p[3] for p in pairs if p[4])
    total = sum(p[3] for p in pairs)
    coverage = (covered / total * 100) if total > 0 else 0
    n_covered = sum(1 for p in pairs if p[4])
    n_total = len(pairs)

    ax.text(0.98, 0.95, f"Preset coverage: {coverage:.1f}% of comm\n"
            f"Circuits: {n_covered} of {len(preset_pairs)} preset pairs\n"
            f"cover {n_covered}/{n_total} active rank pairs",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#8b949e",
            bbox=dict(facecolor="#161b22", edgecolor="#30363d", pad=6, alpha=0.9))

    return _fig_to_b64(fig)


def chart_affinity_summary(
    train_matrix: np.ndarray,
    infer_matrix: np.ndarray,
    preset_pairs: set | None,
) -> str:
    """Summary stat tiles as a compact figure."""
    num_ranks = train_matrix.shape[0]

    # Compute stats
    corr = affinity_correlation(train_matrix, infer_matrix)

    # Rank pairs
    pairs_train = []
    pairs_infer = []
    for i in range(num_ranks):
        for j in range(num_ranks):
            if i != j:
                pairs_train.append((i, j, train_matrix[i, j]))
                pairs_infer.append((i, j, infer_matrix[i, j]))
    pairs_train.sort(key=lambda x: -x[2])
    pairs_infer.sort(key=lambda x: -x[2])

    # Top-5 overlap
    top5_train = {(s, d) for s, d, _ in pairs_train[:5]}
    top5_infer = {(s, d) for s, d, _ in pairs_infer[:5]}
    top5_overlap = len(top5_train & top5_infer)

    # Covered fraction
    if preset_pairs:
        infer_total = infer_matrix.sum()
        covered = sum(infer_matrix[s, d] for s, d in preset_pairs
                      if 0 <= s < num_ranks and 0 <= d < num_ranks)
        preset_coverage = (covered / infer_total * 100) if infer_total > 0 else 0
    else:
        preset_coverage = None

    # Estimate hit rate
    est_hit = estimated_hit_rate(train_matrix, max_circuits=16,
                                 num_ranks=num_ranks, experts_per_rank=1)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")

    stats = [
        ("Affinity Correlation (r)", f"{corr:.4f}", "1.0 = perfect"),
        ("Top-5 Overlap", f"{top5_overlap}/5", "same rank-pairs in top-5"),
        ("Est. Hit Rate", f"{est_hit*100:.1f}%", f"with {num_ranks} ranks"),
    ]
    if preset_coverage is not None:
        stats.append(("Preset Coverage", f"{preset_coverage:.1f}%", "of inference comm"))

    stats.append(("Rank Pairs", f"{num_ranks}×{num_ranks}", f"{num_ranks*(num_ranks-1)} off-diagonal"))

    for i, (label, value, subtitle) in enumerate(stats):
        x = 0.05 + (i % 5) * 0.20
        y = 0.3
        ax.text(x, y + 0.35, label, fontsize=9, color="#8b949e", ha="center", transform=ax.transAxes)
        ax.text(x, y + 0.05, value, fontsize=18, fontweight="bold", color="#58a6ff",
                ha="center", transform=ax.transAxes)
        ax.text(x, y - 0.25, subtitle, fontsize=7, color="#8b949e", ha="center", transform=ax.transAxes)

    return _fig_to_b64(fig)


# ═══════════════════════════════════════════════════════════════════════════
# HTML report builder
# ═══════════════════════════════════════════════════════════════════════════

def build_html_report(
    train_trace: RoutingTrace,
    infer_trace: RoutingTrace | None,
    experts_per_rank: int,
    preset_pairs: set | None,
    charts: dict[str, str],
) -> str:
    """Build a self-contained HTML report with all charts embedded."""
    train_meta = train_trace.meta
    infer_meta = infer_trace.meta if infer_trace else None
    has_infer = infer_trace is not None
    has_preset = preset_pairs is not None and len(preset_pairs) > 0

    num_ranks = train_meta.num_experts // experts_per_rank

    # Build parts piece by piece to avoid f-string backslash issues
    parts = []

    # Subtitle line
    subtitle = (
        f"Training: <strong>{train_meta.model_id}</strong> &mdash; "
        f"{train_meta.num_experts} experts, top-{train_meta.top_k}, "
        f"{train_meta.num_moe_layers} MoE layers, "
        f"{train_meta.total_tokens} tokens "
        f"({train_meta.prompt_len} prompt + {train_meta.generated_len} generated)"
    )
    if infer_meta:
        subtitle += (
            f" | Inference: <strong>{infer_meta.model_id}</strong> &mdash; "
            f"{infer_meta.total_tokens} tokens"
        )
    subtitle += f" | Experts/rank: <strong>{experts_per_rank}</strong>"

    # TOC links — ordered by pipeline story
    toc_links = [
        '<a href="#pipeline">Pipeline</a>',
        '<a href="#train-heatmap">Train Affinity</a>',
    ]
    if has_preset:
        toc_links.append('<a href="#preset-coverage">Preset Coverage</a>')
    toc_links.append('<a href="#expert-load">Expert Load</a>')
    if has_infer:
        toc_links.append('<a href="#js-divergence">Per-Layer JS</a>')
        toc_links.append('<a href="#fidelity">Fidelity Δ</a>')

    # Stat row
    stat_items = [
        f'<div class="stat"><span class="stat-label">Train Experts</span><span class="stat-value">{train_meta.num_experts}</span></div>',
        f'<div class="stat"><span class="stat-label">Top-K</span><span class="stat-value">{train_meta.top_k}</span></div>',
        f'<div class="stat"><span class="stat-label">Train Tokens</span><span class="stat-value">{train_meta.total_tokens}</span></div>',
    ]
    if infer_meta:
        stat_items.append(
            f'<div class="stat"><span class="stat-label">Infer Tokens</span><span class="stat-value">{infer_meta.total_tokens}</span></div>'
        )
    stat_items.append(
        f'<div class="stat"><span class="stat-label">Num Ranks</span><span class="stat-value">{num_ranks}</span></div>'
    )
    if has_preset:
        stat_items.append(
            f'<div class="stat"><span class="stat-label">Preset Circuits</span><span class="stat-value">{len(preset_pairs)}</span></div>'
        )

    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MoE Affinity → OCS Pre-set Pipeline</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 24px; line-height: 1.5; }}
  h1 {{ color: #58a6ff; font-size: 24px; margin-bottom: 4px; }}
  h2 {{ color: #f0f6fc; font-size: 18px; margin: 32px 0 16px; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  .subtitle {{ color: #8b949e; font-size: 14px; margin-bottom: 24px; }}
  .stat-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 18px; min-width: 140px; }}
  .stat-label {{ display: block; color: #8b949e; font-size: 11px; margin-bottom: 4px; }}
  .stat-value {{ display: block; color: #58a6ff; font-size: 20px; font-weight: 700; }}
  .chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                      padding: 16px; margin-bottom: 24px; overflow-x: auto; }}
  .chart-container img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
  .insight-box {{ background: #1a1f2e; border: 1px solid #30363d; border-left: 3px solid #58a6ff;
                  border-radius: 4px; padding: 12px 16px; margin: 16px 0; font-size: 13px; color: #c9d1d9; }}
  .insight-box.warn {{ border-left-color: #d29922; }}
  .insight-box.good {{ border-left-color: #3fb950; }}
  .insight-box.bad {{ border-left-color: #f85149; }}
  .toc {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 24px; }}
  .toc a {{ color: #58a6ff; text-decoration: none; font-size: 13px; padding: 4px 12px;
            border: 1px solid #30363d; border-radius: 20px; }}
  .toc a:hover {{ background: #1f2937; }}
  .pipeline-flow {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
                    margin: 16px 0; font-size: 13px; }}
  .pipeline-step {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                    padding: 12px 16px; text-align: center; min-width: 100px; }}
  .pipeline-step .step-label {{ color: #8b949e; font-size: 10px; text-transform: uppercase; }}
  .pipeline-step .step-name {{ color: #58a6ff; font-weight: 700; font-size: 14px; }}
  .pipeline-arrow {{ color: #3fb950; font-size: 20px; font-weight: bold; }}
  hr.section-divider {{ border-color: #30363d; margin: 32px 0; opacity: 0.5; }}
</style>
</head>
<body>

<h1>MoE Affinity → OCS Pre-set Pipeline</h1>
<div class="subtitle">{{subtitle}}</div>

<div class="stat-row">{{stat_items}}</div>

<div class="toc">{{toc_links}}</div>

<!-- Pipeline Overview -->
<h2 id="pipeline">Pipeline — Training Affinity → OCS Pre-set</h2>
<div class="chart-container">
  <div class="pipeline-flow" style="justify-content: center; padding: 20px 0;">
    <div class="pipeline-step">
      <div class="step-label">Stage 1</div>
      <div class="step-name">🔍 Record Affinity</div>
      <div style="color:#8b949e;font-size:11px;margin-top:4px;">Qwen pretrained / LoRA<br>token→expert routing</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">
      <div class="step-label">Stage 2</div>
      <div class="step-name">📊 Affinity Graph</div>
      <div style="color:#8b949e;font-size:11px;margin-top:4px;">Rank-pair co-activation<br>communication matrix</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step" style="border-color:#3fb950;">
      <div class="step-label">Stage 3</div>
      <div class="step-name" style="color:#3fb950;">🔌 OCS Pre-set</div>
      <div style="color:#8b949e;font-size:11px;margin-top:4px;">Pre-configure circuits<br>from affinity ranking</div>
    </div>
    <div class="pipeline-arrow">→</div>
    <div class="pipeline-step">
      <div class="step-label">Result</div>
      <div class="step-name" style="color:#f0883e;">📈 vs EPS Baseline</div>
      <div style="color:#8b949e;font-size:11px;margin-top:4px;">Pre-set circuits<br>hide reconfig latency</div>
    </div>
  </div>
</div>
<div class="insight-box">
  <strong>How it works:</strong> We record every token's expert selection from the Qwen model
  (pretrained or LoRA-finetuned) — which experts are co-activated, and on which ranks.
  This builds an <strong>affinity graph</strong> (rank-pair communication matrix). The top-<em>k</em>
  highest-affinity rank pairs are pre-configured as OCS circuits <em>before</em> inference starts.
  At inference time, these circuits are already established — no runtime reconfiguration penalty.
  The result: OCS pre-set hides circuit setup latency under the known affinity pattern.
</div>

<!-- Training Affinity Heatmap -->
<h2 id="train-heatmap">Training Affinity — Rank-Pair Communication</h2>
<div class="chart-container">
  <img src="data:image/png;base64,{{train_heatmap_b64}}" alt="Training Affinity Heatmap">
</div>
<div class="insight-box">
  <strong>How to read:</strong> Cell (R<em>i</em>, R<em>j</em>) shows communication frequency between
  rank <em>i</em> and rank <em>j</em>, derived from expert co-activation in the Qwen model routing trace.
  This is the affinity graph that drives OCS circuit pre-configuration.
  Darker cells = more co-activation = higher priority for pre-set circuits.{{preset_note}}
</div>
""")

    # Replace template placeholders (f-string {{ }} → literal { }, so match single braces)
    html = parts[0]
    html = html.replace("{subtitle}", subtitle)
    html = html.replace("{stat_items}", "\n  ".join(stat_items))
    html = html.replace("{toc_links}", "\n  ".join(toc_links))
    html = html.replace("{train_heatmap_b64}", charts.get("train_heatmap", ""))
    preset_note = " Dashed green outlines show preset circuits." if has_preset else ""
    html = html.replace("{preset_note}", preset_note)

    # Preset coverage — right after training heatmap (this IS the OCS pre-set result)
    if has_preset:
        html += f"""<!-- Preset Coverage -->
<h2 id="preset-coverage">OCS Pre-set — Circuit Coverage</h2>
<div class="chart-container">
  <img src="data:image/png;base64,{charts.get('preset_coverage', '')}" alt="Preset Coverage">
</div>
<div class="insight-box good">
  <strong>How to read:</strong> Green bars = rank pairs covered by OCS pre-set circuits
  (derived from training affinity). Red bars = rank pairs NOT covered.
  High green coverage = pre-set circuits are well-placed for the actual inference traffic.
</div>
"""

    # Expert load
    if has_infer:
        html += f"""<!-- Expert Load -->
<h2 id="expert-load">Expert Load Distribution</h2>
<div class="chart-container">
  <img src="data:image/png;base64,{charts.get('expert_load', '')}" alt="Expert Load">
</div>
<div class="insight-box">
  <strong>How to read:</strong> Which experts are most frequently selected.
  Blue = training (affinity source), orange = inference. Similar heights =
  stable expert preference across train/infer. Divergence = preference shift.
</div>

<!-- Per-Layer JS Divergence -->
<h2 id="js-divergence">Per-Layer Routing Consistency (JS Divergence)</h2>
<div class="chart-container">
  <img src="data:image/png;base64,{charts.get('js_divergence', '')}" alt="Per-Layer JS Divergence">
</div>
<div class="insight-box">
  <strong>How to read:</strong> Each bar is one MoE layer. Low JS divergence (&lt; 0.1, green) =
  training/inference routing nearly identical — pre-set circuits should work well.
  High divergence (&gt; 0.3, red) = routing patterns differ — pre-set may need adjustment
  for that layer.
</div>

<hr class="section-divider">
<h2 id="fidelity">Train / Inference Fidelity Analysis</h2>
<div class="insight-box warm">
  <strong>Supplementary:</strong> These charts show how well the training-time affinity
  (used for OCS pre-set) matches actual inference-time routing. Large discrepancies
  indicate the pre-set may need recalibration.
</div>

<!-- Inference Affinity Heatmap -->
<h2 id="infer-heatmap">Inference Affinity — Rank-Pair Communication</h2>
<div class="chart-container">
  <img src="data:image/png;base64,{charts.get('infer_heatmap', '')}" alt="Inference Affinity Heatmap">
</div>

<!-- Delta Heatmap -->
<h2 id="delta">Affinity Delta — Train &#8722; Inference</h2>
<div class="chart-container">
  <img src="data:image/png;base64,{charts.get('delta_heatmap', '')}" alt="Delta Heatmap">
</div>
<div class="insight-box">
  <strong>How to read:</strong> Blue = training over-estimates (pre-set may allocate
  unnecessary circuits). Red = training under-estimates (pre-set may miss important
  circuits).
</div>

<!-- Correlation Scatter -->
<h2 id="correlation">Affinity Correlation — Train vs Inference</h2>
<div class="chart-container">
  <img src="data:image/png;base64,{charts.get('correlation', '')}" alt="Correlation Scatter">
</div>
"""

    html += """<hr style="border-color: #30363d; margin: 32px 0;">
<div style="color: #8b949e; font-size: 12px; text-align: center;">
  Generated by scripts/viz_affinity.py &mdash; MoE OCS Learning Testbed
</div>

</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Visualize MoE affinity and train/inference discrepancy",
    )
    parser.add_argument(
        "--train-trace", required=True,
        help="Path to training routing trace JSON",
    )
    parser.add_argument(
        "--infer-trace", default=None,
        help="Path to inference routing trace JSON (optional; if omitted, single-trace analysis)",
    )
    parser.add_argument(
        "--preset-plan", default=None,
        help="Path to preset circuit plan JSON (optional; overlays on heatmaps)",
    )
    parser.add_argument(
        "--experts-per-rank", type=int, default=64,
        help="Experts per rank for rank mapping (default: 64 → 4 ranks for 256 experts)",
    )
    parser.add_argument(
        "--output", "-o", default="outputs/affinity_report.html",
        help="Output HTML path",
    )
    parser.add_argument(
        "--top-n-experts", type=int, default=30,
        help="Top N experts to show in load distribution chart",
    )
    args = parser.parse_args()

    # ── Load traces ──
    if not os.path.exists(args.train_trace):
        print(f"Error: train trace not found: {args.train_trace}")
        sys.exit(1)

    print(f"Loading train trace: {args.train_trace}")
    train_trace = RoutingTrace.load(args.train_trace)
    print(f"  {train_trace.meta.num_experts} experts, top-{train_trace.meta.top_k}, "
          f"{train_trace.meta.num_moe_layers} layers, {train_trace.meta.total_tokens} tokens")

    infer_trace = None
    if args.infer_trace:
        if not os.path.exists(args.infer_trace):
            print(f"Error: infer trace not found: {args.infer_trace}")
            sys.exit(1)
        print(f"Loading infer trace: {args.infer_trace}")
        infer_trace = RoutingTrace.load(args.infer_trace)
        print(f"  {infer_trace.meta.num_experts} experts, top-{infer_trace.meta.top_k}, "
              f"{infer_trace.meta.num_moe_layers} layers, {infer_trace.meta.total_tokens} tokens")

    # ── Load preset plan ──
    preset_pairs = None
    if args.preset_plan:
        if os.path.exists(args.preset_plan):
            with open(args.preset_plan) as f:
                plan_data = json.load(f)
            preset_pairs = {(int(src), int(dst)) for src, dst, _ in plan_data.get("circuits", [])}
            print(f"Preset plan: {len(preset_pairs)} circuits loaded")
        else:
            print(f"Warning: preset plan not found: {args.preset_plan}")

    # ── Build data matrices ──
    experts_per_rank = args.experts_per_rank
    num_experts = train_trace.meta.num_experts
    num_ranks = (num_experts + experts_per_rank - 1) // experts_per_rank
    print(f"Rank mapping: {num_experts} experts / {experts_per_rank} per rank = {num_ranks} ranks")

    print("Building train rank communication matrix...")
    train_matrix = build_rank_comm_matrix(train_trace, experts_per_rank)

    infer_matrix = None
    if infer_trace:
        print("Building inference rank communication matrix...")
        infer_matrix = build_rank_comm_matrix(infer_trace, experts_per_rank)

    # ── Generate charts ──
    charts = {}

    print("Generating summary...")
    display_matrix = infer_matrix if infer_matrix is not None else train_matrix
    charts["summary"] = chart_affinity_summary(train_matrix, display_matrix, preset_pairs)

    print("Generating train heatmap...")
    charts["train_heatmap"] = chart_rank_heatmap(
        train_matrix, "Training Affinity — Rank-Pair Communication",
        experts_per_rank, preset_pairs=preset_pairs,
    )

    if infer_matrix is not None:
        print("Generating inference heatmap...")
        charts["infer_heatmap"] = chart_rank_heatmap(
            infer_matrix, "Inference Affinity — Rank-Pair Communication",
            experts_per_rank, preset_pairs=preset_pairs,
        )

        print("Generating delta heatmap...")
        charts["delta_heatmap"] = chart_delta_heatmap(train_matrix, infer_matrix, experts_per_rank)

        print("Generating correlation scatter...")
        charts["correlation"] = chart_correlation_scatter(train_matrix, infer_matrix)

        print("Generating per-layer JS divergence...")
        charts["js_divergence"] = chart_per_layer_js_divergence(train_trace, infer_trace)

        print("Generating expert load comparison...")
        train_load = build_expert_load(train_trace)
        infer_load = build_expert_load(infer_trace)
        charts["expert_load"] = chart_expert_load_comparison(
            train_load, infer_load, top_n=args.top_n_experts,
        )

    if preset_pairs:
        print("Generating preset coverage...")
        charts["preset_coverage"] = chart_preset_coverage(
            train_matrix, infer_matrix, preset_pairs, experts_per_rank,
        )

    # ── Build HTML ──
    print("Building HTML report...")
    html = build_html_report(
        train_trace=train_trace,
        infer_trace=infer_trace,
        experts_per_rank=experts_per_rank,
        preset_pairs=preset_pairs,
        charts=charts,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"\n✅ Report saved → {args.output}")
    print(f"   Open with: open {args.output}")


if __name__ == "__main__":
    main()
