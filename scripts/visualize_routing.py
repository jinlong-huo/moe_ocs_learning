#!/usr/bin/env python3
"""
可视化：LoRA 微调 Router Gate 前后路由对比
==============================================
生成 4 张图，说明路由模式的可预测性与 OCS 预触发可行性。

用法:  .venv/bin/python visualize_routing.py
输出:  logs/routing_comparison.png
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

TRACE_DIR = Path(__file__).resolve().parent.parent / "data" / "routing_traces"

# Use system CJK font on macOS
_cjk_font = None
for f in fm.fontManager.ttflist:
    if "Heiti SC" in f.name or "STHeiti" in f.name:
        _cjk_font = f.name
        break
if _cjk_font:
    plt.rcParams["font.family"] = _cjk_font
else:
    # fallback: avoid CJK titles
    pass

LOG_DIR = Path("logs")
TRACE_DIR = Path(__file__).resolve().parent.parent / "data" / "routing_traces"
PRE_PATH = TRACE_DIR / "routing_pretrained.json"
FT_PATH = TRACE_DIR / "routing_finetuned.json"
OUT_PATH = LOG_DIR / "routing_comparison.png"

N_EXPERTS = 256
N_LAYERS = 40
TOP_K = 8
PROMPT_LEN = 36  # first 36 tokens are prompt (identical input)


def load(path):
    with open(path) as f:
        return json.load(f)


def compute_stats(trace):
    """Return per-layer active experts and per-layer expert→count mapping."""
    active = {}       # layer → number of active experts
    usage = {}        # layer → Counter(expert → count)
    for route in trace["routes"]:
        for lid_str, lr in route["layers"].items():
            lid = int(lid_str)
            usage.setdefault(lid, Counter())
            for e in lr["experts"]:
                usage[lid][e] += 1
    for lid in range(N_LAYERS):
        active[lid] = sum(1 for v in usage.get(lid, Counter()).values() if v > 0)
    return active, usage


def load_all():
    pre = load(PRE_PATH)
    ft = load(FT_PATH)
    pre_active, pre_usage = compute_stats(pre)
    ft_active, ft_usage = compute_stats(ft)
    return pre, ft, pre_active, pre_usage, ft_active, ft_usage


def build_figure():
    pre, ft, pre_active, pre_usage, ft_active, ft_usage = load_all()

    # ── helpers ──────────────────────────────────────────────────
    layers = list(range(N_LAYERS))

    # 1) per-layer routing agreement (prompt tokens only)
    agreement = {}
    for lid in range(N_LAYERS):
        match = 0
        total = 0
        # only compare prompt tokens (both traces share first 36)
        n_cmp = min(len(pre["routes"]), len(ft["routes"]), PROMPT_LEN)
        for i in range(n_cmp):
            pre_exps = set(pre["routes"][i]["layers"][str(lid)]["experts"])
            ft_exps = set(ft["routes"][i]["layers"][str(lid)]["experts"])
            match += len(pre_exps & ft_exps)
            total += TOP_K
        agreement[lid] = match / total * 100 if total > 0 else 0

    # 2) expert assignment delta (prompt only)
    gain_loss = Counter()
    n_cmp = min(len(pre["routes"]), len(ft["routes"]), PROMPT_LEN)
    for i in range(n_cmp):
        for lid in range(N_LAYERS):
            pre_list = pre["routes"][i]["layers"][str(lid)]["experts"]
            ft_list = ft["routes"][i]["layers"][str(lid)]["experts"]
            for e in pre_list:
                gain_loss[e] -= 1
            for e in ft_list:
                gain_loss[e] += 1
    # net change
    net = sorted(gain_loss.items(), key=lambda x: x[1])  # most lost → most gained

    # 3) coactivation matrix for prompt — pretrained only, top experts
    coact = [[0] * N_EXPERTS for _ in range(N_EXPERTS)]
    for i in range(min(len(pre["routes"]), PROMPT_LEN)):
        for lid in range(N_LAYERS):
            exps = pre["routes"][i]["layers"][str(lid)]["experts"]
            for a, e1 in enumerate(exps):
                for e2 in exps[a + 1 :]:
                    coact[e1][e2] += 1
                    coact[e2][e1] += 1

    # 4) expert load distribution for a representative deep layer
    deep_lid = 38
    pre_load = [pre_usage[deep_lid].get(e, 0) for e in range(N_EXPERTS)]
    ft_load = [ft_usage[deep_lid].get(e, 0) for e in range(N_EXPERTS)]

    # ── plotting ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "LoRA Router Gate Fine-tuning: Routing Pattern Comparison\n"
        "Qwen3.6-35B-A3B | 256 experts × 40 layers | top-8 | same 36-token input",
        fontsize=13,
        fontweight="bold",
    )

    # ── [A] Per-layer routing agreement ──────────────────────────
    ax = axes[0][0]
    colors = ["#2ecc71" if v >= 80 else "#f39c12" if v >= 60 else "#e74c3c" for v in agreement.values()]
    ax.bar(layers, [agreement[l] for l in layers], color=colors, edgecolor="white", lw=0.3)
    ax.axhline(y=71.1, color="#3498db", linestyle="--", linewidth=1.5, label="mean = 71.1%")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Routing agreement (%)")
    ax.set_title("A) Per-layer Routing Agreement (同一输入, Top-8 专家交集)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # annotate early / deep
    ax.annotate(
        f"Layer 0: {agreement[0]:.1f}%",
        xy=(0, agreement[0]),
        xytext=(4, agreement[0] + 8),
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        fontsize=8,
        color="#2ecc71",
    )
    ax.annotate(
        f"Layer 39: {agreement[39]:.1f}%",
        xy=(39, agreement[39]),
        xytext=(28, agreement[39] + 12),
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        fontsize=8,
        color="#e74c3c",
    )

    # ── [B] Active experts per layer ─────────────────────────────
    ax = axes[0][1]
    x = np.arange(N_LAYERS)
    w = 0.35
    ax.bar(x - w / 2, [pre_active[l] for l in layers], w, label="Pre-trained", color="#3498db", alpha=0.85)
    ax.bar(x + w / 2, [ft_active[l] for l in layers], w, label="LoRA Fine-tuned", color="#e74c3c", alpha=0.85)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Active experts (out of 256)")
    ax.set_title("B) Active Expert Count per Layer — Universal Entropy Drop")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # annotate totals
    pre_total = sum(pre_active.values())
    ft_total = sum(ft_active.values())
    reduction = (1 - ft_total / pre_total) * 100
    ax.text(
        0.98, 0.95,
        f"Σ Pre:  {pre_total}\nΣ FT:   {ft_total}\nΔ:     −{reduction:.1f}%",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # ── [C] Expert gain/loss ─────────────────────────────────────
    ax = axes[1][0]
    top_n = 25
    losers = net[:top_n]
    winners = net[-top_n:]
    combined = losers + winners
    labels = [f"{eid}" for eid, _ in combined]
    values = [cnt for _, cnt in combined]
    bar_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in values]
    ax.barh(labels, values, color=bar_colors, edgecolor="white", lw=0.3)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Net assignment change")
    ax.set_ylabel("Expert ID")
    ax.set_title(f"C) Expert Assignment Δ (Top {top_n} Lost / Gained, prompt only)")
    ax.grid(axis="x", alpha=0.3)

    # highlight expert 41
    for i, (eid, cnt) in enumerate(combined):
        if eid == "41":
            ax.annotate(
                f"+{cnt} — dominant",
                xy=(cnt, i),
                xytext=(cnt + 6, i),
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=0.8),
                fontsize=8,
                color="#2ecc71",
                va="center",
            )

    # ── [D] Layer 38 expert load scatter ─────────────────────────
    ax = axes[1][1]
    mask_pre = np.array(pre_load) > 0
    mask_ft = np.array(ft_load) > 0
    mask_both = mask_pre | mask_ft
    ax.scatter(
        [i for i, m in enumerate(mask_both) if m and not mask_pre[i]],
        [ft_load[i] for i, m in enumerate(mask_both) if m and not mask_pre[i]],
        s=12, color="#e74c3c", alpha=0.5, label="FT-only",
    )
    ax.scatter(
        [i for i, m in enumerate(mask_both) if m and not mask_ft[i]],
        [pre_load[i] for i, m in enumerate(mask_both) if m and not mask_ft[i]],
        s=12, color="#3498db", alpha=0.5, label="Pre-only",
    )
    ax.scatter(
        [i for i, m in enumerate(mask_both) if m and mask_pre[i] and mask_ft[i]],
        [pre_load[i] for i, m in enumerate(mask_both) if m and mask_pre[i] and mask_ft[i]],
        s=14, color="#9b59b6", alpha=0.6, label="Both",
    )
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Activation count")
    ax.set_title(f"D) Layer {deep_lid} Expert Load: Pre ({pre_active[deep_lid]} active) → FT ({ft_active[deep_lid]} active)")
    ax.legend(fontsize=8, markerscale=1.2)
    ax.grid(alpha=0.3)

    # ── footer ───────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        "LoRA on router gate → routing patterns predictable & converge → coactivation graph = OCS pre-config table",
        ha="center",
        fontsize=10,
        fontstyle="italic",
        color="gray",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved to {OUT_PATH}")


def print_summary():
    """Print the key numbers to stdout."""
    pre, ft, pre_active, pre_usage, ft_active, ft_usage = load_all()

    # Agreement
    n_cmp = min(len(pre["routes"]), len(ft["routes"]), PROMPT_LEN)
    total_match, total_slots = 0, 0
    for i in range(n_cmp):
        for lid in range(N_LAYERS):
            pre_set = set(pre["routes"][i]["layers"][str(lid)]["experts"])
            ft_set = set(ft["routes"][i]["layers"][str(lid)]["experts"])
            total_match += len(pre_set & ft_set)
            total_slots += TOP_K
    print(f"Routing agreement: {total_match}/{total_slots} = {total_match/total_slots*100:.1f}%")
    print(f"Prompt tokens compared: {n_cmp}")
    print(f"Pre-trained routes: {len(pre['routes'])}")
    print(f"Fine-tuned routes:  {len(ft['routes'])}")

    # Active experts
    pre_total = sum(pre_active.values())
    ft_total = sum(ft_active.values())
    print(f"\nActive expert slots: {pre_total} → {ft_total} ({(1-ft_total/pre_total)*100:.1f}% reduction)")

    # Coactivation pairs
    for name, trace in [("pre", pre), ("ft", ft)]:
        pair_set = set()
        for i in range(min(len(trace["routes"]), PROMPT_LEN)):
            for lid in range(N_LAYERS):
                exps = trace["routes"][i]["layers"][str(lid)]["experts"]
                for a, e1 in enumerate(exps):
                    for e2 in exps[a + 1 :]:
                        pair_set.add((min(e1, e2), max(e1, e2)))
        print(f"Unique coact pairs ({name}): {len(pair_set)}")


if __name__ == "__main__":
    print_summary()
    print()
    try:
        build_figure()
    except Exception as e:
        print(f"Plot failed: {e}")
        print("(text summary above is still valid)")
