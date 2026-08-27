#!/usr/bin/env python3
"""make_figures.py — figures for the routing -> affinity -> placement -> OCS chain.

Every figure is generated from the artifacts on disk (``manifest.json`` +
traces + ``evidence_chain.json``), never from a live model, so figures are
reproducible without re-running inference.

    python3 scripts/make_figures.py --workload logs/workload/qwen36 \
        --out logs/figures/qwen36 --world-size 32
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eval.affinity_graph import (  # noqa: E402
    layer_affinities, pooled_affinity, signature_similarity_matrix,
)
from src.eval.cost_model import (  # noqa: E402
    CostConfig, DispatchMode, evaluate, hierarchy_for, traffic_matrix,
)
from src.eval.placement_opt import make_placement  # noqa: E402
from src.eval.trace_ir import load_workload  # noqa: E402

plt.rcParams.update({"figure.dpi": 130, "font.size": 8,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {p}")


# ── F1/F2: invariance under topology and placement ──────────────────────
def fig_invariance(rep: dict, out: Path):
    q1 = rep.get("Q1_routing_invariance")
    q3 = rep.get("Q3_placement_cost")
    if not (q1 and q3):
        return
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))

    s = q1["structural"]["samples"]
    labels = [f"{d['placement'][:16]}\n{d['topology']}" for d in s]
    match = [d["routing_match_rate"] for d in s]
    cost = [d["bottleneck_us"] for d in s]
    x = np.arange(len(s))
    ax[0].bar(x, match, color="#2b6cb0", label="routing match rate")
    ax[0].set_ylim(0, 1.15)
    ax[0].axhline(1.0, ls="--", c="k", lw=.8)
    a2 = ax[0].twinx()
    a2.plot(x, cost, "o-", c="#c05621", ms=3.5, label="bottleneck (us)")
    a2.set_ylabel("all-to-all bottleneck (us)", color="#c05621")
    a2.grid(False)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax[0].set_ylabel("token->expert match rate")
    ax[0].set_title("F1  routing invariant, cost moves\n"
                    "(placement x topology configurations)")

    ded = next(r for r in q3["by_dispatch_mode"] if r["dispatch_mode"] == "DEDUP_RANK")
    v = ded["variants"]
    names = [d["variant"] for d in v]
    bt = np.array([d["bottleneck_us"] for d in v])
    mr = np.array([d["routing_match_rate"] for d in v])
    xx = np.arange(len(v))
    ax[1].bar(xx, bt / bt[0], color=np.where(mr >= 1.0, "#2f855a", "#c53030"))
    ax[1].axhline(1.0, ls="--", c="k", lw=.8)
    ax[1].set_xticks(xx)
    ax[1].set_xticklabels(names, rotation=90, fontsize=5.5)
    ax[1].set_ylabel("bottleneck / linear")
    ax[1].set_title("F2  placement perturbations\n"
                    "green = routing bit-identical (all of them)")
    _save(fig, out, "F1_F2_invariance.png")


# ── F3: expert specialization across workload categories ────────────────
def fig_specialization(t, rep: dict, out: Path):
    cat_t = t.by_role("category")
    cats = sorted({r.category for r in cat_t.runs})
    if not cats:
        return
    L = cat_t.n_layers
    best = None
    q2 = rep.get("Q2_routing_structure", {})
    pl = q2.get("category_decoding", {}).get("per_layer_accuracy", {})
    if pl:
        best = int(max(pl, key=lambda k: pl[k]))
    layer = best if best is not None else int(cat_t.layers[L // 2])

    sub = cat_t.by_layer(layer)
    M = np.zeros((len(cats), t.num_experts))
    for i, c in enumerate(cats):
        s = sub.by_category(c)
        if s.n_cells:
            M[i] = np.bincount(s.experts.ravel(),
                               minlength=t.num_experts) / s.n_cells
    order = np.argsort(-M.sum(0))
    Mn = M[:, order]
    Mn = Mn / np.maximum(Mn.max(axis=0, keepdims=True), 1e-12)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6),
                           gridspec_kw={"width_ratios": [2.2, 1]})
    im = ax[0].imshow(Mn[:, :min(96, Mn.shape[1])], aspect="auto",
                      cmap="magma", interpolation="nearest")
    ax[0].set_yticks(range(len(cats)))
    ax[0].set_yticklabels(cats, fontsize=6)
    ax[0].set_xlabel("expert (sorted by total load)")
    ax[0].set_title(f"F3a  expert usage by workload category, layer {layer}\n"
                    "(column-normalised; a vertical stripe = a specialist)")
    ax[0].grid(False)
    fig.colorbar(im, ax=ax[0], fraction=.025)

    if pl:
        ks = sorted(pl, key=int)
        ax[1].plot([int(k) for k in ks], [pl[k] for k in ks], "o-", ms=3,
                   c="#2b6cb0")
        ch = q2["category_decoding"].get("perm_null_mean", 0)
        ax[1].axhline(ch, ls="--", c="r", lw=.9,
                      label=f"permutation null {ch:.3f}")
        ax[1].axhline(q2["category_decoding"]["accuracy"], ls=":", c="g", lw=1.2,
                      label=f"all layers {q2['category_decoding']['accuracy']:.3f}")
        ax[1].set_xlabel("MoE layer")
        ax[1].set_ylabel("LOO category accuracy")
        ax[1].set_title("F3b  workload decodable from routing\n"
                        f"({q2['category_decoding']['n_categories']} classes, "
                        "leave-one-sequence-out)")
        ax[1].legend(fontsize=6)
    _save(fig, out, "F3_specialization.png")


# ── F4: affinity graph ──────────────────────────────────────────────────
def fig_affinity(t, rep: dict, out: Path):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    layer = int(t.layers[t.n_layers // 2])
    A = layer_affinities(t.by_layer(layer), "cooccurrence")[layer]
    P = pooled_affinity(t, "cooccurrence")

    def show(a, m, title):
        d = m.copy()
        np.fill_diagonal(d, 0)
        o = np.argsort(-d.sum(1))
        d = d[np.ix_(o, o)]
        n = min(128, d.shape[0])
        im = a.imshow(np.log1p(d[:n, :n]), cmap="viridis")
        a.set_title(title, fontsize=7.5)
        a.grid(False)
        fig.colorbar(im, ax=a, fraction=.045)

    show(ax[0], A, f"F4a  per-layer co-activation (layer {layer})\n"
                   "log1p, affinity-sorted")
    show(ax[1], P, "F4b  layer-POOLED co-activation\n"
                   "(structure washed out: layers are independent namespaces)")

    q2 = rep.get("Q2_routing_structure", {})
    st = q2.get("affinity_structure_vs_load_null", {})
    if st:
        ks = list(st.keys())
        obs = [st[k]["observed_concentration"] for k in ks]
        nul = [st[k]["null_mean"] for k in ks]
        sd = [st[k]["null_sd"] for k in ks]
        x = np.arange(len(ks))
        ax[2].bar(x - .18, obs, .34, label="observed", color="#2b6cb0")
        ax[2].bar(x + .18, nul, .34, yerr=sd, label="load-preserving null",
                  color="#a0aec0", capsize=2)
        for i, k in enumerate(ks):
            z = st[k].get("z")
            if z is not None:
                ax[2].text(i, max(obs[i], nul[i]) * 1.03, f"z={z:.0f}",
                           ha="center", fontsize=6)
        ax[2].set_xticks(x)
        ax[2].set_xticklabels(ks, rotation=20, fontsize=6.5)
        ax[2].set_ylabel("top-1% edge concentration")
        ax[2].set_title("F4c  affinity beyond marginal load\n"
                        "(null resamples top-k from per-layer load)")
        ax[2].legend(fontsize=6)
    _save(fig, out, "F4_affinity_graph.png")


# ── F5/F6: cost under placements and topologies ─────────────────────────
def fig_cost(rep: dict, out: Path):
    q4 = rep.get("Q4_affinity_value", {}).get("splits", {})
    lco = q4.get("leave_categories_out", {}).get("results")
    wc = q4.get("within_category", {}).get("results")
    if not lco:
        return
    keys = [k for k in lco if "error" not in lco[k]]
    keys.sort(key=lambda k: lco[k]["out_of_sample"]["bottleneck_us"])

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    rnd = lco.get("random", {}).get("out_of_sample", {})

    def pct(d, m):
        b = rnd.get(m)
        return 100 * (1 - d["out_of_sample"][m] / b) if b else 0.0

    x = np.arange(len(keys))
    bt = [pct(lco[k], "bottleneck_us") for k in keys]
    nb = [pct(lco[k], "network_bytes") for k in keys]
    cols = ["#2f855a" if v > 0 else "#c53030" for v in bt]
    ax[0].barh(x, bt, color=cols)
    ax[0].set_yticks(x)
    ax[0].set_yticklabels(keys, fontsize=6.5)
    ax[0].axvline(0, c="k", lw=.8)
    ax[0].set_xlabel("bottleneck reduction vs random (%)")
    ax[0].set_title("F5  placement value, OUT OF SAMPLE\n"
                    "(fit on some categories, scored on held-out ones)")

    ax[1].scatter(nb, bt, s=22, c="#2b6cb0")
    for k, a, b in zip(keys, nb, bt):
        ax[1].annotate(k, (a, b), fontsize=5.5,
                       textcoords="offset points", xytext=(3, 2))
    ax[1].axhline(0, c="k", lw=.7)
    ax[1].axvline(0, c="k", lw=.7)
    ax[1].set_xlabel("volume reduction (%)")
    ax[1].set_ylabel("bottleneck reduction (%)")
    ax[1].set_title("F6  volume is NOT the objective\n"
                    "(points below y=0 cut bytes but raise the critical path)")

    if wc:
        ks = [k for k in keys if k in wc]
        i_s = [wc[k]["in_sample"]["mean_fanout"] for k in ks]
        o_s = [lco[k]["out_of_sample"]["mean_fanout"] for k in ks]
        xx = np.arange(len(ks))
        ax[2].bar(xx - .18, i_s, .34, label="in-sample", color="#4a5568")
        ax[2].bar(xx + .18, o_s, .34, label="held-out categories",
                  color="#3182ce")
        ax[2].set_xticks(xx)
        ax[2].set_xticklabels(ks, rotation=90, fontsize=5.5)
        ax[2].set_ylabel("mean fan-out (ranks / token-layer)")
        ax[2].set_title("F6b  generalisation gap of the fitted placement")
        ax[2].legend(fontsize=6)
    _save(fig, out, "F5_F6_placement_cost.png")


def fig_topology(t, rep: dict, out: Path, world: int):
    styles = ["single_node", "single_pod", "multi_pod", "realistic"]
    cost = CostConfig(hidden_size=rep.get("hidden_size", 2048))
    specs_fit = t
    pl = {}
    for k in ("random", "load_balanced", "affinity_coordinated_layer"):
        try:
            pl[k] = make_placement(k, specs_fit, world, seed=0)
        except Exception:
            pass
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    width = .8 / max(len(pl), 1)
    x = np.arange(len(styles))
    for i, (name, p) in enumerate(pl.items()):
        bt, xp = [], []
        for s in styles:
            topo = hierarchy_for(world, s)
            r = evaluate(t, p, topo, cost, DispatchMode.DEDUP_RANK)
            bt.append(r["bottleneck_us"])
            xp.append(r["cross_pod_bytes"] / 1e9)
        ax[0].bar(x + i * width - .4 + width / 2, bt, width, label=name)
        ax[1].bar(x + i * width - .4 + width / 2, xp, width, label=name)
    for a, ttl, yl in ((ax[0], "F7a  bottleneck vs topology class",
                        "all-to-all bottleneck (us)"),
                       (ax[1], "F7b  cross-pod volume (what OCS can promote)",
                        "cross-pod bytes (GB)")):
        a.set_xticks(x)
        a.set_xticklabels(styles, rotation=20, fontsize=6.5)
        a.set_title(ttl)
        a.set_ylabel(yl)
        a.legend(fontsize=6)
    _save(fig, out, "F7_topology.png")


# ── F8: OCS + temporal stability ────────────────────────────────────────
def fig_ocs(rep: dict, out: Path):
    q5 = rep.get("Q5_ocs")
    if not q5 or "by_topology" not in q5:
        return
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))

    rows = []
    for style, entry in q5["by_topology"].items():
        for cls, r in entry.items():
            if cls == "topology" or not isinstance(r, dict):
                continue
            if not r.get("applicable"):
                continue
            rows.append((style, cls,
                         r["static_ocs"]["bottleneck_reduction_pct"],
                         r["oracle_ocs"]["bottleneck_reduction_pct"],
                         r["reconfiguration"].get("breakeven_token_passes")))
    if rows:
        labs = [f"{s}\n{c}" for s, c, *_ in rows]
        x = np.arange(len(rows))
        ax[0].bar(x - .18, [r[2] for r in rows], .34, label="static (fit-only)",
                  color="#2b6cb0")
        ax[0].bar(x + .18, [r[3] for r in rows], .34, label="oracle (sees eval)",
                  color="#90cdf4")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labs, rotation=90, fontsize=5.5)
        ax[0].set_ylabel("bottleneck reduction (%)")
        ax[0].set_title("F8a  OCS gain by switch class\n"
                        "(tier promotion only; 0 = nothing to promote)")
        ax[0].legend(fontsize=6)

        be = [(l, r[4]) for l, r in zip(labs, rows) if r[4]]
        if be:
            ax[1].barh(np.arange(len(be)), [b for _, b in be], color="#c05621")
            ax[1].set_yticks(np.arange(len(be)))
            ax[1].set_yticklabels([l.replace("\n", " ") for l, _ in be],
                                  fontsize=5.5)
            ax[1].set_xscale("log")
            ax[1].axvline(1, ls="--", c="k", lw=.8)
            ax[1].set_xlabel("token passes to amortise one reconfiguration")
            ax[1].set_title("F8b  reconfiguration break-even\n"
                            "(right of the dashed line = never pays off "
                            "within one token)")

    st = q5.get("stability", {})
    if st:
        ks = [k for k in ("run", "token", "layer") if k in st
              and not st[k].get("insufficient")]
        x = np.arange(len(ks))
        ax[2].bar(x - .18, [st[k]["traffic_cosine_mean"] for k in ks], .34,
                  label="traffic-matrix cosine", color="#2f855a")
        ax[2].bar(x + .18, [st[k]["plan_persistence_mean"] for k in ks], .34,
                  label="circuit-plan Jaccard", color="#c53030")
        ax[2].set_xticks(x)
        ax[2].set_xticklabels([f"per-{k}" for k in ks])
        ax[2].set_ylim(0, 1.05)
        ax[2].set_ylabel("similarity across windows")
        ax[2].set_title("F8c  temporal stability by timescale\n"
                        "(traffic is stable; the PLAN is not)")
        ax[2].legend(fontsize=6)
    _save(fig, out, "F8_ocs_stability.png")


# ── F9: semantics vs surface form ───────────────────────────────────────
def fig_controls(t, rep: dict, out: Path):
    q2 = rep.get("Q2_routing_structure", {})
    sv = q2.get("semantics_vs_lexis")
    if not sv:
        return
    order = ["noise_floor_repeat", "within_paraphrase_set",
             "within_length_ladder", "within_category",
             "within_lexical_control", "between_category"]
    lab = {"noise_floor_repeat": "identical prompt\n(noise floor)",
           "within_paraphrase_set": "paraphrase\nsame meaning,\ndiff words",
           "within_length_ladder": "length ladder\nsame topic",
           "within_category": "same category",
           "within_lexical_control": "lexical control\nsame words,\ndiff meaning",
           "between_category": "different category"}
    ks = [k for k in order if sv.get(k)]
    m = [sv[k]["mean"] for k in ks]
    lo = [sv[k]["mean"] - sv[k]["ci95"][0] for k in ks]
    hi = [sv[k]["ci95"][1] - sv[k]["mean"] for k in ks]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    cols = ["#4a5568", "#2f855a", "#38a169", "#3182ce", "#c05621", "#a0aec0"]
    ax[0].bar(range(len(ks)), m, yerr=[lo, hi], capsize=3,
              color=cols[:len(ks)])
    ax[0].set_xticks(range(len(ks)))
    ax[0].set_xticklabels([lab[k] for k in ks], fontsize=5.8)
    ax[0].set_ylim(min(m) - .04, 1.02)
    ax[0].set_ylabel("routing-signature cosine")
    ax[0].set_title("F9a  semantics vs surface form\n"
                    f"verdict: {sv.get('verdict')}")

    M, infos = signature_similarity_matrix(t.by_role("category"))
    if M.shape[0] > 2:
        cats = [i.category for i in infos]
        o = np.argsort(cats)
        im = ax[1].imshow(M[np.ix_(o, o)], cmap="viridis")
        bnd, prev = [], None
        for i, k in enumerate(np.array(cats)[o]):
            if k != prev:
                bnd.append(i)
                prev = k
        for b in bnd[1:]:
            ax[1].axhline(b - .5, c="w", lw=.5)
            ax[1].axvline(b - .5, c="w", lw=.5)
        ax[1].set_xticks([]); ax[1].set_yticks(bnd)
        ax[1].set_yticklabels(sorted(set(cats)), fontsize=5.5)
        ax[1].grid(False)
        ax[1].set_title("F9b  routing-signature similarity\n"
                        "(sequences grouped by category)")
        fig.colorbar(im, ax=ax[1], fraction=.045)
    _save(fig, out, "F9_semantic_controls.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="logs/workload/qwen36")
    ap.add_argument("--out", default=None)
    ap.add_argument("--world-size", type=int, default=None)
    args = ap.parse_args()

    wl = Path(args.workload)
    rep_p = wl / "evidence_chain.json"
    if not rep_p.exists():
        print(f"[fig] missing {rep_p}; run verify_live_invariance.py first")
        return 2
    rep = json.load(open(rep_p))
    t = load_workload(wl / "manifest.json")
    world = args.world_size or rep.get("world_size") or 8
    out = Path(args.out or (wl / "figures"))
    print(f"[fig] {t}  world={world}  -> {out}")

    for fn, a in ((fig_invariance, (rep, out)),
                  (fig_specialization, (t, rep, out)),
                  (fig_affinity, (t, rep, out)),
                  (fig_cost, (rep, out)),
                  (fig_topology, (t, rep, out, world)),
                  (fig_ocs, (rep, out)),
                  (fig_controls, (t, rep, out))):
        try:
            fn(*a)
        except Exception as e:
            print(f"  [skip] {fn.__name__}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
