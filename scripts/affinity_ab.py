#!/usr/bin/env python3
"""affinity_ab.py — Phase 5: the intra-layer and inter-layer affinities, in ONE objective.

THE QUESTION
────────────
The two affinities are projections of the same object (a token's expert path) and
they are measured to CONFLICT: rho(inter-layer objective, this repo's bottleneck)
= -0.49 / -0.32 / -0.32 across three models.  So "take either" is wrong, and the
claim to test is the plan's: **carry both terms and score on one microsecond
metric**.

THE ARMS
────────
    linear            the deployed default (reference)
    A  intra only     affinity_coordinated_layer  (the repo's best-EPS generator)
    B  inter only     ExFlow's objective, swap-based local search on the same
                      objective their ILP minimises (a stand-in, not their solver)
    C(lam)  both      a NEW combined local search:
                          cost = eps_proxy(P) + lam * reroute_fraction(P)
                      started from A, so the intra term is already good and the
                      search has to TRADE the two.

DISCIPLINE
──────────
Everything is fitted on the fit window and scored on a DISJOINT eval window
(leave-categories-out, the repo's own split).  The search optimises a cheap proxy;
the verdict is the real microsecond metric.  The inter-layer transitions used by B
and C come from the fit window only.

Usage
─────
    python3.12 scripts/affinity_ab.py --workload logs/workload/qwen36 --world-size 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "scripts"))

import src.eval.cost_model as cm  # noqa: E402
from src.eval.cost_model import (  # noqa: E402
    CostConfig, DispatchMode, FabricConfig, Placement, Tier, Topology, evaluate,
    hierarchy_for, traffic_matrix,
)
from src.eval.ocs_eval import OcsConfig, plan_circuits, with_circuits  # noqa: E402
from src.eval.placement_opt import dedup_ingress, make_placement  # noqa: E402
from src.eval.trace_ir import load_workload  # noqa: E402
from src.serving.suite import build_suite, split_by_category  # noqa: E402
from exflow_compare import as_per_layer, build_transitions, exflow_local_search  # noqa: E402
from exflow_compare import exflow_reroute_fraction  # noqa: E402

LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def layer_experts(t, layers) -> dict:
    """{layer_id: [n_cells_l, K] expert ids} for one window."""
    out = {}
    for l in layers:
        sub = t.by_layer(int(l))
        if sub.n_cells:
            out[int(l)] = sub.experts
    return out


def eps_proxy(P: np.ndarray, lexp: dict, layers, W: int) -> float:
    """Average per-cell max-ingress — the collective's critical path in count space."""
    tot, n = 0.0, 0
    for i, l in enumerate(layers):
        ex = lexp.get(int(l))
        if ex is None:
            continue
        tot += float(dedup_ingress(ex, P[i], W).max())
        n += ex.shape[0]
    return tot / max(n, 1)


def combined_local_search(lexp: dict, layers, trans, init: np.ndarray, E: int, W: int,
                          lam: float, rng, n_sweeps: int = 24, n_cand: int = 8):
    """Swap search on  eps_proxy + lam * reroute_fraction,  capacity-preserving.

    Swaps stay INSIDE a layer, so every rank keeps exactly E/W experts — the
    constraint both this repo's generators and ExFlow's formulation require.
    """
    L = init.shape[0]
    P = init.copy()
    touch = [[] for _ in range(L)]
    for l in range(L):
        if l <= L - 2:
            touch[l].append(l)
        if l >= 1:
            touch[l].append(l - 1)

    def boundary_cost(i):
        a, b = trans[i][:, 0], trans[i][:, 1]
        return int((P[i, a] != P[i + 1, b]).sum())

    cur_b = {i: boundary_cost(i) for i in range(L - 1)}
    n_pairs = sum(int(p.shape[0]) for p in trans)
    cur_e = {i: (float(dedup_ingress(lexp[int(layers[i])], P[i], W).max())
                 / max(lexp[int(layers[i])].shape[0], 1))
             if int(layers[i]) in lexp else 0.0 for i in range(L)}
    n_cells = sum(int(lexp[int(l)].shape[0]) for l in layers if int(l) in lexp) or 1

    def total():
        return sum(cur_e.values()) / n_cells + lam * sum(cur_b.values()) / max(n_pairs, 1)

    best = total()
    for _ in range(n_sweeps):
        improved = False
        for l in rng.permutation(L):
            if int(layers[l]) not in lexp:
                continue
            for _ in range(max(1, E // 8)):
                u, v = rng.choice(E, size=2, replace=False)
                if P[l, u] == P[l, v]:
                    continue
                e_before = cur_e[l]
                b_before = sum(cur_b[i] for i in touch[l])
                P[l, u], P[l, v] = P[l, v], P[l, u]
                ex = lexp[int(layers[l])]
                e_after = (float(dedup_ingress(ex, P[l], W).max()) / max(ex.shape[0], 1))
                b_after = sum(
                    int((P[i, trans[i][:, 0]] != P[i + 1, trans[i][:, 1]]).sum())
                    for i in touch[l])
                if (e_after + lam * b_after / max(n_pairs, 1)) < \
                   (e_before + lam * b_before / max(n_pairs, 1)):
                    cur_e[l] = e_after
                    for i in touch[l]:
                        cur_b[i] = int((P[i, trans[i][:, 0]] != P[i + 1, trans[i][:, 1]]).sum())
                    improved = True
                else:
                    P[l, u], P[l, v] = P[l, v], P[l, u]
        if not improved:
            break
    return P, total()


def score(pl: Placement, fit, ev, topo, cfg, cost, W: int, seed: int) -> dict:
    """The out-of-sample verdict: the REAL microsecond metric on the eval window."""
    eps = evaluate(ev, pl, topo, cost, DispatchMode.DEDUP_RANK, n_dp=W, seed=seed)
    tm = traffic_matrix(fit, pl, topo, DispatchMode.DEDUP_RANK, n_dp=W, seed=seed)
    plan, info = plan_circuits(tm.counts, topo, cfg)
    oc = evaluate(ev, pl, with_circuits(topo, plan), cost, DispatchMode.DEDUP_RANK,
                  n_dp=W, seed=seed)
    gain = 100.0 * (1 - oc["bottleneck_us"] / eps["bottleneck_us"]) if eps["bottleneck_us"] else None
    return {
        "eps_bottleneck_us": round(eps["bottleneck_us"], 2),
        "ocs_us": round(oc["bottleneck_us"], 2),
        "ocs_gain_pct": None if gain is None else round(gain, 4),
        "cross_pod_fraction": round(eps["cross_pod_bytes"] / max(eps["network_bytes"], 1e-9), 4),
        "mean_fanout": round(eps["mean_fanout"], 4),
        "n_candidate_promotable_pairs": info["n_candidate_promotable_pairs"],
        "covered_fraction": info["promotable_traffic_covered_fraction"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    W, seed = args.world_size, args.seed
    rng = np.random.default_rng(seed)
    t = load_workload(args.workload / "manifest.json", decode_only=True)
    specs = build_suite(n_repeats=0)
    fu, eu = split_by_category(specs, seed=seed)
    present = {r.uid for r in t.runs}
    fit = t.by_runs([u for u in fu if u in present])
    ev = t.by_runs([u for u in eu if u in present])
    E = int(t.experts.max()) + 1

    cost = CostConfig(hidden_size=2048)
    base = hierarchy_for(W, "multi_pod")
    fab = FabricConfig(core_oversubscription=4.0, pod_oversubscription=1.0)
    topo = Topology(W, base.gpus_per_node, base.nodes_per_pod, fab, set(),
                    base.rank_to_slot, (Tier.CROSS_POD,))
    cfg = OcsConfig(n_circuits=max(4, W // 2), ports_per_rank=2)

    layers = np.sort(np.unique(fit.layers))
    lexp = layer_experts(fit, layers)
    trans = build_transitions(fit)
    L = len(layers)
    print(f"== affinity A/B/C: {args.workload.name} world={W} E={E} L={L} "
          f"({args.workload.name})")
    print(f"   fit {fit.n_cells} cells / eval {ev.n_cells} cells (disjoint categories)")
    print(f"   {len(trans)} inter-layer boundaries built from the FIT window only\n")

    results: dict = {}
    t0 = time.time()

    lin = make_placement("linear", fit, W, seed=seed)
    lin_pl = as_per_layer(lin, L)
    results["linear"] = score(lin, fit, ev, topo, cfg, cost, W, seed)

    A = make_placement("affinity_coordinated_layer", fit, W, seed=seed)
    A_pl = as_per_layer(A, L)
    results["A_intra_only"] = score(A, fit, ev, topo, cfg, cost, W, seed)
    results["A_intra_only"]["reroute_fraction"] = round(
        exflow_reroute_fraction(trans, A_pl), 4)
    results["A_intra_only"]["eps_proxy"] = round(eps_proxy(A_pl, lexp, layers, W), 4)

    B_pl, B_obj = exflow_local_search(trans, lin_pl.copy(), E, rng)
    B = Placement(B_pl, W, name="B_inter_only")
    results["B_inter_only"] = score(B, fit, ev, topo, cfg, cost, W, seed)
    results["B_inter_only"]["reroute_fraction"] = round(
        exflow_reroute_fraction(trans, B_pl), 4)
    results["B_inter_only"]["eps_proxy"] = round(eps_proxy(B_pl, lexp, layers, W), 4)

    for lam in LAMBDAS:
        C_pl, C_obj = combined_local_search(lexp, layers, trans, A_pl.copy(),
                                            E, W, lam, rng)
        name = f"C_both_lam{lam:g}"
        C = Placement(C_pl, W, name=name)
        r = score(C, fit, ev, topo, cfg, cost, W, seed)
        r["reroute_fraction"] = round(exflow_reroute_fraction(trans, C_pl), 4)
        r["eps_proxy"] = round(eps_proxy(C_pl, lexp, layers, W), 4)
        results[name] = r

    print(f"{'arm':<18}{'eps_us':>10}{'ocs_us':>10}{'gain%':>8}"
          f"{'reroute':>9}{'eps_proxy':>11}{'fanout':>8}{'cover':>8}")
    for name, v in results.items():
        g = v["ocs_gain_pct"]
        print(f"{name:<18}{v['eps_bottleneck_us']:>10.1f}{v['ocs_us']:>10.1f}"
              f"{(f'{g:.3f}' if g is not None else 'n/a'):>8}"
              f"{v.get('reroute_fraction', float('nan')):>9.4f}"
              f"{v.get('eps_proxy', float('nan')):>11.4f}"
              f"{v['mean_fanout']:>8.3f}{v['covered_fraction']:>8.4f}")
    print(f"\n   ({time.time()-t0:.1f} s)")

    dest = args.out or Path(f"outputs/affinity/ab_compare.{args.workload.name}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "workload": str(args.workload), "world_size": W, "E": E, "n_layers": L,
        "lambdas": list(LAMBDAS),
        "protocol": ("fit on window A, score on disjoint window B (leave-categories-out); "
                     "search minimises a cheap proxy, the verdict is evaluate()'s us"),
        "combined_objective": "eps_proxy + lam * reroute_fraction",
        "results": results}, indent=1))
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
