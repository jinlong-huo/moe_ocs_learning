"""
ocs_eval.py — optical circuit planning, reconfiguration accounting, and the
temporal-stability question that decides whether any of it is feasible.

The mechanism being modelled
────────────────────────────
An optical cross-connect gives a *dedicated* path between two endpoints.  It
does not make a link faster than the endpoints' NICs; what it removes is the
oversubscribed core layer that a cross-pod flow would otherwise share.  So a
circuit is a **tier promotion**: CROSS_POD -> OPTICAL (see
``cost_model.FabricConfig``).  Two consequences follow immediately and both are
enforced here rather than assumed away:

  * A circuit is worthless for a pair that is already intra-node or intra-pod.
    Provisioning one is legal and yields exactly zero benefit.
  * Therefore **OCS has no effect at all on a deployment that fits inside one
    pod.**  Reporting an OCS gain on an 8-rank experiment is a modelling
    artifact, not a result.

The feasibility question
────────────────────────
A MEMS optical cross-connect reconfigures in the millisecond range; fast
research switches reach microseconds.  One MoE all-to-all is tens of
microseconds.  So a circuit plan can NEVER track per-layer traffic, and can
only track per-request traffic if requests are long-lived.  ``breakeven``
turns this into an explicit number: how long an epoch must last for the saved
transfer time to exceed the reconfiguration time paid to enter it.

``stability`` then measures whether the traffic matrix actually holds still for
that long.  If the required epoch exceeds the observed stability horizon, a
dynamic controller is infeasible on this workload and only a static plan is
defensible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.eval.cost_model import (
    CostConfig, DispatchMode, Placement, Tier, Topology, evaluate,
    traffic_matrix,
)
from src.eval.trace_ir import CellTable


# ═══════════════════════════════════════════════════════════════════════
# Switch model
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OcsConfig:
    """Optical switch capability.

    ``reconfig_us`` reference points:
      MEMS beam-steering (Google Palomar / Apollo class)   ~1e4 us  (10 ms)
      Piezo / smaller MEMS                                 ~1e3 us  (1 ms)
      Fast research switches (SOA, AWGR + tunable laser)   ~1e0 us
    ``ports_per_rank`` is the number of simultaneous circuits one endpoint can
    hold; it is the degree bound of the matching problem, and the reason a
    circuit plan cannot simply cover every hot pair.
    """
    n_circuits: int = 32
    ports_per_rank: int = 2
    reconfig_us: float = 10_000.0
    parallel_reconfig: bool = True     # one switch action repoints all mirrors


RECONFIG_CLASSES = {
    "mems_10ms": 10_000.0,
    "mems_1ms": 1_000.0,
    "fast_10us": 10.0,
    "ideal_0": 0.0,
}


# ═══════════════════════════════════════════════════════════════════════
# Planning
# ═══════════════════════════════════════════════════════════════════════

def plan_circuits(counts: np.ndarray, topo: Topology, cfg: OcsConfig
                  ) -> tuple[set[frozenset], dict]:
    """Greedy degree-bounded matching over CROSS_POD pairs by traffic volume.

    Only cross-pod pairs are candidates, because only they can be promoted.
    The degree bound (``ports_per_rank``) makes this a b-matching; greedy on
    edge weight is the standard 1/2-approximation and is what a real controller
    would run.
    """
    T = topo.tier_matrix()
    D, W = counts.shape
    sym = np.zeros((topo.world_size, topo.world_size))
    d = min(D, topo.world_size)
    sym[:d, :W] += counts[:d, :W]
    sym[:W, :d] += counts[:d, :W].T

    promotable = {int(x) for x in topo.promote_from
                  if topo.fabric.oversubscription(Tier(int(x))) > 1.0
                  and topo.fabric.bandwidth(Tier.OPTICAL)
                  > topo.fabric.bandwidth(Tier(int(x)))}
    cands = []
    for a in range(topo.world_size):
        for b in range(a + 1, topo.world_size):
            if int(T[a, b]) in promotable and sym[a, b] > 0:
                cands.append((sym[a, b], a, b))
    cands.sort(reverse=True)

    deg = np.zeros(topo.world_size, dtype=np.int64)
    chosen: set[frozenset] = set()
    covered = 0.0
    for wgt, a, b in cands:
        if len(chosen) >= cfg.n_circuits:
            break
        if deg[a] >= cfg.ports_per_rank or deg[b] >= cfg.ports_per_rank:
            continue
        chosen.add(frozenset((a, b)))
        deg[a] += 1
        deg[b] += 1
        covered += wgt

    total_cross = float(sum(w for w, _, _ in cands))
    return chosen, {
        "n_candidate_promotable_pairs": len(cands),
        "n_circuits_provisioned": len(chosen),
        "circuit_budget": cfg.n_circuits,
        "ports_per_rank": cfg.ports_per_rank,
        "promotable_traffic_covered_fraction": (
            round(covered / total_cross, 6) if total_cross > 0 else 0.0),
        "port_saturated_ranks": int((deg >= cfg.ports_per_rank).sum()),
    }


def with_circuits(topo: Topology, circuits: set[frozenset]) -> Topology:
    """A copy of ``topo`` carrying a different circuit set (never mutates)."""
    return Topology(topo.world_size, topo.gpus_per_node, topo.nodes_per_pod,
                    topo.fabric, set(circuits), topo.rank_to_slot,
                    topo.promote_from)


# ═══════════════════════════════════════════════════════════════════════
# Reconfiguration economics
# ═══════════════════════════════════════════════════════════════════════

def breakeven(saved_us_per_layer_pass: float, n_layers: int,
              n_circuit_changes: int, cfg: OcsConfig) -> dict:
    """When does a reconfiguration pay for itself?

    ``saved_us_per_layer_pass`` is the bottleneck-time reduction of one MoE
    all-to-all (dispatch+combine) obtained by the new circuit set.  A token pass
    performs ``n_layers`` of them.  The reconfiguration is paid once on entering
    the epoch.
    """
    recon = cfg.reconfig_us * (1 if cfg.parallel_reconfig else max(1, n_circuit_changes))
    per_pass = saved_us_per_layer_pass * n_layers
    if per_pass <= 0:
        return {"reconfig_us": recon, "saved_us_per_token_pass": per_pass,
                "breakeven_token_passes": None, "feasible": False,
                "note": "circuit set saves nothing; reconfiguration is pure loss"}
    n = recon / per_pass
    return {
        "reconfig_us": recon,
        "saved_us_per_token_pass": round(per_pass, 4),
        "breakeven_token_passes": round(n, 2),
        "breakeven_ms_at_20ms_per_token": round(n * 20.0, 1),
        "feasible": bool(n < 1e4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Temporal stability of the traffic pattern
# ═══════════════════════════════════════════════════════════════════════

def _win_traffic(t: CellTable, placement: Placement, topo: Topology,
                 mode: DispatchMode, seed: int) -> np.ndarray:
    return traffic_matrix(t, placement, topo, mode, seed=seed).counts


def stability(t: CellTable, placement: Placement, topo: Topology,
              cfg: OcsConfig, mode: DispatchMode = DispatchMode.DEDUP_RANK,
              window: str = "run", n_windows: int = 12, seed: int = 0) -> dict:
    """How much does the traffic matrix — and the plan derived from it — move?

    ``window="run"``    each window is a group of sequences (request-level
                        churn: the realistic serving timescale).
    ``window="token"``  each window is a contiguous slice of token positions
                        (within-request drift).
    ``window="layer"``  each window is one MoE layer (the timescale a circuit
                        plan could never track; included to show why).

    ``plan_persistence`` is the metric a controller cares about: the Jaccard
    overlap of the circuit sets two windows would independently choose.  Low
    persistence means every window demands a reconfiguration.
    """
    wins: list[CellTable] = []
    if window == "run":
        ids = [r.uid for r in t.runs]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(ids))
        chunks = np.array_split(order, min(n_windows, max(2, len(ids) // 2)))
        for ch in chunks:
            if ch.size:
                wins.append(t.by_runs([ids[i] for i in ch]))
    elif window == "token":
        pmax = int(t.pos.max()) + 1
        edges = np.linspace(0, pmax, n_windows + 1).astype(int)
        for i in range(n_windows):
            m = (t.pos >= edges[i]) & (t.pos < edges[i + 1])
            if m.sum() > 0:
                wins.append(t.select(m))
    elif window == "layer":
        for l in t.layers:
            wins.append(t.by_layer(int(l)))
    else:
        raise ValueError(window)

    if len(wins) < 2:
        return {"insufficient": True, "n_windows": len(wins)}

    mats = [_win_traffic(w, placement, topo, mode, seed) for w in wins]
    plans = [plan_circuits(m, topo, cfg)[0] for m in mats]

    def cos(a, b):
        av, bv = a.ravel(), b.ravel()
        return float(av @ bv / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-18))

    def jac(a, b):
        if not a and not b:
            return 1.0
        return len(a & b) / max(len(a | b), 1)

    cs = [cos(mats[i], mats[j]) for i in range(len(mats))
          for j in range(i + 1, len(mats))]
    js = [jac(plans[i], plans[j]) for i in range(len(plans))
          for j in range(i + 1, len(plans))]
    adj_c = [cos(mats[i], mats[i + 1]) for i in range(len(mats) - 1)]
    adj_j = [jac(plans[i], plans[i + 1]) for i in range(len(plans) - 1)]

    return {
        "window": window, "n_windows": len(wins),
        "traffic_cosine_mean": round(float(np.mean(cs)), 6),
        "traffic_cosine_min": round(float(np.min(cs)), 6),
        "traffic_cosine_adjacent": round(float(np.mean(adj_c)) if adj_c else 0.0, 6),
        "plan_persistence_mean": round(float(np.mean(js)) if js else 0.0, 6),
        "plan_persistence_adjacent": round(float(np.mean(adj_j)) if adj_j else 0.0, 6),
        "plan_size_mean": round(float(np.mean([len(p) for p in plans])), 2),
        "reconfigurations_needed_per_window": round(
            float(np.mean([len(plans[i] ^ plans[i + 1]) for i in range(len(plans) - 1)]))
            if len(plans) > 1 else 0.0, 3),
    }


# ═══════════════════════════════════════════════════════════════════════
# The Q5 comparison
# ═══════════════════════════════════════════════════════════════════════

def ocs_comparison(fit: CellTable, ev: CellTable, placement: Placement,
                   topo: Topology, cfg: OcsConfig,
                   cost: CostConfig | None = None,
                   mode: DispatchMode = DispatchMode.DEDUP_RANK,
                   seed: int = 0) -> dict:
    """Static electrical vs static OCS vs oracle OCS, all scored on ``ev``.

    ``static_ocs`` provisions circuits from the FIT workload only, so its gain
    is honestly out-of-sample.  ``oracle_ocs`` provisions from the evaluation
    workload and is an upper bound no controller can beat; the gap between them
    is exactly the value of prediction, and if it is small then a static plan is
    all anyone needs.
    """
    cost = cost or CostConfig()
    base = evaluate(ev, placement, topo, cost, mode, seed=seed)

    tm_probe = traffic_matrix(ev, placement, topo, mode, seed=seed)
    _probe, probe_info = plan_circuits(tm_probe.counts, topo, cfg)
    if probe_info["n_candidate_promotable_pairs"] == 0:
        return {
            "applicable": False,
            "reason": (f"deployment occupies {topo.n_pods} pod(s) / "
                       f"{topo.n_nodes} node(s); no contended (oversubscribed) "
                       "rank pair carries traffic, so there is nothing for an "
                       "optical circuit to promote"),
            "baseline": {k: base[k] for k in
                         ("bottleneck_us", "network_bytes", "inter_node_bytes",
                          "cross_pod_bytes", "active_pairs")},
        }

    tm_fit = traffic_matrix(fit, placement, topo, mode, seed=seed)
    tm_ev = traffic_matrix(ev, placement, topo, mode, seed=seed)
    plan_fit, info_fit = plan_circuits(tm_fit.counts, topo, cfg)
    plan_ora, info_ora = plan_circuits(tm_ev.counts, topo, cfg)

    r_static = evaluate(ev, placement, with_circuits(topo, plan_fit), cost, mode, seed=seed)
    r_oracle = evaluate(ev, placement, with_circuits(topo, plan_ora), cost, mode, seed=seed)

    def gain(r):
        return round(100.0 * (1 - r["bottleneck_us"] / base["bottleneck_us"]), 4) \
            if base["bottleneck_us"] > 0 else 0.0

    n_layers = ev.n_layers
    saved_per_pass = (base["bottleneck_us"] - r_static["bottleneck_us"])
    be = breakeven(saved_per_pass, n_layers, len(plan_fit), cfg)

    return {
        "applicable": True,
        "baseline_electrical": {k: base[k] for k in
                               ("bottleneck_us", "cross_pod_bytes", "optical_bytes",
                                "network_bytes", "active_pairs")},
        "static_ocs": {**{k: r_static[k] for k in
                          ("bottleneck_us", "cross_pod_bytes", "optical_bytes")},
                       "bottleneck_reduction_pct": gain(r_static), **info_fit},
        "oracle_ocs": {**{k: r_oracle[k] for k in
                          ("bottleneck_us", "cross_pod_bytes", "optical_bytes")},
                       "bottleneck_reduction_pct": gain(r_oracle), **info_ora},
        "plan_overlap_fit_vs_oracle": round(
            len(plan_fit & plan_ora) / max(len(plan_fit | plan_ora), 1), 4),
        "value_of_prediction_pct": round(gain(r_oracle) - gain(r_static), 4),
        "reconfiguration": be,
        "reconfig_class_us": cfg.reconfig_us,
    }
