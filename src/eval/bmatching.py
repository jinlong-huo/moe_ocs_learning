#!/usr/bin/env python3
"""bmatching.py — exact circuit selection, and a test of the claim that the LP is exact.

WHY
───
```ocs_eval.plan_circuits``` is a greedy 1/2-approximation of a degree-bounded
b-matching over cross-pod rank pairs.  ```docs/ocs_moe_program.md``` V11 states that
the exact version is available "for free" because the polytope is integral, so the
LP relaxation is exact.

**That claim is worth testing rather than inheriting.**  The degree-constrained
subgraph polytope is integral for BIPARTITE graphs; a rank graph is not bipartite
(it is the complete graph on ranks), and the cardinality constraint
```sum z <= n_circuits``` is not part of that matroid structure either.  So this
module exposes both:

    plan_circuits_lp    LP relaxation (scipy linprog / HiGHS), reports integrality
    plan_circuits_milp  integer programme (scipy milp / HiGHS), exact by construction

and the comparison script measures whether they differ, and by how much.

Both mirror ```plan_circuits```'s semantics exactly: the same symmetric traffic
matrix, the same promotable-pair filter, the same degree bound and circuit budget.

Usage
─────
    python3.12 scripts/bmatching_vs_greedy.py --workload logs/workload/qwen36 --world-size 32
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import lil_matrix

from src.eval.cost_model import Tier, Topology


def candidate_pairs(counts: np.ndarray, topo: Topology):
    """The greedy planner's candidate set, reproduced exactly."""
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
    cands = [(sym[a, b], a, b)
             for a in range(topo.world_size)
             for b in range(a + 1, topo.world_size)
             if int(T[a, b]) in promotable and sym[a, b] > 0]
    cands.sort(reverse=True)
    return cands


def _build(counts, topo, cfg):
    W = topo.world_size
    cands = candidate_pairs(counts, topo)
    if not cands:
        return np.zeros(0), [], np.zeros(0), 0.0
    w = np.array([c[0] for c in cands], dtype=float)
    A = lil_matrix((W + 1, len(cands)))
    for j, (_, a, b) in enumerate(cands):
        A[a, j] = 1.0
        A[b, j] = 1.0
    A[W, :] = 1.0                                   # cardinality: sum z <= n_circuits
    ub = np.array([cfg.ports_per_rank] * W + [cfg.n_circuits], dtype=float)
    return w, cands, A.tocsr(), ub


def _result(z, cands, total_cross, cfg, extra=None):
    chosen = {frozenset((a, b)) for j, (_, a, b) in enumerate(cands) if z[j] > 0.5}
    deg = np.zeros(max((b for _, _, b in cands), default=0) + 1, dtype=np.int64)
    for _, a, b in cands:
        if frozenset((a, b)) in chosen:
            deg[a] += 1
            deg[b] += 1
    covered = float(sum(w for w, a, b in cands if frozenset((a, b)) in chosen))
    info = {
        "n_candidate_promotable_pairs": len(cands),
        "n_circuits_provisioned": len(chosen),
        "circuit_budget": cfg.n_circuits,
        "ports_per_rank": cfg.ports_per_rank,
        "promotable_traffic_covered_fraction": (
            round(covered / total_cross, 6) if total_cross > 0 else 0.0),
        "port_saturated_ranks": int((deg >= cfg.ports_per_rank).sum()),
    }
    if extra:
        info.update(extra)
    return chosen, info


def plan_circuits_lp(counts: np.ndarray, topo: Topology, cfg) -> tuple[set, dict]:
    """LP relaxation.  ```fractional_edges``` > 0 means the polytope is NOT integral."""
    w, cands, A, ub = _build(counts, topo, cfg)
    if not cands:
        return set(), {"n_candidate_promotable_pairs": 0,
                       "promotable_traffic_covered_fraction": 0.0}
    res = linprog(-w, A_ub=A, b_ub=ub, bounds=(0, 1), method="highs")
    z = res.x
    frac = int(np.sum((z > 1e-6) & (z < 1 - 1e-6)))
    total_cross = float(w.sum())
    return _result(z, cands, total_cross, cfg, {
        "solver": "linprog/highs", "status": int(res.status),
        "lp_objective": round(float(-res.fun), 3),
        "fractional_edges": frac, "polytope_integral": frac == 0,
    })


def plan_circuits_milp(counts: np.ndarray, topo: Topology, cfg,
                       time_limit: float = 30.0) -> tuple[set, dict]:
    """Integer programme — exact by construction, at a solver's cost."""
    w, cands, A, ub = _build(counts, topo, cfg)
    if not cands:
        return set(), {"n_candidate_promotable_pairs": 0,
                       "promotable_traffic_covered_fraction": 0.0}
    res = milp(c=-w, constraints=LinearConstraint(A, -np.inf, ub),
               integrality=np.ones(len(cands)),
               bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    z = res.x if res.x is not None else np.zeros(len(cands))
    total_cross = float(w.sum())
    return _result(z, cands, total_cross, cfg, {
        "solver": "milp/highs", "status": int(res.status),
        "milp_objective": round(float(-res.fun), 3) if res.fun is not None else None,
        "mip_gap": getattr(res, "mip_gap", None),
    })
