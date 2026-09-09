"""
harvest_sched.py — Harvest-style reconfiguration schedules over MoE dispatch.

What this adds, and why it fits here
────────────────────────────────────
The evidence chain treats one workload window as a *static* aggregate
(``traffic_matrix`` / ``evaluate``).  But a window's MoE layers execute strictly
in sequence, and each layer ℓ issues its own dispatch(+combine) round: a
*step sequence* <m_ℓ · M_ℓ> in exactly the sense the Harvest paper
(arXiv:2602.09188, reproduced in ``~/Downloads/Projects/harvest``) models a
collective:

    step ℓ   : per-layer demand matrix M_ℓ over ranks (from routing cells)
    topo  G  : EPS fabric + a *circuit set* (degree-bounded OCS plan)
    DCT_ℓ(G) : the repo's own bottleneck cost of layer ℓ under G
                (= ``evaluate(by_layer(ℓ), …, topo_G)["bottleneck_us"]``)
    schedule : which circuit set is live for which contiguous layer range,
               and where the optical switch rewires (paying ``alpha_r``).

Routing is placement/topology-independent (Q1, bit-exact), so M_ℓ is knowable
before any scheduling decision — the whole point of trace-guided planning.

No MILP anywhere (deliberate): the "topology pool" is the restricted,
deterministic set of circuit plans this repo already knows how to build
(EPS-only, window-fit greedy, per-layer greedy), and interval scoring is the
existing analytic bottleneck model, not an integer program.

The DP is Algorithm 1 / Theorem 1 of the Harvest paper:

    DP[a][t]  = min over cut b in (a, s] of  best_interval(a, b-1) + DP[b][t-1]
    DP[a][0]  = best_interval(a, s)                      (no further rewires)
    DP[s][*]  = 0                                        (sentinel)
    answer    = min over t of DP[0][t] + t * alpha_r     (Theorem-1 sweep)

Adjacent segments that reuse the same circuit set are merged, and alpha_r is
charged per *actual* circuit-set change (the physical-cost convention, as in
the Harvest reproduction).

What the numbers mean here
──────────────────────────
* k* = 0  → no rewire pays for itself: the DP *certifies* the static-plan
           conclusion of the OCS feasibility analysis instead of assuming it.
* k* > 0  → the within-pass structure is worth switching for (small alpha_r,
           big messages, cross-pod traffic present).  Segments tell the
           inference runtime *when* to repoint the switch (between layers).

All inputs are read-only; nothing in the existing evidence chain is modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.eval.cost_model import (
    CostConfig, DispatchMode, Placement, Topology, evaluate,
)
from src.eval.ocs_eval import OcsConfig, breakeven, plan_circuits, with_circuits
from src.eval.trace_ir import CellTable

INF = float("inf")


# ═══════════════════════════════════════════════════════════════════════
# Step sequence: one dispatch round per MoE layer
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StepTable:
    """The collective view of one workload window.

    ``layers``    ordered MoE layer ids that carry routing cells (the step
                  sequence axis; layers without cells are dropped).
    ``counts``    [s, D, W] message-count matrices, one per step.
    ``label``     human-readable id of the window (for reports).
    """
    layers: np.ndarray
    counts: np.ndarray
    label: str = "window"

    @property
    def s(self) -> int:
        return int(self.layers.shape[0])


def per_layer_steps(t: CellTable, placement: Placement, topo: Topology,
                    mode: DispatchMode = DispatchMode.DEDUP_RANK,
                    n_dp: int | None = None,
                    seed: int = 0) -> StepTable:
    """Build the per-layer dispatch step sequence of a routing trace.

    Step ℓ's demand is exactly the traffic matrix of the cells routed at
    layer ℓ — recovered from the trace, never assumed.  Layers with no routed
    cells contribute no step (there is nothing to communicate).
    """
    from src.eval.cost_model import traffic_matrix

    W = placement.world_size
    n_dp = n_dp or W
    rows: List[np.ndarray] = []
    kept: List[int] = []
    for l in t.layers:
        sub = t.by_layer(int(l))
        if sub.n_cells == 0:
            continue
        m = traffic_matrix(sub, placement, topo, mode, n_dp, seed)
        rows.append(m.counts)
        kept.append(int(l))
    if not rows:
        raise ValueError("trace has no routed cells to schedule")
    return StepTable(layers=np.asarray(kept, dtype=np.int32),
                     counts=np.stack(rows), label=f"{t.model_id}:{t.n_cells} cells")


# ═══════════════════════════════════════════════════════════════════════
# Restricted topology pool (no MILP): circuit sets this repo can already plan
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PoolMember:
    """One candidate "topology": EPS fabric plus a circuit set."""
    name: str
    circuits: frozenset
    topo: Topology

    def describe(self) -> dict:
        return {"name": self.name, "n_circuits": len(self.circuits)}


def circuit_pool(steps: StepTable, placement: Placement, topo: Topology,
                 cfg: OcsConfig) -> List[PoolMember]:
    """Deterministic restricted pool of circuit plans.

    Members (deduped by circuit set):
      * ``eps``            the empty set — the electrical-only baseline;
      * ``fit_static``     greedy plan over the window's whole traffic
                           (``plan_circuits`` on the summed matrix);
      * ``plan.<layer>``   greedy plan of each single layer's demand — the
                           "step-matching" candidates that let the DP decide
                           whether following the per-layer churn pays.

    Only CROSS_POD pairs are promotable (``plan_circuits`` honours this), so
    on realistic pod sizes every plan degenerates to ``eps`` and the DP will
    certify k* = 0 — the correct answer, reached by computation.
    """
    W = placement.world_size
    counts = steps.counts
    total = counts.sum(0)
    members: Dict[frozenset, str] = {frozenset(): "eps"}

    def add(mat: np.ndarray, name: str) -> None:
        circ, meta = plan_circuits(mat, topo, cfg)
        key = frozenset(circ)
        # prefer the most specific name when a set already exists
        if key not in members:
            members[key] = name

    add(total, "fit_static")
    for i, l in enumerate(steps.layers):
        add(counts[i], f"plan.L{l}")

    out = []
    for key, name in members.items():
        out.append(PoolMember(name, key, with_circuits(topo, set(key))))
    # deterministic order: eps first, then by name
    out.sort(key=lambda m: (m.name != "eps", m.name))
    return out


# ═══════════════════════════════════════════════════════════════════════
# Cost: DCT per step under a pool member (the repo's own bottleneck model)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StepCosts:
    """Precomputed per-step DCT matrix ``dct[i][p]`` (us)."""
    dct: np.ndarray            # [s, P]
    names: List[str]
    prefix: np.ndarray = field(init=False, default=None)   # [P, s+1] prefix sums

    def __post_init__(self):
        s, P = self.dct.shape
        self.prefix = np.zeros((P, s + 1))
        for p in range(P):
            self.prefix[p, 1:] = np.cumsum(self.dct[:, p])

    def interval(self, a: int, b: int, p: int) -> float:
        """Σ DCT over steps a..b (inclusive, 0-based) under member p."""
        return float(self.prefix[p, b + 1] - self.prefix[p, a])


def step_costs(t: CellTable, steps: StepTable, placement: Placement,
               pool: Sequence[PoolMember],
               cost: CostConfig | None = None,
               mode: DispatchMode = DispatchMode.DEDUP_RANK,
               n_dp: int | None = None,
               seed: int = 0) -> StepCosts:
    """Score every (step, pool member) pair with the evidence-chain model.

    This reuses ``evaluate`` verbatim: the DCT of a step is the bottleneck_us
    of that layer's routing slice under the member's circuit set.  Combine is
    included (``CostConfig.include_combine``), matching how the repo measures
    one layer's communication round.
    """
    W = placement.world_size
    n_dp = n_dp or W
    P = len(pool)
    s = steps.s
    dct = np.full((s, P), INF)
    for i, l in enumerate(steps.layers):
        sub = t.by_layer(int(l))
        for p, mem in enumerate(pool):
            rep = evaluate(sub, placement, mem.topo, cost, mode, n_dp, seed)
            dct[i, p] = float(rep["bottleneck_us"])
    return StepCosts(dct, [m.name for m in pool])


# ═══════════════════════════════════════════════════════════════════════
# Harvest DP (Algorithm 1 + Theorem 1), self-contained
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Segment:
    a: int                  # first step index (0-based, inclusive)
    b: int                  # last step index (inclusive)
    member: str             # pool member name
    n_circuits: int
    cost_us: float = 0.0    # Σ DCT over the segment (no rewire inside)

    def span(self) -> int:
        return self.b - self.a + 1


@dataclass
class Schedule:
    name: str
    segments: List[Segment]
    k: int                  # reconfigurations *allowed* by the DP state
    changes: int            # actual circuit-set changes after merging
    cost_no_reconf_us: float
    total_us: float         # + changes * alpha_r
    model_total_us: Optional[float] = None

    def summary(self) -> dict:
        return {
            "schedule": self.name, "segments": len(self.segments),
            "rewires": self.changes,
            "cost_no_reconf_us": round(self.cost_no_reconf_us, 4),
            "total_us": round(self.total_us, 4),
            "per_layer": [s.member for s in self.segments
                          for _ in range(s.span())],
            "segment_list": [
                {"a": int(s.a), "b": int(s.b), "member": s.member,
                 "n_circuits": int(s.n_circuits),
                 "cost_us": round(s.cost_us, 4)} for s in self.segments],
        }


def _best_interval(cost: StepCosts, a: int, b: int,
                   pool: Sequence[PoolMember]) -> Tuple[int, float]:
    """Eq. 5 of Harvest: argmin over the restricted pool of Σ DCT(a..b)."""
    best_p, best_c = 0, INF
    for p in range(len(pool)):
        c = cost.interval(a, b, p)
        # tie-break: fewer circuits (fewer future rewires to pay for)
        if c < best_c - 1e-12 or (
                abs(c - best_c) <= 1e-12
                and len(pool[p].circuits) < len(pool[best_p].circuits)):
            best_p, best_c = p, c
    return best_p, best_c


def _reconstruct(cost: StepCosts, pool: Sequence[PoolMember],
                 s: int, k: int) -> Tuple[List[Segment], float]:
    """Trace DP[0][k] into concrete segments (Harvest Algorithm 1)."""
    dp: List[List[float]] = [[INF] * (s + 1) for _ in range(k + 1)]
    nxt: List[List[int]] = [[0] * (s + 1) for _ in range(k + 1)]
    topo: List[List[int]] = [[0] * (s + 1) for _ in range(k + 1)]
    for t in range(k + 1):
        dp[t][s] = 0.0
    for a in range(s):
        p, c = _best_interval(cost, a, s - 1, pool)
        dp[0][a], topo[0][a] = c, p
    for t in range(1, k + 1):
        for a in range(s):
            best, arg, g = INF, s, 0
            for b in range(a + 1, s + 1):       # b = s  → no rewire after a
                p, c = _best_interval(cost, a, b - 1, pool)
                v = c + dp[t - 1][b]
                if v < best:
                    best, arg, g = v, b, p
            dp[t][a], nxt[t][a], topo[t][a] = best, arg, g

    segs: List[Segment] = []
    a, t = 0, k
    while t >= 0:
        g = topo[t][a]
        if t == 0 or nxt[t][a] >= s or g is None:
            if a < s and g is not None:
                segs.append(Segment(a=a, b=s - 1, member=pool[g].name,
                                    n_circuits=len(pool[g].circuits)))
            break
        b = nxt[t][a]
        segs.append(Segment(a=a, b=b - 1, member=pool[g].name,
                            n_circuits=len(pool[g].circuits)))
        a, t = b, t - 1
        if a >= s:
            break
    return segs, dp[k][0]


def _finalize(segs: List[Segment], cost: StepCosts,
              pool: Sequence[PoolMember], k: int,
              name: str, alpha_r_us: float) -> Schedule:
    """Merge same-member neighbours; charge alpha_r per actual change."""
    names = [m.name for m in pool]
    merged: List[Segment] = []
    for seg in segs:
        seg.cost_us = cost.interval(seg.a, seg.b, names.index(seg.member))
        if seg.cost_us == INF:
            return Schedule(name, [], k, 0, INF, INF)
        if merged and merged[-1].member == seg.member \
                and merged[-1].b + 1 == seg.a:
            prev = merged[-1]
            prev.b = seg.b
            prev.cost_us += seg.cost_us
        else:
            merged.append(seg)
    no_reconf = sum(x.cost_us for x in merged)
    changes = len(merged) - 1
    return Schedule(name, merged, k, changes, no_reconf,
                    no_reconf + changes * alpha_r_us)


def solve(steps: StepTable, cost: StepCosts, pool: Sequence[PoolMember],
          alpha_r_us: float) -> Schedule:
    """Theorem-1 sweep: best schedule over every rewire count k."""
    s = steps.s
    best: Optional[Schedule] = None
    for k in range(s):                       # k reconfigurations (≤ s-1 useful)
        segs, _ = _reconstruct(cost, pool, s, k)
        res = _finalize(segs, cost, pool, k, "harvest", alpha_r_us)
        if best is None or (res.total_us < best.total_us and res.total_us < INF):
            best = res
    if best is None or best.total_us == INF:
        raise RuntimeError("no feasible reconfiguration schedule")
    return best


def static_best(steps: StepTable, cost: StepCosts,
                pool: Sequence[PoolMember],
                alpha_r_us: float,
                eps_only: bool = False) -> Schedule:
    """Baseline 1: one circuit set for the whole sequence, never rewired."""
    names = [m.name for m in pool]
    if eps_only:
        idx = names.index("eps")
    else:
        idx, _ = _best_interval(cost, 0, steps.s - 1, pool)
    seg = Segment(a=0, b=steps.s - 1, member=pool[idx].name,
                  n_circuits=len(pool[idx].circuits),
                  cost_us=cost.interval(0, steps.s - 1, idx))
    return _finalize([seg], cost, pool, 0, "eps_static" if eps_only else "static_best",
                     alpha_r_us)


def bvn_per_step(steps: StepTable, cost: StepCosts, pool: Sequence[PoolMember],
                 alpha_r_us: float) -> Schedule:
    """Baseline 2: rewire before every layer to its own greedy plan (BvN)."""
    # the per-layer plan is a pool member named plan.L<layer>; fall back to
    # the interval best if the trace's per-layer plan collapsed to eps
    names = [m.name for m in pool]
    segs: List[Segment] = []
    for i, l in enumerate(steps.layers):
        want = f"plan.L{l}"
        p = names.index(want) if want in names else _best_interval(cost, i, i, pool)[0]
        segs.append(Segment(a=i, b=i, member=pool[p].name,
                            n_circuits=len(pool[p].circuits)))
    return _finalize(segs, cost, pool, max(0, steps.s - 1), "bvn_per_step",
                     alpha_r_us)


# ═══════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════

def schedule_report(t: CellTable, placement: Placement, base_topo: Topology,
                    ocs: OcsConfig,
                    cost: CostConfig | None = None,
                    mode: DispatchMode = DispatchMode.DEDUP_RANK,
                    n_dp: int | None = None,
                    seed: int = 0,
                    alpha_r_us: float | None = None) -> dict:
    """End-to-end: trace window -> step sequence -> pool -> DP -> report.

    ``alpha_r_us=None`` sweeps all ``RECONFIG_CLASSES`` so the report shows
    which optical switch class (if any) makes rewiring worthwhile.
    """
    from src.eval.ocs_eval import RECONFIG_CLASSES

    W = placement.world_size
    n_dp = n_dp or W
    steps = per_layer_steps(t, placement, base_topo, mode, n_dp, seed)
    pool = circuit_pool(steps, placement, base_topo, ocs)
    cst = step_costs(t, steps, placement, pool, cost, mode, n_dp, seed)
    topo0 = pool[0].topo

    report: dict = {
        "steps": [int(l) for l in steps.layers],
        "n_steps": steps.s,
        "window": steps.label,
        "world_size": W,
        "n_dp": n_dp,
        "dispatch_mode": mode.name,
        "placement": placement.name,
        "placement_scope": placement.scope,
        "topology": base_topo.describe(),
        "ocs": {"n_circuits": ocs.n_circuits, "ports_per_rank": ocs.ports_per_rank},
        "pool": [m.describe() for m in pool],
        "per_step_cost_us": {
            "layer": [int(l) for l in steps.layers],
            "members": [m.name for m in pool],
            "dct": [[round(float(x), 4) for x in row] for row in cst.dct],
        },
        "alpha_r_sweep": {},
    }

    eps_static = static_best(steps, cst, pool, 0.0, eps_only=True)
    report["eps_static_us"] = round(eps_static.cost_no_reconf_us, 4)
    report["pool_static_us"] = round(
        static_best(steps, cst, pool, 0.0).cost_no_reconf_us, 4)

    classes = (list(RECONFIG_CLASSES) if alpha_r_us is None
               else [k for k, v in RECONFIG_CLASSES.items()
                     if abs(v - alpha_r_us) < 1e-12] or ["custom"])
    for name in classes:
        ar = RECONFIG_CLASSES[name] if name != "custom" else alpha_r_us
        h = solve(steps, cst, pool, ar)
        bvn = bvn_per_step(steps, cst, pool, ar)
        report["alpha_r_sweep"][name] = {
            "alpha_r_us": ar,
            "harvest": h.summary(),
            "bvn_per_step": bvn.summary(),
            "saved_vs_eps_static_us": round(eps_static.total_us - h.total_us, 4),
            "saved_vs_eps_static_pct": round(
                100.0 * (1 - h.total_us / eps_static.total_us), 3)
            if eps_static.total_us > 0 else None,
            "beats_eps_static": bool(h.total_us < eps_static.total_us - 1e-9),
            "beats_bvn": bool(h.total_us < bvn.total_us - 1e-9),
            "certifies_static_k0": bool(h.changes == 0),
        }

    # amortisation: repeated token passes pay the rewire once per window
    if alpha_r_us is None:
        alpha_r_us = float(RECONFIG_CLASSES["mems_1ms"])
        h = solve(steps, cst, pool, alpha_r_us)
    saved_total = eps_static.cost_no_reconf_us - h.total_us
    if saved_total > 0 and h.changes:
        bk = breakeven(saved_total / max(1, steps.s), steps.s, h.changes,
                       OcsConfig(reconfig_us=alpha_r_us,
                                 parallel_reconfig=ocs.parallel_reconfig))
        report["breakeven_token_passes"] = bk
    else:
        report["breakeven_token_passes"] = None
    return report
