"""
milp_bound.py — optimality bounds for expert placement, so a heuristic result
can be reported as "within X% of optimal" instead of "better than random".

What is bounded, and what is not
────────────────────────────────
The placement heuristics in ``placement_opt.py`` minimise objectives that are
mostly *nonlinear* in the assignment: "how many distinct destination ranks does
this token reach" is a union cardinality, and a union is not a linear function
of the assignment.  There is no compact MILP that reproduces the true
bottleneck, so this module bounds two formulations instead, and each is labelled
with exactly what it does and does not imply.

1. ``linear_minmax`` — the **union-relaxed** objective.

       minimise  T   s.t.  sum_e reach[e] * x[e,r] <= T   for every rank r

   ``reach[e]`` is the number of routing cells selecting expert e.  Summing a
   rank's experts *double counts* cells that select two experts on the same
   rank, so this is an upper bound on that rank's dedup ingress.  It is
   apples-to-apples with ``placement_opt._lpt_map``, which minimises the same
   linearised quantity by bin-packing, so the claim it supports is
   **"LPT is within X% of the optimal min-max packing"** — not "the placement is
   within X% of optimal".

2. ``union_minmax`` — the **exact union** objective (true dedup ingress).

       y[c,r] >= x[e,r]   for every e in S_c      (y = union indicator)
       minimise T  s.t.  sum_c w_c * y[c,r] <= T

   Cell ``c`` sends one message to rank ``r`` iff at least one selected expert
   lives there; ``w_c`` is how many cells share that expert set.  This *is* the
   objective the repo's dedup metrics measure, but the full instance is too
   large to solve, so it is restricted to the ``max_sets`` heaviest distinct
   expert sets (retained traffic is reported).

Why the bound is an LP, and why that is the honest choice
─────────────────────────────────────────────────────────
Both formulations are solved twice:

  * a **linear relaxation** (all variables continuous), which HiGHS solves to
    optimality in seconds even at full ``max_sets``.  Its optimum is a *proven
    lower bound* on the corresponding integer optimum.
  * an optional **MIP** with a time limit, which can sharpen the bound and gives
    a feasible assignment to score.

The MIP alone would not support the claim: on the real instance the union MIP
found no feasible assignment within 60 s, and scipy reports no dual bound in
that case — a bound that exists only when the solver happens to succeed is not a
bound.  The LP bound always exists, so it is what the reported gaps use.

Both restrictions push the bound in the same, safe direction:

    LP(restricted)  <=  MIP(restricted)  <=  MIP(full) = true optimum

so a gap ``heuristic / bound - 1`` is an **upper bound on the heuristic's true
suboptimality**.  Reporting it as "at most X%" is rigorous; reporting it as
"exactly X%" would not be.

Every formulation fixes exactly ``E / W`` experts per rank, matching the
invariant every generator in ``placement_opt.py`` preserves, plus one symmetry
breaking constraint: ranks are interchangeable in these per-rank-sum objectives,
and without it the solver spends its budget re-deriving equivalent permutations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from src.eval.trace_ir import CellTable


@dataclass
class MilpResult:
    status: str
    objective: float | None          # LP optimum, or MIP incumbent if one exists
    dual_bound: float | None         # MIP: proven bound (None without an incumbent)
    mip_gap: float | None
    wall_s: float
    n_vars: int
    n_constraints: int
    assignment: np.ndarray | None    # [E] expert -> rank, if an incumbent exists
    relaxed: bool = False
    note: str = ""

    @property
    def bound(self) -> float | None:
        """The number a heuristic is divided by.

        For an LP relaxation the optimum *is* a proven bound.  For a MIP only
        ``mip_dual_bound`` is, and scipy leaves it unset when no incumbent was
        found — in which case this returns None rather than silently quoting the
        incumbent (an upper bound) as if it bounded the optimum from below.
        """
        if self.relaxed:
            return self.objective
        return self.dual_bound if self.dual_bound is not None else None

    def as_dict(self) -> dict:
        return {
            "kind": "LP relaxation" if self.relaxed else "MIP",
            "status": self.status,
            "objective": None if self.objective is None else round(float(self.objective), 6),
            "dual_bound": None if self.dual_bound is None else round(float(self.dual_bound), 6),
            "bound_used": None if self.bound is None else round(float(self.bound), 6),
            "mip_gap": None if self.mip_gap is None else round(float(self.mip_gap), 6),
            "wall_s": round(self.wall_s, 3),
            "n_vars": self.n_vars,
            "n_constraints": self.n_constraints,
            "has_incumbent": self.assignment is not None,
            "note": self.note,
        }


class _Rows:
    """Sparse constraint assembly with explicit logical row ids.

    Row ids must be allocated once and then reused across ``add`` calls when a
    logical constraint has entries in several columns (e.g. ``T - sum x <= 0``).
    Letting each ``add`` allocate fresh ids instead would inflate the declared
    matrix shape with all-zero trailing rows and misalign the bounds array
    against the rows they are supposed to constrain.
    """

    def __init__(self):
        self.rows: list[np.ndarray] = []
        self.cols: list[np.ndarray] = []
        self.vals: list[np.ndarray] = []
        self.lb: list[np.ndarray] = []
        self.ub: list[np.ndarray] = []
        self.n_rows = 0

    def new_rows(self, k: int, lb: float, ub: float) -> np.ndarray:
        ids = np.arange(self.n_rows, self.n_rows + k, dtype=np.int64)
        self.n_rows += k
        self.lb.append(np.full(k, float(lb)))
        self.ub.append(np.full(k, float(ub)))
        return ids

    def add(self, row_ids, cols, vals) -> None:
        self.rows.append(np.asarray(row_ids, dtype=np.int64))
        self.cols.append(np.asarray(cols, dtype=np.int64))
        self.vals.append(np.asarray(vals, dtype=np.float64))

    def build(self):
        return (np.concatenate(self.rows), np.concatenate(self.cols),
                np.concatenate(self.vals), np.concatenate(self.lb),
                np.concatenate(self.ub), self.n_rows)


def _solve(c, rows, cols, vals, lb, ub, n_rows, n_vars, integrality,
           time_limit, mip_rel_gap):
    """Assemble and solve one model.

    ``integrality`` is 1 for integer variables and 0 for continuous ones; the
    continuous block is always the trailing makespan variable ``T``, which is
    the only one allowed an infinite upper bound.  Bounding ``T`` by 1 would
    make every instance infeasible: it counts cells, not fractions.
    """
    import time
    n_bin = int(np.count_nonzero(integrality))
    hi = np.concatenate([np.ones(n_bin), np.full(n_vars - n_bin, np.inf)])
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_rows, n_vars))
    t0 = time.time()
    res = milp(c=c,
               constraints=[LinearConstraint(A, np.asarray(lb), np.asarray(ub))],
               integrality=np.asarray(integrality, dtype=np.float64),
               bounds=Bounds(np.zeros(n_vars), hi),
               options={"time_limit": float(time_limit),
                        "mip_rel_gap": float(mip_rel_gap), "presolve": True})
    return res, time.time() - t0


def _structure(E: int, W: int) -> tuple[int, int]:
    if E % W:
        raise ValueError(f"E={E} must be divisible by world_size={W}")
    return E // W, E * W


def _base_constraints(R: _Rows, E: int, W: int, cap: int) -> None:
    """(b) one rank per expert, (c) E/W experts per rank, (d) symmetry break."""
    e = np.repeat(np.arange(E), W)
    r = np.tile(np.arange(W), E)
    # (b) every expert is placed exactly once
    ids = R.new_rows(E, 1.0, 1.0)
    R.add(np.repeat(ids, W), e * W + r, np.ones(E * W))
    # (c) every rank holds exactly E/W experts
    ids = R.new_rows(W, float(cap), float(cap))
    R.add(np.tile(ids, E), e * W + r, np.ones(E * W))
    # (d) expert 0 -> rank 0
    ids = R.new_rows(1, 1.0, 1.0)
    R.add(ids, np.array([0]), np.array([1.0]))


# ═══════════════════════════════════════════════════════════════════════
# 1. Union-relaxed min-max load
# ═══════════════════════════════════════════════════════════════════════

def linear_minmax(reach: np.ndarray, world_size: int, *, relax: bool = False,
                  time_limit: float = 60.0, mip_rel_gap: float = 1e-4) -> MilpResult:
    """Optimal min-max packing of expert reach onto ranks.

    Variables ``x[e, r]`` plus one continuous makespan ``T``; the optimum is the
    exact bound for every heuristic that optimises this linearised objective,
    including ``placement_opt._lpt_map``.
    """
    E = int(reach.shape[0])
    W = int(world_size)
    cap, nx = _structure(E, W)
    n = nx + 1
    c = np.zeros(n)
    c[nx] = 1.0

    R = _Rows()
    ids = R.new_rows(W, -np.inf, 0.0)          # load_r - T <= 0, i.e. T >= load_r
    # Row ids are rank-major (rank r's id repeated E times), so the columns and
    # the weights must be rank-major too: within each rank block the expert index
    # varies fastest.  Pairing a rank-major row array with an expert-major column
    # array mixes two ranks into one constraint — a silent relaxation that still
    # solves and still returns a number, which is why the averaging check in
    # tests/test_promote_aware.py exists.
    R.add(np.repeat(ids, E),
          np.tile(np.arange(E), W) * W + np.repeat(np.arange(W), E),
          np.tile(reach.astype(float), W))
    R.add(ids, np.full(W, nx), -np.ones(W))
    _base_constraints(R, E, W, cap)
    rows, cols, vals, lb, ub, n_rows = R.build()

    integrality = (np.zeros(n) if relax
                   else np.concatenate([np.ones(nx), [0.0]]))
    res, wall = _solve(c, rows, cols, vals, lb, ub, n_rows, n, integrality,
                       time_limit, mip_rel_gap)
    assign = None
    if res.x is not None:
        cand = res.x[:nx].reshape(E, W)
        assign = np.argmax(cand, axis=1).astype(np.int32)
    return MilpResult(
        status=str(res.message),
        objective=(None if res.fun is None else float(res.fun)),
        dual_bound=(None if getattr(res, "mip_dual_bound", None) is None
                    else float(res.mip_dual_bound)),
        mip_gap=(None if getattr(res, "mip_gap", None) is None else float(res.mip_gap)),
        wall_s=wall, n_vars=n, n_constraints=n_rows, assignment=assign,
        relaxed=relax,
        note=("union-relaxed (linear) min-max load: double counts cells selecting "
              "two experts on one rank, so it bounds that rank's dedup ingress "
              "from above and is exactly the objective LPT bin-packing minimises"))


def linear_minmax_milp(reach, world_size, **kw) -> MilpResult:
    """Alias for the MIP variant (kept so call sites read explicitly)."""
    return linear_minmax(reach, world_size, relax=False, **kw)


# ═══════════════════════════════════════════════════════════════════════
# 2. Exact union min-max ingress — what can actually be bounded
# ═══════════════════════════════════════════════════════════════════════

def combinatorial_union_bound(experts: np.ndarray, num_experts: int,
                              world_size: int) -> dict:
    """A rigorous, cheap lower bound on min-max **dedup ingress**.

    Two arguments, both exact for any placement holding ``cap = E/W`` experts
    per rank:

    (a) *Perfectly-spread ideal.*  A cell's ``K`` experts occupy at most ``cap``
        of them per rank, so the cell reaches at least ``ceil(K/cap)`` distinct
        ranks.  Summing over cells, the total message count is at least
        ``N * ceil(K/cap)``, and the busiest rank is at least the ``1/W`` share
        of that.  No placement whatsoever can beat this, because it assumes the
        traffic is divided perfectly evenly.

    (b) *Largest expert reach.*  The most-selected expert ``e*`` must live on
        some rank, and every cell selecting it sends that rank a message, so
        some rank's ingress is at least ``reach[e*]``.

    The bound is ``max`` of the two.  It is honest but weak — it ignores that a
    rank owns only ``cap`` experts and must therefore cover many cells — and it
    is included precisely because the alternatives do not work: the LP
    relaxation of the exact union formulation is *slower* (it did not solve in
    60 s at 300 expert sets) and *looser* (41.2 against this bound's 64 for the
    same layer), and the MIP found no feasible assignment at all, so neither can
    support a published claim.
    """
    if num_experts % world_size:
        raise ValueError(f"E={num_experts} must be divisible by W={world_size}")
    cap = num_experts // world_size
    K = int(experts.shape[1])
    N = int(experts.shape[0])
    if cap <= 0:
        raise ValueError("world_size larger than the expert count")
    ranks_per_cell = int(np.ceil(K / cap))
    ideal = ranks_per_cell * N / world_size
    reach = np.bincount(experts.ravel(), minlength=num_experts)
    return {
        "bound": float(max(ideal, float(reach.max()))),
        "perfectly_spread_ideal": float(ideal),
        "largest_expert_reach": int(reach.max()),
        "ranks_touched_per_cell_min": ranks_per_cell,
        "experts_per_rank": cap,
        "note": ("lower bound on min-max dedup ingress for ANY placement with "
                 "E/W experts per rank; loose (it assumes perfect spreading) but "
                 "rigorous, and tighter than both the LP relaxation and the MIP "
                 "that could be solved on this instance"),
    }


def union_minmax(experts: np.ndarray, num_experts: int, world_size: int, *,
                 relax: bool = False, max_sets: int = 300,
                 time_limit: float = 60.0, mip_rel_gap: float = 1e-2) -> MilpResult:
    """Exact union min-max ingress as a MIP/LP — kept for completeness.

    ``y[c,r]`` is the union indicator, linked only from below
    (``y >= x[e,r]``): because ``y`` is minimised, the solver pushes it down to
    exactly the union, so no upper link is needed.

    In practice this does not yield a usable bound on a real layer (see
    ``combinatorial_union_bound``); it is retained so that claim is reproducible
    rather than asserted, and it is exercised on small instances by the tests.
    """
    E = int(num_experts)
    W = int(world_size)
    cap, nx = _structure(E, W)

    sets, weights = np.unique(np.sort(experts, axis=1), axis=0, return_counts=True)
    order = np.argsort(-weights)
    sets, weights = sets[order][:max_sets], weights[order][:max_sets]
    covered = float(weights.sum() / max(int(experts.shape[0]), 1))
    C, K = int(sets.shape[0]), int(sets.shape[1])

    ny = C * W
    n = nx + ny + 1
    c = np.zeros(n)
    c[nx + ny] = 1.0
    ybase = nx

    R = _Rows()
    # y[c,r] - x[e,r] >= 0 : C*K*W entries, one logical row per (c, k, r)
    ids = R.new_rows(C * K * W, 0.0, np.inf)
    ci = np.repeat(np.arange(C), K * W)
    kk = np.tile(np.repeat(np.arange(K), W), C)
    rr = np.tile(np.arange(W), C * K)
    R.add(ids, ybase + ci * W + rr, np.ones(ids.size))
    R.add(ids, sets[ci, kk].astype(np.int64) * W + rr, -np.ones(ids.size))
    # T - sum_c w_c y[c,r] >= 0
    ids = R.new_rows(W, -np.inf, 0.0)
    ci2 = np.tile(np.arange(C), W)
    R.add(np.repeat(ids, C), ybase + ci2 * W + np.repeat(np.arange(W), C),
          np.tile(weights.astype(float), W))
    R.add(ids, np.full(W, nx + ny), -np.ones(W))
    _base_constraints(R, E, W, cap)
    rows, cols, vals, lb, ub, n_rows = R.build()

    integrality = (np.zeros(n) if relax
                   else np.concatenate([np.ones(nx + ny), [0.0]]))
    res, wall = _solve(c, rows, cols, vals, lb, ub, n_rows, n, integrality,
                       time_limit, mip_rel_gap)
    assign = None
    if res.x is not None and not relax:
        cand = res.x[:nx].reshape(E, W)
        assign = np.argmax(cand, axis=1).astype(np.int32)
    return MilpResult(
        status=str(res.message),
        objective=(None if res.fun is None else float(res.fun)),
        dual_bound=(None if getattr(res, "mip_dual_bound", None) is None
                    else float(res.mip_dual_bound)),
        mip_gap=(None if getattr(res, "mip_gap", None) is None else float(res.mip_gap)),
        wall_s=wall, n_vars=n, n_constraints=n_rows, assignment=assign,
        relaxed=relax,
        note=(f"exact union (dedup ingress) over the {C} heaviest distinct expert "
              f"sets covering {100 * covered:.1f}% of the layer's cells; on a real "
              f"layer expect no incumbent and no dual bound — use "
              f"combinatorial_union_bound instead"))


def union_minmax_milp(experts, num_experts, world_size, **kw) -> MilpResult:
    """Alias for the MIP variant."""
    return union_minmax(experts, num_experts, world_size, relax=False, **kw)


# ═══════════════════════════════════════════════════════════════════════
# Heuristic scoring under the same objectives
# ═══════════════════════════════════════════════════════════════════════

def linearized_loads(expert_to_rank: np.ndarray, reach: np.ndarray) -> np.ndarray:
    """[W] union-relaxed load per rank for one layer (double counts)."""
    return np.bincount(expert_to_rank, weights=reach,
                       minlength=int(expert_to_rank.max()) + 1)


def true_ingress(experts: np.ndarray, expert_to_rank: np.ndarray,
                 world_size: int) -> np.ndarray:
    """[W] exact dedup ingress per rank for one layer (union semantics)."""
    dst = expert_to_rank[experts]                     # [N, K]
    out = np.zeros(world_size, dtype=np.int64)
    for r in range(world_size):
        out[r] = int((dst == r).any(axis=1).sum())
    return out


def optimality_report(fit: CellTable, world_size: int, *,
                      placements: dict[str, np.ndarray] | None = None,
                      layer: int | None = None,
                      lp_max_sets: int = 300,
                      mip_time_limit: float = 60.0,
                      run_mip: bool = True,
                      union_max_sets: int | None = None,
                      progress=None) -> dict:
    """Bound the heuristics on one layer under both objectives.

    The LP relaxations are always solved — they are the bounds actually used.
    The MIPs are optional sharpeners.  Both the linearised and the exact ingress
    are computed for every heuristic, so the reader can see how far apart the two
    objectives are, which is why the linear bound alone would mislead.
    """
    from src.eval.placement_opt import make_placement

    L = int(fit.layers[0]) if layer is None else int(layer)
    sub = fit.by_layer(L)
    ex = sub.experts
    E = fit.num_experts
    reach = np.bincount(ex.ravel(), minlength=E).astype(np.float64)
    lidx = int(np.flatnonzero(np.asarray(fit.layers) == L)[0])

    if placements is None:
        placements = {}
        # ``load_balanced_layer`` is the apples-to-apples reference for the
        # per-layer MILP: the pooled ``load_balanced`` optimises one map against
        # the whole workload, so beating it is not evidence about the optimiser.
        for kind in ("linear", "random", "load_balanced", "load_balanced_layer",
                     "affinity_layer", "affinity_coordinated_layer"):
            p = make_placement(kind, fit, world_size, seed=0)
            placements[kind] = (p.expert_to_rank if p.scope == "global"
                                else p.expert_to_rank[lidx])

    def score(m):
        ing = true_ingress(ex, m, world_size)
        return {"linearized_max": float(linearized_loads(m, reach).max()),
                "true_ingress_max": int(ing.max()),
                "true_ingress_sum": int(ing.sum())}

    out: dict = {
        "layer": L, "n_cells": int(sub.n_cells), "num_experts": E,
        "world_size": world_size,
        "n_distinct_sets": int(np.unique(np.sort(ex, axis=1), axis=0).shape[0]),
        "heuristics": {k: score(v) for k, v in placements.items()},
    }
    ums = lp_max_sets if union_max_sets is None else union_max_sets

    if progress:
        progress("  linear LP ...")
    out["linear_lp"] = linear_minmax(reach, world_size, relax=True,
                                     time_limit=300.0).as_dict()
    # The exact-union objective gets the cheap rigorous bound, not an LP or a
    # MIP: on a real layer both are unusable (see combinatorial_union_bound).
    out["union_bound"] = combinatorial_union_bound(ex, E, world_size)
    if progress:
        progress(f"  union bound = {out['union_bound']['bound']:.1f} "
                 f"(ideal {out['union_bound']['perfectly_spread_ideal']:.1f}, "
                 f"max expert reach {out['union_bound']['largest_expert_reach']})")

    if run_mip:
        if progress:
            progress(f"  linear MIP ({mip_time_limit:.0f}s) ...")
        lin_mip = linear_minmax(reach, world_size, relax=False,
                                time_limit=mip_time_limit)
        out["linear_mip"] = lin_mip.as_dict()
        if lin_mip.assignment is not None:
            out["linear_mip"]["achieved"] = score(lin_mip.assignment)
        if progress:
            progress(f"  union MIP ({mip_time_limit:.0f}s, <= {ums} sets) ...")
        uni_mip = union_minmax(ex, E, world_size, relax=False, max_sets=ums,
                               time_limit=mip_time_limit)
        out["union_mip"] = uni_mip.as_dict()

    # ── gaps, each against the best *proven lower bound* available ────
    lin_bound = out["linear_lp"].get("bound_used")
    lin_mip_bound = out.get("linear_mip", {}).get("bound_used")
    if lin_mip_bound is not None:
        lin_bound = max(lin_bound or 0.0, lin_mip_bound)
    uni_bound = out["union_bound"]["bound"]

    gaps: dict = {}
    if lin_bound:
        gaps["linear_bound"] = round(float(lin_bound), 6)
        gaps["linear_vs_bound_pct"] = {
            k: round(100.0 * (v["linearized_max"] / lin_bound - 1.0), 3)
            for k, v in out["heuristics"].items()}
        gaps["linear_basis"] = (
            "denominator is a proven lower bound on the optimum of the "
            "union-relaxed (LPT-style) min-max objective, so each percentage is "
            "an UPPER bound on the heuristic's true suboptimality for that "
            "objective — it may in fact be closer to optimal. Compare "
            "load_balanced_layer, not the pooled load_balanced, against it")
    if uni_bound:
        gaps["union_bound_value"] = round(float(uni_bound), 6)
        gaps["union_vs_bound_pct"] = {
            k: round(100.0 * (v["true_ingress_max"] / uni_bound - 1.0), 3)
            for k, v in out["heuristics"].items()}
        gaps["union_basis"] = (
            "denominator is a rigorous but loose lower bound on exact "
            "dedup-ingress min-max (perfect spreading / largest expert reach); "
            "each percentage is an upper bound on the true suboptimality, and "
            "the true gap may be far smaller")
    out["gaps"] = gaps
    return out
