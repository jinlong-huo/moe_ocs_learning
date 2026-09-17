#!/usr/bin/env python3
"""
affinity_milp.py — token→expert affinity, turned into an expert→rank placement
by an integer program.

This file contains **only the MILP**, written to be read. It imports the
existing trace loader and affinity builder rather than reimplementing them, and
it changes nothing else in the repository.

    python3 scripts/affinity_milp.py --toy          # hand-checkable, brute-forced
    python3 scripts/affinity_milp.py --workload logs/workload/qwen36 \
        --world-size 32 --layer 0 --top-pairs 400 --time-limit 60

──────────────────────────────────────────────────────────────────────────────
THE PROBLEM
──────────────────────────────────────────────────────────────────────────────
We are given an **expert affinity graph** A[e,f] ≥ 0 — how strongly experts e
and f are co-activated by tokens from a given model on given inputs — and a
world size W. We must place each of E experts onto one of W ranks, with exactly
cap = E/W experts per rank, so that co-activated experts share a rank.

Why sharing a rank helps: a token whose selected experts live on the same rank
needs no network hop for them (and under dedup dispatch it needs fewer
*messages*, since one message serves several experts on that rank). So the
quantity to maximise is the **affinity cut kept inside ranks**:

        maximise   Σ_{e<f}  A[e,f] · [ rank(e) == rank(f) ]        (★)

(★) is the classic balanced-graph-partitioning objective, and it is *quadratic*
in the decision variables — the indicator [rank(e)==rank(f)] is a product of two
assignment variables. Section 1 shows the standard linearisation that turns it
into a MILP.

This is the same family of formulation as ExFlow's expert-placement integer
program (arXiv:2401.08383 §4), which maximises inter-layer affinity captured
locally; we keep the single-layer version here for clarity.

──────────────────────────────────────────────────────────────────────────────
WHAT THIS FILE DOES **NOT** DO
──────────────────────────────────────────────────────────────────────────────
1. It optimises the *affinity* objective only. It does not know about tiers,
   bandwidths, or which rank pair a message crosses. If your real metric is
   wall-clock, recall that affinity is a proxy: affinity-cut → message counts →
   tier-aware bytes, and the three rank placements differently.
2. It solves **one layer**. Solving every layer independently is the trap the
   repo already measured: per-layer placement can be ~2× worse than random
   because the collective's critical path is the per-rank ingress *accumulated
   across all layers*, not each layer's own maximum. The fix is one extra
   constraint block per layer against a running accumulator — see §5.
3. It restricts attention to the top `--top-pairs` affinity pairs by default.
   That is what makes it tractable (see §2), and it means the result is a
   bound/heuristic on a *restricted* instance, not a proof about the full one.

──────────────────────────────────────────────────────────────────────────────
WHAT THE SCALING ACTUALLY LOOKS LIKE (measured, HiGHS via scipy.optimize.milp)
──────────────────────────────────────────────────────────────────────────────
  instance                        vars     result
  ------------------------------  -------  -----------------------------------
  toy  E=8,  W=2, all pairs            72  optimal, == brute force (108)
  E=256, W=32, top-10 pairs         8 512  optimal in 0.1 s  -> but see (i)
  E=256, W=32, top-25 pairs         8 992  optimal in 0.1 s
  E=256, W=32, top-50 pairs         9 792  optimal in 9.2 s
  E=256, W=32, top-400 pairs       20 992  NO incumbent in 90 s
  E=64,  W=4,  all 2 016 pairs      8 320  no incumbent in 141 s; proven ceiling
                                          2.65x above what greedy already gets
  E=64,  W=8,  all 2 016 pairs     16 640  no incumbent, no bound, in 234 s

Three lessons, all measured on the real traces:

  (i)   **Pruning the pair list is NOT a harmless tractability lever.** Solving
        the top-50-pair instance to *proven optimality* produced a placement
        scoring 59 % WORSE than the greedy baseline on the full affinity graph
        (80 % worse at top-10). The pruned objective is a different, much
        smaller problem: the sum over all ~19 000 nonzero pairs is dominated by
        thousands of moderate weights that pruning discards, so the optimiser
        spends its whole co-location budget on a handful of pairs. Any number
        from a pruned instance describes the pruned instance, nothing else.

  (ii)  **At realistic sizes the MILP neither finds a good placement nor proves
        a useful bound.** Where it returned anything, the proven ceiling sat
        2.65x above the greedy's achieved value — a certificate that certifies
        nothing.

  (iii) An incumbent's objective value can be **vacuous**. On the E=64/W=4 run
        the solver reported a feasible solution with objective 0 because every w
        was 0 (legal under w ≤ x) and it never improved them; re-scoring the
        *assignment* gave 163 680. Always score the returned placement directly
        with ``affinity_score``; never quote the solver's objective as the
        quality of the placement it produced.

The honest conclusion: (★) is balanced graph partitioning. Use a multilevel
partitioner or the repo's own local search for the placement, and keep the MILP
for certifying optimality on instances small enough to close. Its value here is
the bound on toy-sized problems, not a mechanism for E=256.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ═══════════════════════════════════════════════════════════════════════════
# 1. THE MODEL
# ═══════════════════════════════════════════════════════════════════════════
#
# VARIABLES
#
#   x[e, r] ∈ {0,1}     1 if expert e is placed on rank r
#   w[p, r] ∈ {0,1}     1 if BOTH experts of candidate pair p = (e,f) are on r
#
# CONSTRAINTS
#
#   (A) Σ_r x[e,r] = 1              every expert is placed exactly once
#   (B) Σ_e x[e,r] = E/W            every rank holds exactly cap experts
#   (C) w[p,r] ≤ x[e_p, r]          }  w = 1 is only legal when both experts
#   (D) w[p,r] ≤ x[f_p, r]          }  really are on rank r
#   (E) x[0,0] = 1                  symmetry breaking (see §4)
#
# OBJECTIVE
#
#   maximise  Σ_p  A[e_p, f_p] · Σ_r w[p,r]
#
# THE TRICK WORTH UNDERSTANDING (why (C)/(D) need no lower bound):
# the constraint [rank(e)==rank(f)] is NOT a product here. We introduced w as an
# *upper* bound on that product and put w in the objective with a POSITIVE
# coefficient. A maximiser therefore pushes every w up as far as the constraints
# allow — and (C)/(D) stop it exactly at min(x[e,r], x[f,r]), which IS the
# product for binary variables. No w ≤ x[e,r] + x[f,r] − 1 row is needed.
#
#   If the objective MINIMISED something (e.g. the cut weight
#   Σ_p A_p − Σ_p A_p Σ_r w[p,r]), the same encoding still works, because the
#   minimiser then pushes w DOWN and the missing lower link would be the bug.
#   Rule of thumb: with a reward, bound the AND from above; with a penalty,
#   bound it from below. Getting this backwards yields a model that solves
#   happily and answers a different question.
#
# SIZE
#
#   binaries  = E·W  +  P·W
#   rows      = E + W + 2·P·W + 1
#   For E=256, W=32, P=2000:  8 192 + 64 000 = 72 192 binaries, 128 289 rows.
#   Unrestricted (P = all 32 640 pairs) is ~1 M binaries: that is the wall that
#   makes people use greedy clustering instead.


def build_pairs(A: np.ndarray, top_pairs: int | None) -> tuple[np.ndarray, np.ndarray]:
    """The candidate pair list: (e, f) with e < f, strongest affinity first.

    Only pairs that could ever pay are worth a variable. ``top_pairs=None`` keeps
    every pair; anything smaller is the tractability lever.
    """
    E = A.shape[0]
    iu = np.triu_indices(E, k=1)
    w = A[iu]
    keep = np.ones(w.shape, dtype=bool)
    if top_pairs is not None and top_pairs < w.size:
        thr = np.partition(w, -top_pairs)[-top_pairs]
        keep = w >= thr
    return np.stack([iu[0][keep], iu[1][keep]], axis=1), w[keep]


def solve_affinity_partition(A: np.ndarray, world_size: int, *,
                             top_pairs: int | None = 400,
                             time_limit: float = 60.0,
                             mip_rel_gap: float = 1e-4,
                             relax: bool = False,
                             verbose: bool = True) -> dict:
    """Solve (★) as a MILP. ``relax=True`` solves the LP relaxation instead.

    Returns a dict: assignment, the two objective values, bounds, gap, timings
    and the model dimensions — everything needed to judge the answer.
    """
    E = int(A.shape[0])
    W = int(world_size)
    if E % W:
        raise ValueError(f"E={E} must be divisible by world_size={W}")
    cap = E // W

    pairs, pair_w = build_pairs(A, top_pairs)
    P = int(pairs.shape[0])
    nx, nw = E * W, P * W
    n = nx + nw

    # ── objective: maximise Σ_p A_p Σ_r w[p,r]  →  minimise its negation ────
    c = np.zeros(n)
    for pi in range(P):                       # vectorised enough; P is small
        c[nx + pi * W: nx + (pi + 1) * W] = -pair_w[pi]

    rows, cols, vals, lb, ub = [], [], [], [], []
    n_rows = 0

    def new_rows(k: int, lo: float, hi: float) -> np.ndarray:
        nonlocal n_rows
        ids = np.arange(n_rows, n_rows + k, dtype=np.int64)
        n_rows += k
        lb.extend([lo] * k)
        ub.extend([hi] * k)
        return ids

    def add(ids, cc, vv):
        rows.extend(np.asarray(ids, dtype=np.int64).tolist())
        cols.extend(np.asarray(cc, dtype=np.int64).tolist())
        vals.extend(np.asarray(vv, dtype=np.float64).tolist())

    # (A) Σ_r x[e,r] = 1
    ids = new_rows(E, 1.0, 1.0)
    for e in range(E):
        add([ids[e]] * W, [e * W + r for r in range(W)], [1.0] * W)
    # (B) Σ_e x[e,r] = cap
    ids = new_rows(W, float(cap), float(cap))
    for r in range(W):
        add([ids[r]] * E, [e * W + r for e in range(E)], [1.0] * E)
    # (C) w[p,r] − x[e_p,r] ≤ 0   and   (D) w[p,r] − x[f_p,r] ≤ 0
    ids = new_rows(2 * P * W, -np.inf, 0.0)
    for pi in range(P):
        e, f = int(pairs[pi, 0]), int(pairs[pi, 1])
        for r in range(W):
            j = nx + pi * W + r
            # one row, two entries: [w[p,r] = +1, x[e,r] = -1]  <= 0
            add([ids[pi * W + r]] * 2, [j, e * W + r], [1.0, -1.0])
            add([ids[P * W + pi * W + r]] * 2, [j, f * W + r], [1.0, -1.0])
    # (E) symmetry breaking: expert 0 → rank 0
    ids = new_rows(1, 1.0, 1.0)
    add(ids, [0], [1.0])

    A_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_rows, n))
    integrality = np.zeros(n) if relax else np.ones(n)

    if verbose:
        print(f"   model: {n:,} vars ({nx:,} x + {nw:,} w), "
              f"{n_rows:,} rows, {P:,} pairs "
              f"{'[LP relaxation]' if relax else '[MIP]'}")

    t0 = time.time()
    res = milp(
        c=c,
        constraints=[LinearConstraint(A_mat, np.asarray(lb), np.asarray(ub))],
        integrality=integrality,
        bounds=Bounds(np.zeros(n), np.ones(n)),
        options={"time_limit": float(time_limit), "mip_rel_gap": float(mip_rel_gap),
                 "presolve": True},
    )
    wall = time.time() - t0

    assignment = None
    if res.x is not None:
        assignment = np.argmax(res.x[:nx].reshape(E, W), axis=1).astype(np.int32)

    # scipy minimises; the affinity value is the negated objective.
    value = None if res.fun is None else -float(res.fun)
    # For a MAXIMISATION the dual bound is an UPPER bound on the optimum.
    dual = getattr(res, "mip_dual_bound", None)
    dual = None if dual is None else -float(dual)

    return {
        "status": str(res.message),
        "value": value,                    # achieved affinity-cut score (incumbent)
        "upper_bound": dual,               # proven ceiling on the optimum
        "assignment": assignment,
        "n_pairs_used": P,
        "n_pairs_total": E * (E - 1) // 2,
        "n_vars": n,
        "n_rows": n_rows,
        "wall_s": wall,
        "relaxed": relax,
        "mip_gap": (None if getattr(res, "mip_gap", None) is None
                    else float(res.mip_gap)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. WHY THE PAIR LIST IS PRUNED
# ═══════════════════════════════════════════════════════════════════════════
# The affinity matrix has E(E−1)/2 candidate pairs: 32 640 for E=256. Every pair
# brings W more binaries, so the full model is ~1 M binaries — far past what a
# general MILP solver will close. Two honest responses:
#
#   (a) keep the top-P pairs and say so. The result is then optimal *for the
#       restricted instance*, and since dropping pairs can only remove reward,
#       the restricted optimum is ≤ the true optimum. So the number you report
#       is a LOWER bound on the achievable score, and a gap measured against it
#       OVERSTATES how suboptimal your heuristic is — the safe direction.
#   (b) solve the LP relaxation on the full pair set for a cheap, weak ceiling.
#
# Either way, state which one you did. A MILP answer on a pruned instance is not
# an answer about the unpruned problem.


# ═══════════════════════════════════════════════════════════════════════════
# 3. SCORING AND THE BASELINE
# ═══════════════════════════════════════════════════════════════════════════

def affinity_score(A: np.ndarray, mapping: np.ndarray) -> float:
    """The true objective (★) of a placement, computed directly — no w variables.

    Always evaluate the returned assignment with this, never with the MILP's
    reported objective: an objective value computed by the solver on a pruned
    instance is not the same number as the score of the placement it produced.
    """
    same = mapping[:, None] == mapping[None, :]
    return float((A * same).sum() / 2.0)      # each pair counted once


def brute_force_best(A: np.ndarray, world_size: int) -> tuple[float, np.ndarray]:
    """Exhaustive optimum of (★) — only usable on tiny instances.

    This is the correctness check for the formulation: if the MILP's solution
    does not match this on a toy instance, the model is wrong, not the solver.
    """
    E = A.shape[0]
    W = world_size
    cap = E // W
    best, best_m = -np.inf, None
    experts = list(range(E))
    for combo in itertools.combinations(experts[1:], cap - 1):
        group0 = (0,) + combo                      # expert 0 pinned: symmetry
        rest = [e for e in experts if e not in group0]
        # enumerate the remaining partition recursively
        for mapping in _partitions(rest, W - 1, cap, np.full(E, -1, np.int32),
                                   group0, W):
            s = affinity_score(A, mapping)
            if s > best:
                best, best_m = s, mapping.copy()
    return best, best_m


def _partitions(rest, ranks_left, cap, mapping, group0, W):
    """Yield balanced assignments of ``rest`` to ranks 1..W-1."""
    if ranks_left <= 1:
        m = mapping.copy()
        m[list(group0)] = 0
        m[list(rest)] = W - 1
        yield m
        return
    for combo in itertools.combinations(rest, cap):
        m = mapping.copy()
        m[list(combo)] = W - ranks_left
        remaining = [e for e in rest if e not in combo]
        yield from _partitions(remaining, ranks_left - 1, cap, m, group0, W)


def greedy_score(A: np.ndarray, world_size: int) -> tuple[float, np.ndarray]:
    """The repository's existing greedy affinity clustering, for comparison.

    Reused rather than reimplemented, so the baseline is identical to the one
    the rest of the pipeline uses.
    """
    from src.eval.placement_opt import _greedy_cluster
    m = _greedy_cluster(A, world_size)
    return affinity_score(A, m), m


# ═══════════════════════════════════════════════════════════════════════════
# 4. SYMMETRY BREAKING, AND WHEN IT IS NO LONGER VALID
# ═══════════════════════════════════════════════════════════════════════════
# Constraint (E) pins expert 0 to rank 0. Without it, the model has W! equivalent
# optima that differ only by relabelling ranks, and a branch-and-bound solver
# spends its budget re-deriving permutations. Measurement on the real instance:
# without (E) the solver found no incumbent at all within 30 s; with it, a good
# one almost immediately.
#
# This is valid ONLY while the objective is symmetric in the ranks — true for
# (★), where a rank is just a bucket. The moment the objective becomes
# topology-aware (tier-weighted bytes, per-rank ingress limits, "rank 0 is on a
# different leaf switch from rank 1"), ranks are no longer interchangeable and
# (E) must be dropped or weakened to a partial ordering. Keeping it after adding
# tier structure silently removes valid solutions — and it still returns an
# answer, which is what makes it dangerous.


# ═══════════════════════════════════════════════════════════════════════════
# 5. THE COORDINATION CORRECTION (why one layer is not enough)
# ═══════════════════════════════════════════════════════════════════════════
# Optimising each layer against (★) independently is the mistake this repo has
# already measured: per-layer affinity clustering came out ~2× WORSE than random
# placement. The reason is that the collective's critical path is
#
#       max over ranks of  Σ_l  ingress_l(rank)
#
# i.e. what a rank ACCUMULATES over all L layers — but independent per-layer
# optimisation minimises Σ_l max_rank ingress_l(rank). Those differ, and each
# layer cheerfully piles its own hot spot onto the same rank.
#
# To make this MILP coordinated, solve layers in order and carry the load
# forward. One extra block per layer, against the running accumulator acc[r]:
#
#       for every rank r:   acc[r] + Σ_e reach_l[e] · x[e,r] ≤ T        (F)
#       minimise T, with the affinity reward as a secondary objective
#
# where reach_l[e] is how many cells in layer l select expert e. That makes the
# model a min-max load problem with an affinity reward — solvable, but note what
# changed: T is what you actually care about, and the affinity term is now a
# regulariser steering *which* of the many load-optimal placements you get. If
# you want the affinity objective to lead, keep it in the objective and put (F)
# in the constraints as a cap.
#
# A ready-made version of the min-max side of that model (without the affinity
# reward) is in src/eval/milp_bound.py — see linear_minmax / union_minmax.


# ═══════════════════════════════════════════════════════════════════════════
# 6. DRIVERS
# ═══════════════════════════════════════════════════════════════════════════

def toy_instance() -> tuple[np.ndarray, int]:
    """A tiny instance whose optimum you can check by hand.

    E=8, W=2 so cap=4. The planted structure: {0,1,2,3} co-activate strongly and
    {4,5,6,7} co-activate strongly, with weak cross pairs. The best balanced
    partition is therefore {0,1,2,3} | {4,5,6,7}.

    Score by hand (each pair counted once):
        within {0,1,2,3}: 6 pairs × 9 = 54
        within {4,5,6,7}: 6 pairs × 9 = 54
        cross: 16 pairs × 1 = 16, none of it kept
        total = 108
    Any mixed partition keeps fewer strong pairs, so 108 is the optimum.
    """
    E = 8
    A = np.ones((E, E))
    A[:4, :4] = 9.0
    A[4:, 4:] = 9.0
    np.fill_diagonal(A, 0.0)
    return A, 2


def run_toy(args) -> int:
    A, W = toy_instance()
    print("== TOY INSTANCE (E=8, W=2, cap=4) — verify by hand, then by brute force")
    print(f"   optimum by hand      : 108  (see toy_instance docstring)")

    bf_value, bf_map = brute_force_best(A, W)
    print(f"   brute force over all balanced partitions: {bf_value:.1f} "
          f"-> {bf_map.tolist()}")

    g_value, g_map = greedy_score(A, W)
    print(f"   greedy clustering   : {g_value:.1f} -> {g_map.tolist()}")

    out = solve_affinity_partition(A, W, top_pairs=None, time_limit=30.0,
                                   verbose=True)
    print(f"   MILP                : {out['value']} ({out['status'][:40]})")
    if out["assignment"] is not None:
        print(f"   MILP assignment     : {out['assignment'].tolist()}")
        print(f"   MILP score re-scored: "
              f"{affinity_score(A, out['assignment']):.1f}")

    lp = solve_affinity_partition(A, W, top_pairs=None, time_limit=30.0,
                                  relax=True, verbose=True)
    print(f"   LP relaxation ceiling: {lp['value']}  "
          f"(>= the integer optimum, as it must be for a maximisation)")

    ok = (out["assignment"] is not None
          and abs(affinity_score(A, out["assignment"]) - bf_value) < 1e-6)
    print(f"\n   VERDICT: MILP == brute force? {'YES' if ok else 'NO — model is wrong'}")
    return 0 if ok else 1


def run_trace(args) -> int:
    from src.eval.affinity_graph import affinity_matrix, layer_affinities
    from src.eval.trace_ir import load_workload

    t = load_workload(args.workload / "manifest.json", decode_only=True)
    layer = args.layer
    sub = t.by_layer(layer)
    print(f"== {t.model_id}  E={t.num_experts} K={t.top_k}  layer {layer}  "
          f"cells={sub.n_cells}  W={args.world_size}")

    A = affinity_matrix(sub.experts, t.num_experts, kind="cooccurrence")
    A = np.asarray(A, dtype=np.float64)
    print(f"   affinity matrix {A.shape}, "
          f"nonzero off-diagonal {int((A > 0).sum() - t.num_experts)}")

    g_value, g_map = greedy_score(A, args.world_size)
    print(f"\n   greedy clustering score : {g_value:,.1f}")

    out = solve_affinity_partition(
        A, args.world_size, top_pairs=args.top_pairs,
        time_limit=args.time_limit, mip_rel_gap=args.mip_rel_gap, verbose=True)
    def fmt(v, dash="none"):
        return dash if v is None else f"{v:,.1f}"

    print(f"   MILP status             : {out['status'][:70]}")
    print(f"   MILP value (pruned)     : "
          f"{'none (no incumbent found)' if out['value'] is None else fmt(out['value'])}")
    print(f"   MILP proven ceiling     : {fmt(out['upper_bound'])}")
    print(f"   pairs used / available  : {out['n_pairs_used']:,} / "
          f"{out['n_pairs_total']:,}")
    print(f"   wall                    : {out['wall_s']:.1f}s")

    if out["assignment"] is not None:
        s = affinity_score(A, out["assignment"])
        print(f"\n   MILP placement re-scored on the FULL graph: {s:,.1f}")
        print(f"   greedy                                    : {g_value:,.1f}")
        if g_value > 0:
            print(f"   difference                                : "
                  f"{100.0 * (s / g_value - 1.0):+.2f}% vs greedy")
        per_rank = np.bincount(out["assignment"], minlength=args.world_size)
        print(f"   experts per rank (must all be "
              f"{t.num_experts // args.world_size}): {np.unique(per_rank).tolist()}")

    print("\n   NOTE: this is a bound on a PRUNED instance, and it optimises the "
          "affinity proxy only.\n"
          "         Affinity → message counts → tier-aware bytes; the three rank "
          "placements differently.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toy", action="store_true",
                    help="tiny instance, verified against brute force")
    ap.add_argument("--workload", type=Path, default=None,
                    help="workload dir containing manifest.json")
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--top-pairs", type=int, default=400,
                    help="affinity pairs that get a variable (tractability lever)")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--mip-rel-gap", type=float, default=1e-4)
    args = ap.parse_args(argv)

    if args.toy:
        return run_toy(args)
    if args.workload is None:
        ap.error("pass --toy or --workload")
    return run_trace(args)


if __name__ == "__main__":
    raise SystemExit(main())
