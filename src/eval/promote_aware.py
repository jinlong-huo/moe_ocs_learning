"""
promote_aware.py — placement optimised against the bottleneck that REMAINS
after the optical circuit plan has been applied.

Why this module exists
──────────────────────
Every placement generator in ``placement_opt.py`` optimises an *electrical*
objective: fan-out, dedup ingress, or a load cap.  ``ocs_eval.plan_circuits``
then fits circuits to whatever traffic those placements happened to produce.
That ordering is sequential, and it structurally cannot test the co-design
claim, because the placement never sees the circuit plan it will be paired
with:

    sequential   placement* = argmin_p  cost(p, EPS)
                 plan*      = argmin_c  cost(placement*, c)
                 -> the pair (placement*, plan*) is never jointly selected

    this module  placement* = argmin_p  min_c  cost(p, c)
                 -> for each p the circuit plan is re-derived, so placement is
                    scored on the residual it leaves for the plan.

The two orderings can genuinely disagree, and the disagreement is the point of
the experiment: affinity placement works by *coalescing destinations* (fewer
distinct destination ranks per token), which removes exactly the cross-pod rank
pairs a circuit would otherwise have promoted.  Improving the placement can
therefore *shrink* the circuit's gain — affinity and OCS may be substitutes
rather than complements.  An objective that optimises the residual after
promotion is the only way to find out which.

Why alternating, not joint
──────────────────────────
``plan_circuits`` is a greedy degree-bounded b-matching over edge weights, and
those weights are themselves a function of the placement.  Solving both at once
is a hard combinatorial problem (the QAP the Affinity note maps onto Aurora's
provably NP-hard case).  What is tractable, and what a real controller would
actually run, is **alternating optimisation**:

    1. init placement            (the best electrical placement we have)
    2. plan circuits from it     (greedy b-matching on the aggregate fit slice)
    3. re-place each layer against the residual under that plan
    4. re-plan, repeat

Each half is cheap; the loop is the co-design.  The iteration is logged so the
report can show whether it converged or oscillated.

Cost-model fidelity
───────────────────
``PostCircuitOracle`` reproduces ``cost_model.evaluate``'s bottleneck term
exactly for a fixed placement and circuit set — checked in
``tests/test_promote_aware.py``, not asserted here.  Two details of that model
are easy to get wrong and are load-bearing:

  * A rank's NVLink drain and its NIC drain are **separate resources**, so the
    critical path is the max over *four* accumulated vectors
    (egress/ingress x NIC/NVLink), never the sum of a rank's NVLink and NIC
    traffic.  Adding them overstates the bottleneck exactly when the same rank
    is hot on both, which is precisely the case an optimiser explores.
  * The collective's critical path is  max over ranks of  SUM over layers
    drain_l(rank) , which is why each layer is optimised against a running
    accumulator rather than layer by layer in isolation.

Destination sets are encoded as one ``uint64`` bitmask per routing cell, so an
evaluation is a couple of vectorised ops instead of an O(K) sort per cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.eval.cost_model import (
    CostConfig, DispatchMode, FabricConfig, Placement, Tier, Topology,
    traffic_matrix,
)
from src.eval.ocs_eval import OcsConfig, plan_circuits, with_circuits
from src.eval.trace_ir import CellTable


# ═══════════════════════════════════════════════════════════════════════
# The four-resource drain
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Drain:
    """Per-rank drain (us) split by resource and direction.

    NIC and NVLink are independent resources per rank, so the critical path is
    ``max`` over these four vectors — the same decomposition
    ``cost_model.evaluate`` performs.
    """

    egress_nic: np.ndarray
    ingress_nic: np.ndarray
    egress_nvl: np.ndarray
    ingress_nvl: np.ndarray

    @staticmethod
    def zeros(w: int) -> "Drain":
        z = np.zeros(w, dtype=np.float64)
        return Drain(z.copy(), z.copy(), z.copy(), z.copy())

    def vectors(self) -> tuple[np.ndarray, ...]:
        return (self.egress_nic, self.ingress_nic, self.egress_nvl, self.ingress_nvl)

    def __add__(self, other: "Drain") -> "Drain":
        return Drain(*(a + b for a, b in zip(self.vectors(), other.vectors())))

    def critical_path(self) -> float:
        """max over ranks and resources (us); the quantity to minimise."""
        return float(max(v.max(initial=0.0) for v in self.vectors()))


@dataclass
class EvalBudget:
    """Hard cap on objective evaluations, so a search cannot run away.

    The objective is a matrix product per candidate swap, so the budget — not
    the sweep count — is what bounds wall-clock time.  ``exhausted`` is
    reported rather than raised: a search that hits its budget still returns a
    valid, strictly improving placement.
    """
    max_evals: int = 4000
    n_evals: int = 0
    exhausted: bool = False

    def spend(self) -> bool:
        if self.n_evals >= self.max_evals:
            self.exhausted = True
            return False
        self.n_evals += 1
        return True


class PostCircuitOracle:
    """Per-layer drain under a FIXED circuit set, with a cross-layer accumulator.

    One instance serves one MoE layer.  Tiers are fixed for the topology, so the
    two inverse-bandwidth matrices are built once; only the destination sets
    change as the placement is searched.
    """

    __slots__ = ("experts", "E", "W", "n_cells", "src", "ohT", "shifts",
                 "inv_nic", "inv_nvl", "B", "alpha_us", "_acc", "budget",
                 "circuits", "mult")

    def __init__(self, experts: np.ndarray, num_experts: int, world_size: int,
                 src: np.ndarray, topo: Topology, circuits: set[frozenset],
                 cost: CostConfig, budget: EvalBudget | None = None,
                 mult: int = 1):
        if world_size > 64:
            raise ValueError(
                f"PostCircuitOracle encodes destination sets in one uint64 "
                f"bitmask, so world_size <= 64 is required (got {world_size})")
        n = int(experts.shape[0])
        self.experts, self.E, self.W, self.n_cells = experts, num_experts, world_size, n
        self.circuits = circuits
        self.budget = budget or EvalBudget()
        self.mult = mult

        # ── source rank of every cell: fixed by the workload, not by placement
        self.src = np.asarray(src, dtype=np.int64)
        oh = np.zeros((n, world_size), dtype=np.float32)
        oh[np.arange(n), self.src] = 1.0
        self.ohT = np.ascontiguousarray(oh.T)
        self.shifts = np.arange(world_size, dtype=np.uint64)

        # ── per-pair inverse bandwidth, with the circuit set applied
        t = with_circuits(topo, circuits)
        T = t.tier_matrix()
        fab: FabricConfig = t.fabric
        inv_nic = np.zeros((world_size, world_size), dtype=np.float32)
        inv_nvl = np.zeros((world_size, world_size), dtype=np.float32)
        nvl = (T == int(Tier.INTRA_NODE))
        for i in range(4):
            m = (T == i) & ~nvl
            if m.any():
                inv_nic[m] = np.float32(1.0 / fab.bandwidth(Tier(i)))
        inv_nvl[nvl] = np.float32(1.0 / fab.intra_node_gbytes_per_s)
        np.fill_diagonal(inv_nic, 0.0)          # a rank does not send to itself
        np.fill_diagonal(inv_nvl, 0.0)
        self.inv_nic, self.inv_nvl = inv_nic, inv_nvl

        self.B = float(cost.hidden_size * cost.dtype_bytes)
        # alpha is a constant offset for a fixed topology, so it cannot change a
        # swap comparison; it is added back only when reporting a comparable
        # bottleneck.
        off = ~np.eye(world_size, dtype=bool)
        self.alpha_us = (float(fab.latency_us(Tier(int(T[off].max()))))
                         if off.any() else 0.0)
        self._acc = Drain.zeros(world_size)

    # ── the drain itself ─────────────────────────────────────────────
    def pair_bytes(self, p: np.ndarray) -> np.ndarray:
        """[W, W] dispatch bytes per (src rank, dst rank) for this layer.

        The contraction order matters: ``(ohT @ bits)[s, r]`` counts cells owned
        by ``s`` that reach ``r``, i.e. ``counts[src, dst]``.  The mirror
        product ``bits.T @ oh`` yields the transpose, which silently swaps the
        egress and ingress bottleneck.
        """
        dst = p[self.experts].astype(np.uint64)                    # [N, K]
        mask = np.bitwise_or.reduce(np.uint64(1) << dst, axis=1)   # [N]
        bits = ((mask[:, None] >> self.shifts[None, :]) & np.uint64(1))
        counts = self.ohT @ bits.astype(np.float32)                # [W, W]
        return counts * self.B

    def drain(self, p: np.ndarray) -> Drain:
        return self.drain_of(self.pair_bytes(p))

    def drain_of(self, byts: np.ndarray) -> Drain:
        """Split bytes into the four per-rank resources (us).

        bytes / (GB/s) = ns, hence the 1e-3 to microseconds.
        """
        d_nic = byts * self.inv_nic * 1e-3
        d_nvl = byts * self.inv_nvl * 1e-3
        return Drain(d_nic.sum(1), d_nic.sum(0), d_nvl.sum(1), d_nvl.sum(0))

    def set_accumulator(self, acc: Drain) -> None:
        self._acc = Drain(*(np.asarray(v, dtype=np.float64).copy()
                            for v in acc.vectors()))

    def objective(self, p: np.ndarray) -> float:
        """Accumulated critical path (us) — what the optimiser minimises."""
        if not self.budget.spend():
            return np.inf
        return (self._acc + self.drain(p)).critical_path()

    # ── reporting ────────────────────────────────────────────────────
    def bottleneck_us(self, p: np.ndarray, include_alpha: bool = True) -> float:
        """Comparable to ``cost_model.evaluate(...)['bottleneck_us']``."""
        v = self.drain(p).critical_path()
        return float((self.alpha_us + v) * self.mult) if include_alpha \
            else float(v * self.mult)


# ═══════════════════════════════════════════════════════════════════════
# Alternating placement <-> plan
# ═══════════════════════════════════════════════════════════════════════

def _layer_src(t: CellTable, layer: int, n_dp: int, seed: int = 0) -> np.ndarray:
    """The DP-rank owner of every cell of one layer (same rule as cost_model)."""
    from src.eval.cost_model import token_rank
    return token_rank(t.by_layer(layer), n_dp, seed)


def _layer_experts(t: CellTable, layer: int) -> np.ndarray:
    return t.experts[t.layer == layer]


def _swap_sweep(init: np.ndarray, rng: np.random.Generator,
                oracle: PostCircuitOracle, n_cand: int) -> np.ndarray:
    """One swap sweep on the post-circuit objective (balance preserved)."""
    p = init.copy()
    best = oracle.objective(p)
    E = p.shape[0]
    for a in rng.permutation(E):
        if oracle.budget.exhausted:
            break
        cand = rng.choice(E, size=min(E, n_cand), replace=False)
        for b in cand:
            if p[a] == p[b]:
                continue
            p[a], p[b] = p[b], p[a]
            v = oracle.objective(p)
            if v < best - 1e-12:
                best = v
            else:
                p[a], p[b] = p[b], p[a]
    return p


def promote_aware_placement(
    fit: CellTable,
    topo: Topology,
    ocs_cfg: OcsConfig,
    world_size: int,
    *,
    init_kind: str = "affinity_coordinated_layer",
    n_rounds: int = 2,
    n_sweeps: int = 2,
    n_cand: int = 8,
    max_evals_per_layer: int = 3000,
    cost: CostConfig | None = None,
    mode: DispatchMode = DispatchMode.DEDUP_RANK,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Alternating optimisation of (placement, circuit plan).

    Returns a dict rather than a bare ``Placement`` because the result of this
    generator is a *pair*: the placement and the plan it was co-designed with.
    ``history`` records each round's plan size and objective so a report can
    show convergence instead of asserting it.
    """
    from src.eval.placement_opt import make_placement

    cost = cost or CostConfig()
    E, W = fit.num_experts, world_size
    if E % W:
        raise ValueError(f"E={E} not divisible by W={W}")
    rng = np.random.default_rng(seed)
    layers = [int(l) for l in fit.layers]
    mult = (2 if cost.include_combine else 1) * cost.n_microbatches

    lx = {l: _layer_experts(fit, l) for l in layers}
    lsrc = {l: _layer_src(fit, l, W, seed) for l in layers}

    placement = make_placement(init_kind, fit, world_size, seed=seed)
    rows = {l: np.asarray(placement.expert_to_rank[i]).copy()
            for i, l in enumerate(layers)}
    history: list[dict] = []
    circuits: set[frozenset] = set()
    pinfo: dict = {}
    # Alternating optimisation is not monotone: round r+1 re-plans against a
    # placement whose traffic changed, so its objective is measured against a
    # *different* circuit set and can be worse than round r's.  The best
    # (placement, plan) pair seen is therefore kept and returned, which makes
    # the generator's result monotonically at least as good as its own
    # initialisation — otherwise a failed round reads as a co-design failure
    # when it is only a failed iteration.
    best_obj = float("inf")
    best_rows: dict[int, np.ndarray] | None = None
    best_circuits: set[frozenset] = set()
    best_round = -1

    for rnd in range(n_rounds):
        # ── 2. plan circuits from the current placement (aggregate fit slice)
        agg = Placement(np.stack([rows[l] for l in layers]), W, "per_layer",
                        layers=fit.layers, name="promote_aware.probe")
        tm = traffic_matrix(fit, agg, topo, mode, n_dp=W, seed=seed)
        circuits, pinfo = plan_circuits(tm.counts, topo, ocs_cfg)

        # ── 3. re-place every layer against the residual under that plan
        # Each layer gets its own budget: a single global cap would be spent by
        # the first layers and leave the rest unoptimised, which reads as "the
        # objective does not help" when it is only starvation.
        exhausted_layers: list[int] = []
        acc = Drain.zeros(W)
        evals_spent = 0
        for l in layers:
            budget = EvalBudget(max_evals=max_evals_per_layer)
            orc = PostCircuitOracle(lx[l], E, W, lsrc[l], topo, circuits, cost,
                                    budget, mult)
            orc.set_accumulator(acc)
            p = rows[l]
            for _ in range(n_sweeps):
                if budget.exhausted:
                    break
                p = _swap_sweep(p, rng, orc, n_cand)
            rows[l] = p
            acc = acc + orc.drain(p)
            evals_spent += budget.n_evals
            if budget.exhausted:
                exhausted_layers.append(l)
        obj = acc.critical_path()
        if obj < best_obj - 1e-9:
            best_obj = obj
            best_rows = {l: rows[l].copy() for l in layers}
            best_circuits = set(circuits)
            best_round = rnd
        history.append({
            "round": rnd,
            "n_circuits": len(circuits),
            "covered_fraction": float(pinfo["promotable_traffic_covered_fraction"]),
            "n_candidate_promotable_pairs": pinfo["n_candidate_promotable_pairs"],
            "objective_us": round(obj * mult, 4),
            "evals_spent": evals_spent,
            "layers_hitting_budget": len(exhausted_layers),
            "n_layers": len(layers),
            "kept": bool(best_round == rnd),
        })
        if verbose:
            print(f"  [promote_aware] round {rnd}: circuits={len(circuits)} "
                  f"obj={obj * mult:.1f}us evals={evals_spent} "
                  f"budget_hit={len(exhausted_layers)}/{len(layers)}"
                  f"{'  <- kept' if best_round == rnd else ''}")

    if best_rows is not None:
        rows, circuits = best_rows, best_circuits
    out = Placement(np.stack([rows[l] for l in layers]), W, "per_layer",
                    layers=fit.layers, name="promote_aware_layer")
    if out.expert_to_rank.min() < 0 or out.expert_to_rank.max() >= W:
        raise ValueError("promote_aware_layer produced out-of-range rank ids")
    return {"placement": out, "circuits": circuits, "history": history,
            "plan_info": pinfo, "best_round": best_round,
            "best_objective_us": round(best_obj * mult, 4)}
