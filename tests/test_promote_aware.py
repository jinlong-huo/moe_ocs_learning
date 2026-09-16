#!/usr/bin/env python3
"""Unit tests for the post-circuit placement objective and the MILP bounds.

Run:
    python3 -m unittest discover -s tests -v

Two of these tests encode bugs that were actually hit while building this:

  * ``test_critical_path_is_max_over_four_resources`` — a rank's NVLink drain
    and NIC drain are separate resources; adding them overstates the
    bottleneck exactly when the optimiser makes one rank hot on both, which is
    the case a placement search explores constantly.
  * ``test_oracle_matches_evaluate_asymmetric`` — the byte contraction
    ``bits.T @ oh`` silently returns ``[dst, src]`` instead of ``[src, dst]``,
    swapping the egress and ingress bottleneck.  It is invisible on any
    symmetric traffic pattern, so the test instance is deliberately asymmetric.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.eval.completion import (  # noqa: E402
    ServingModel, first_decode_step, prefill_pass, speedup, timing_report,
)
from src.eval.cost_model import (  # noqa: E402
    CostConfig, DispatchMode, Placement, Tier, Topology, evaluate, traffic_matrix,
)
from src.eval.milp_bound import (  # noqa: E402
    linear_minmax, true_ingress, union_minmax, union_minmax_milp,
)
from src.eval.ocs_eval import OcsConfig, plan_circuits, with_circuits  # noqa: E402
from src.eval.promote_aware import PostCircuitOracle, Drain  # noqa: E402
from src.eval.trace_ir import CellTable, RunInfo  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# A synthetic workload small enough to reason about by hand
# ═══════════════════════════════════════════════════════════════════════

def synth_table(rows: list[list[int]], srcs: list[int], *, num_experts: int,
                world_size: int, layer: int = 0) -> CellTable:
    """Build a CellTable whose cell->DP-rank ownership is exactly ``srcs``.

    ``token_rank`` maps a cell to ``(run * 1_000_003 + pos) % n_dp``, so with
    ``run = 0`` the source rank is simply ``pos % n_dp`` — which lets a test
    place traffic between specific ranks deliberately.
    """
    n = len(rows)
    experts = np.asarray(rows, dtype=np.int32)
    return CellTable(
        run=np.zeros(n, dtype=np.int32),
        layer=np.full(n, layer, dtype=np.int32),
        pos=np.asarray(srcs, dtype=np.int32),
        tok=np.full(n, -1, dtype=np.int32),
        phase=np.zeros(n, dtype=np.uint8),
        experts=experts,
        weights=np.full(experts.shape, 1.0 / experts.shape[1], dtype=np.float32),
        runs=[RunInfo(run_idx=0, uid="r0", category="t", group="t", role="category",
                      variant=0, model_id="synthetic", prompt_len=8,
                      generated_len=8, total_tokens=16)],
        num_experts=num_experts, top_k=int(experts.shape[1]),
        model_id="synthetic", layers=np.asarray([layer], dtype=np.int32),
    )


def two_pod_topo(world: int = 8) -> Topology:
    """8 ranks in 2 pods of 4 (2 GPUs/node, 2 nodes/pod): both tiers exist."""
    return Topology(world, gpus_per_node=2, nodes_per_pod=2)


def check_oracle(rows, srcs, mapping, topo, circuits, cost=None):
    """Oracle vs cost_model.evaluate on the same table, map and circuit set."""
    cost = cost or CostConfig(hidden_size=64)
    W = topo.world_size
    t = synth_table(rows, srcs, num_experts=len(mapping), world_size=W)
    lp = Placement(mapping, W, "global", name="m")
    orc = PostCircuitOracle(t.experts, t.num_experts, W,
                            np.asarray(srcs) % W, topo, circuits, cost, mult=2)
    got = orc.bottleneck_us(mapping)
    ref = evaluate(t, lp, with_circuits(topo, circuits), cost,
                   DispatchMode.DEDUP_RANK, seed=0)["bottleneck_us"]
    return got, ref, orc, t, lp


class TestOracleFidelity(unittest.TestCase):
    def test_oracle_matches_evaluate_symmetric(self):
        W, E, K = 8, 16, 2
        rng = np.random.default_rng(0)
        rows = [list(rng.choice(E, size=K, replace=False)) for _ in range(200)]
        srcs = list(rng.integers(0, W, size=200))
        mapping = (np.arange(E) // (E // W)).astype(np.int32)
        topo = two_pod_topo(W)
        tm = traffic_matrix(synth_table(rows, srcs, num_experts=E, world_size=W),
                            Placement(mapping, W, "global", name="m"), topo,
                            DispatchMode.DEDUP_RANK, seed=0)
        circ, info = plan_circuits(tm.counts, topo, OcsConfig(n_circuits=4))
        self.assertGreater(info["n_candidate_promotable_pairs"], 0,
                           "test instance must actually have promotable traffic")
        for circuits in (set(), circ):
            got, ref, *_ = check_oracle(rows, srcs, mapping, topo, circuits)
            self.assertAlmostEqual(got, ref, delta=abs(ref) * 1e-5 + 1e-6)

    def test_oracle_matches_evaluate_asymmetric(self):
        """Egress skew and ingress skew must land on different ranks.

        Rank 1 owns the experts that every other rank sends to (ingress hot);
        rank 6's tokens fan out to every rank (egress hot).  A transposed
        contraction swaps the two and the mismatch shows up immediately.
        """
        W, E, K = 8, 16, 2
        epr = E // W                                   # 2 experts per rank
        mapping = (np.arange(E) // epr).astype(np.int32)
        hot_ingress = [int(mapping.tolist().index(4))]  # expert living on rank 2
        rows, srcs = [], []
        for s in range(W):                              # everyone -> rank 2's expert
            for _ in range(5):
                rows.append([hot_ingress[0], (s * epr) % E]); srcs.append(s)
        for d in range(W):                              # rank 6 -> everyone
            e = int(np.flatnonzero(mapping == d)[0])
            rows.append([e, e]); srcs.append(6)
        topo = two_pod_topo(W)
        got, ref, orc, _t, _lp = check_oracle(rows, srcs, mapping, topo, set())
        self.assertAlmostEqual(got, ref, delta=abs(ref) * 1e-5 + 1e-6)
        d = orc.drain(mapping)
        eg = d.egress_nic + d.egress_nvl
        ing = d.ingress_nic + d.ingress_nvl
        self.assertNotAlmostEqual(float(np.argmax(eg)), float(np.argmax(ing)),
                                  msg="instance must have distinct egress/ingress hotspots")

    def test_critical_path_is_max_over_four_resources(self):
        """NIC and NVLink drains are independent: max, never sum."""
        W = 8
        byts = np.zeros((W, W))
        byts[0, 1] = 1e9          # intra-node, rank 0 egress on NVLink
        byts[0, 4] = 1e6          # cross-pod, rank 0 egress on the NIC
        topo = two_pod_topo(W)
        orc = PostCircuitOracle(np.zeros((1, 1), dtype=np.int32), 1, W,
                                np.zeros(1, dtype=np.int64), topo, set(),
                                CostConfig(hidden_size=64))
        d = orc.drain_of(byts)
        summed = max(float((d.egress_nic + d.egress_nvl).max()),
                     float((d.ingress_nic + d.ingress_nvl).max()))
        self.assertAlmostEqual(d.critical_path(),
                               max(float(d.egress_nic.max()), float(d.egress_nvl.max())),
                               places=6)
        self.assertGreater(summed, d.critical_path(),
                           "summing the two resources must be strictly worse here, "
                           "which is what made the bug visible")

    def test_oracle_accepts_no_promotable_regime(self):
        """A single-pod fabric has nothing to promote; the oracle must agree."""
        W, E = 8, 16
        rng = np.random.default_rng(1)
        rows = [list(rng.choice(E, size=2, replace=False)) for _ in range(50)]
        srcs = list(rng.integers(0, W, size=50))
        mapping = (np.arange(E) // (E // W)).astype(np.int32)
        topo = Topology(W, gpus_per_node=2, nodes_per_pod=4)
        tm = traffic_matrix(synth_table(rows, srcs, num_experts=E, world_size=W),
                            Placement(mapping, W, "global", name="m"), topo,
                            DispatchMode.DEDUP_RANK, seed=0)
        circ, info = plan_circuits(tm.counts, topo, OcsConfig(n_circuits=4))
        self.assertEqual(info["n_candidate_promotable_pairs"], 0)
        self.assertEqual(len(circ), 0)
        got, ref, *_ = check_oracle(rows, srcs, mapping, topo, circ)
        self.assertAlmostEqual(got, ref, delta=abs(ref) * 1e-5 + 1e-6)


# ═══════════════════════════════════════════════════════════════════════
# The co-design generator
# ═══════════════════════════════════════════════════════════════════════

class TestPromoteAwareGenerator(unittest.TestCase):
    def _fit(self, layers=(0, 1), n=240, E=16, K=2, seed=0):
        rng = np.random.default_rng(seed)
        run, lay, pos, exp = [], [], [], []
        for i in range(n):
            for l in layers:
                run.append(0); lay.append(l); pos.append(i % 8)
                # a planted affinity: low experts co-occur, so clustering matters
                base = int(rng.integers(0, E // 4)) * 4
                exp.append([base, base + int(rng.integers(1, 4))])
        experts = np.asarray(exp, dtype=np.int32)
        return CellTable(
            run=np.asarray(run, dtype=np.int32), layer=np.asarray(lay, dtype=np.int32),
            pos=np.asarray(pos, dtype=np.int32), tok=np.full(len(run), -1, np.int32),
            phase=np.zeros(len(run), dtype=np.uint8), experts=experts,
            weights=np.full(experts.shape, 0.5, dtype=np.float32),
            runs=[RunInfo(run_idx=0, uid="r0", category="t", group="t",
                          role="category", variant=0, model_id="synthetic",
                          prompt_len=8, generated_len=8, total_tokens=16)],
            num_experts=E, top_k=K, model_id="synthetic",
            layers=np.asarray(list(layers), dtype=np.int32))

    def test_generator_is_balanced_and_logged(self):
        from src.eval.promote_aware import promote_aware_placement
        fit = self._fit()
        W, E = 8, fit.num_experts
        topo = two_pod_topo(W)
        res = promote_aware_placement(fit, topo, OcsConfig(n_circuits=4), W,
                                      n_rounds=2, n_sweeps=1, n_cand=4,
                                      max_evals_per_layer=200, seed=0)
        p = res["placement"]
        counts = p.experts_per_rank_counts()
        self.assertTrue((counts == E // W).all(),
                        f"every rank must hold exactly {E // W} experts per layer, "
                        f"got {counts}")
        self.assertEqual(p.scope, "per_layer")
        self.assertEqual(p.expert_to_rank.shape, (fit.n_layers, E))
        self.assertEqual(len(res["history"]), 2)
        for h in res["history"]:
            self.assertIn("objective_us", h)
            self.assertLessEqual(h["evals_spent"], 200 * fit.n_layers)
        self.assertTrue(all(isinstance(c, frozenset) for c in res["circuits"]))

    def test_registered_in_make_placement(self):
        from src.eval.placement_opt import PLACEMENT_KINDS, make_placement
        self.assertIn("promote_aware_layer", PLACEMENT_KINDS)
        fit = self._fit(layers=(0,))
        W = 8
        p = make_placement("promote_aware_layer", fit, W, topo=two_pod_topo(W),
                           ocs_cfg=OcsConfig(n_circuits=4), n_sweeps=1,
                           promote_max_evals_per_layer=100)
        self.assertTrue(hasattr(p, "_promote_plan"))
        self.assertEqual(p.expert_to_rank.max() < W, True)

    def test_missing_topo_is_a_clear_error(self):
        from src.eval.placement_opt import make_placement
        with self.assertRaises(ValueError) as cm:
            make_placement("promote_aware_layer", self._fit(layers=(0,)), 8)
        self.assertIn("topo", str(cm.exception))


# ═══════════════════════════════════════════════════════════════════════
# MILP bounds
# ═══════════════════════════════════════════════════════════════════════

class TestMilpBounds(unittest.TestCase):
    def test_linear_minmax_known_optimum(self):
        """E=4, W=2, reach=[3,2,1,1]: the best pairing is {3,1},{2,1} -> 4."""
        r = linear_minmax(np.array([3.0, 2.0, 1.0, 1.0]), 2, relax=False,
                          time_limit=20.0, mip_rel_gap=1e-6)
        self.assertIsNotNone(r.objective)
        self.assertAlmostEqual(r.objective, 4.0, places=4)

    def test_lp_bound_always_available_and_never_above_the_mip(self):
        """The LP relaxation is the bound that always exists.

        A bound reported only when the MIP happens to find an incumbent is not a
        bound: on the real instance the union MIP returns neither an objective
        nor a dual bound.  The LP must be strictly usable and provably <= the
        integer optimum.
        """
        rng = np.random.default_rng(5)
        reach = rng.integers(1, 40, size=24).astype(np.float64)
        lp = linear_minmax(reach, 4, relax=True)
        mip = linear_minmax(reach, 4, relax=False, time_limit=20.0,
                            mip_rel_gap=1e-6)
        self.assertTrue(lp.relaxed)
        self.assertIsNotNone(lp.bound)
        self.assertLessEqual(lp.bound, mip.objective + 1e-6)

        experts = rng.integers(0, 24, size=(200, 2)).astype(np.int32)
        ulp = union_minmax(experts, 24, 4, relax=True, max_sets=16)
        self.assertIsNotNone(ulp.bound)

    def test_mip_without_incumbent_reports_no_bound(self):
        """Documenting the scipy behaviour the module works around."""
        rng = np.random.default_rng(7)
        experts = rng.integers(0, 256, size=(4000, 8)).astype(np.int32)
        r = union_minmax_milp(experts, 256, 32, max_sets=200, time_limit=1.0,
                              mip_rel_gap=1e-4)
        if r.assignment is None:
            self.assertIsNone(r.bound,
                              "a MIP with no incumbent must not offer a bound")

    def test_linear_milp_beats_lpt_or_matches_it(self):
        """LPT is feasible for the MILP, so the optimum can only be lower."""
        from src.eval.placement_opt import _lpt_map
        from src.eval.milp_bound import linearized_loads
        rng = np.random.default_rng(3)
        reach = rng.integers(1, 40, size=24).astype(np.float64)
        W = 4
        r = linear_minmax(reach, W, relax=False, time_limit=30.0, mip_rel_gap=1e-6)
        lpt = linearized_loads(_lpt_map(reach, W), reach).max()
        self.assertIsNotNone(r.objective)
        self.assertLessEqual(r.objective, lpt + 1e-6)

    def test_true_ingress_uses_union_semantics(self):
        """A cell selecting two experts on one rank sends ONE message there."""
        experts = np.array([[0, 1], [0, 2]], dtype=np.int32)
        m = np.zeros(4, dtype=np.int32)     # experts 0,1,2 all on rank 0
        self.assertEqual(true_ingress(experts, m, 4).tolist(), [2, 0, 0, 0])
        # The linearised count credits each expert's whole reach, so cell 0 is
        # counted twice (once via expert 0, once via expert 1) and cell 1 twice:
        # 4 <-> 2 true messages.  That factor is why the module reports both
        # objectives and never quotes a linear-bound gap as if it were the real
        # ingress gap.
        from src.eval.milp_bound import linearized_loads
        reach = np.bincount(experts.ravel(), minlength=4).astype(float)
        self.assertEqual(linearized_loads(m, reach)[0], 4.0)
        self.assertGreater(linearized_loads(m, reach)[0],
                           true_ingress(experts, m, 4)[0])

    def test_union_milp_runs_and_reports_coverage(self):
        experts = np.array([[0, 1], [0, 1], [2, 3], [1, 3]], dtype=np.int32)
        r = union_minmax_milp(experts, 4, 2, max_sets=8, time_limit=30.0,
                              mip_rel_gap=1e-3)
        self.assertIsNotNone(r.objective)
        self.assertIn("heaviest distinct expert sets", r.note)
        self.assertIsNotNone(r.assignment)
        # the union objective can never exceed the linearised (double counting)
        # one for the same assignment
        lin = true_ingress(experts, r.assignment, 2).max()
        self.assertLessEqual(lin, r.objective * (1 + 1e-3) + 1e-9)

    def test_bound_respects_the_averaging_argument(self):
        """`sum_r load_r == sum_e reach_e` for every feasible x, so

            min_x max_r load_r  >=  mean_e reach_e

        This is the cheapest possible sanity check on the formulation wiring, and
        it catches exactly one failure mode: a coefficient array whose order does
        not match the column order, which silently *relaxes* the problem and
        reports a bound below the averaging floor.  (The real instance reported
        93 against a floor of 512 before this test existed.)
        """
        rng = np.random.default_rng(13)
        reach = rng.integers(1, 60, size=32).astype(np.float64)
        W = 4
        floor = reach.sum() / W
        for relax in (True, False):
            r = linear_minmax(reach, W, relax=relax, time_limit=20.0,
                              mip_rel_gap=1e-6)
            self.assertIsNotNone(r.bound if relax else r.objective)
            got = r.bound if relax else r.objective
            self.assertGreaterEqual(got, floor - 1e-6,
                                    f"bound {got} below the averaging floor {floor}")

    def test_combinatorial_union_bound_is_a_lower_bound(self):
        """The bound must hold for EVERY placement, not just the ones tested."""
        from src.eval.milp_bound import combinatorial_union_bound
        rng = np.random.default_rng(11)
        E, W, K, N = 24, 4, 3, 200
        experts = rng.integers(0, E, size=(N, K)).astype(np.int32)
        b = combinatorial_union_bound(experts, E, W)
        self.assertGreater(b["bound"], 0)
        # perfectly-spread ideal: ceil(K/cap) * N / W with cap = E/W = 6 -> 1*200/4
        self.assertAlmostEqual(b["perfectly_spread_ideal"], N / W)
        for seed in range(5):
            keep = rng.permutation(E)
            m = np.empty(E, dtype=np.int32)
            m[keep] = np.arange(E) // (E // W)     # any balanced placement
            self.assertGreaterEqual(true_ingress(experts, m, W).max(),
                                    int(np.floor(b["bound"])))

    def test_expert_reach_dominates_when_it_is_the_binding_term(self):
        """A single very popular expert forces ingress on whichever rank holds it."""
        from src.eval.milp_bound import combinatorial_union_bound
        E, W, K = 8, 2, 2
        # expert 0 is selected by 90 of 100 cells; the rest are spread
        rows = [[0, 1]] * 90 + [[2, 3], [4, 5], [6, 7]] * 3 + [[0, 2]] * 7
        experts = np.asarray(rows, dtype=np.int32)
        b = combinatorial_union_bound(experts, E, W)
        self.assertEqual(b["largest_expert_reach"], 97)
        self.assertGreaterEqual(b["bound"], 97)
        for m in (np.array([0, 0, 0, 0, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 0, 0, 0, 0])):
            self.assertGreaterEqual(true_ingress(experts, m, W).max(), int(np.floor(b["bound"])))


# ═══════════════════════════════════════════════════════════════════════
# Completion-time composition
# ═══════════════════════════════════════════════════════════════════════

class TestCompletion(unittest.TestCase):
    def _two_phase_table(self):
        n_run, layers = 3, 2
        run, lay, pos, phase = [], [], [], []
        for r in range(n_run):
            for l in range(layers):
                for p in range(4):                       # 4 prefill tokens
                    run.append(r); lay.append(l); pos.append(p); phase.append(0)
                run.append(r); lay.append(l); pos.append(4); phase.append(1)
        experts = np.tile(np.array([[0, 1]]), (len(run), 1)).astype(np.int32)
        return CellTable(
            run=np.asarray(run, np.int32), layer=np.asarray(lay, np.int32),
            pos=np.asarray(pos, np.int32), tok=np.full(len(run), -1, np.int32),
            phase=np.asarray(phase, np.uint8), experts=experts,
            weights=np.full(experts.shape, 0.5, np.float32),
            runs=[RunInfo(run_idx=i, uid=f"r{i}", category="t", group="t",
                          role="category", variant=0, model_id="s",
                          prompt_len=4, generated_len=1, total_tokens=5)
                  for i in range(n_run)],
            num_experts=8, top_k=2, model_id="s", layers=np.asarray([0, 1], np.int32))

    def test_slices_split_prefill_and_one_decode_step(self):
        t = self._two_phase_table()
        self.assertEqual(prefill_pass(t).n_cells, 3 * 2 * 4)
        step = first_decode_step(t)
        self.assertEqual(step.n_cells, 3 * 2)          # one position per run per layer
        self.assertEqual(set(np.unique(step.pos).tolist()), {4})

    def test_timing_arithmetic(self):
        t = self._two_phase_table()
        m = ServingModel(compute_us_per_layer=10.0, reconfig_us=1000.0,
                         passes_per_reconfig=100)
        r = timing_report(500.0, 100.0, t, m)
        self.assertAlmostEqual(r["compute_us"], 20.0)          # 10 * 2 layers
        self.assertAlmostEqual(r["reconfig_per_pass_us"], 10.0)
        self.assertAlmostEqual(r["ttft_us"], 530.0)
        self.assertAlmostEqual(r["itl_us"], 130.0)
        self.assertAlmostEqual(r["throughput_tok_s"], 3 / 130e-6, places=1)

    def test_comm_only_reports_comm_share_one(self):
        t = self._two_phase_table()
        r = timing_report(500.0, 100.0, t, ServingModel())
        self.assertAlmostEqual(r["comm_share_of_itl"], 1.0)
        s = speedup(r, {**r, "itl_us": 50.0, "ttft_us": 250.0})
        self.assertAlmostEqual(s["itl_reduction_pct"], 50.0)
        self.assertAlmostEqual(s["ttft_reduction_pct"], 50.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
