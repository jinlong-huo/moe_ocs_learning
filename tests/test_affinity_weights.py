#!/usr/bin/env python3
"""Unit tests for the weight-aware affinity metrics (src/serving/affinity.py).

Run:
    .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.routing_schema import (  # noqa: E402
    LayerRoute,
    RoutingTrace,
    RunMeta,
    TokenRoute,
)
from src.serving.affinity import (  # noqa: E402
    _cell_dist,
    load_repeats,
    pairwise_metrics,
    repeat_noise_floor,
    z_score,
)

E = 8
LAYERS = ["3", "5"]


def make_trace(cells: dict, *, num_experts: int = E, top_k: int = 2,
               prompt_len: int = 4, gen_len: int = 2,
               backend: str = "test") -> RoutingTrace:
    """Build a synthetic RoutingTrace from {(pos, layer): (experts, weights)}."""
    total = prompt_len + gen_len
    token_ids = list(range(100, 100 + total))
    routes = []
    for (pos, layer), (experts, weights) in sorted(cells.items()):
        routes.append(TokenRoute(
            token_pos=pos,
            token_id=token_ids[pos],
            token_str="x",
            phase="prefill" if pos < prompt_len else "decode",
            layers={str(layer): LayerRoute(experts=list(experts),
                                           weights=list(weights))},
        ))
    return RoutingTrace(
        meta=RunMeta(
            model_id="synthetic", model_type="synthetic", num_layers=8,
            num_moe_layers=2, num_experts=num_experts, top_k=top_k,
            prompt_len=prompt_len, generated_len=gen_len, total_tokens=total,
            backend=backend,
        ),
        prompt_tokens=token_ids[:prompt_len],
        generated_tokens=token_ids[prompt_len:],
        routes=routes,
    )


def std_cells(experts=(0, 3), weights=(0.7, 0.2)) -> dict:
    return {
        (pos, lid): (list(experts), list(weights))
        for pos in range(6)
        for lid in (3, 5)
    }


class TestCellDist(unittest.TestCase):
    def test_basic_mass_and_residual(self):
        p = _cell_dist([0, 3], [0.6, 0.3], E)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=9)
        self.assertAlmostEqual(p[0], 0.6, places=9)
        self.assertAlmostEqual(p[3], 0.3, places=9)
        residual = 0.1 / (E - 2)
        for e in (1, 2, 4, 5, 6, 7):
            self.assertAlmostEqual(p[e], residual, places=9)

    def test_renormalized_weights(self):
        # norm_topk_prob traces: masses sum above 1 → normalise to support.
        p = _cell_dist([2, 6], [0.6, 0.6], E)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=9)
        self.assertAlmostEqual(p[2], 0.5, places=9)
        self.assertAlmostEqual(p[6], 0.5, places=9)
        self.assertAlmostEqual(p[0], 0.0, places=9)

    def test_full_support(self):
        # All experts selected: no residual target, mass renormalises.
        p = _cell_dist([0, 1], [0.7, 0.2], 2)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=9)
        self.assertAlmostEqual(p[0], 0.7 / 0.9, places=9)
        self.assertAlmostEqual(p[1], 0.2 / 0.9, places=9)

    def test_zero_mass(self):
        p = _cell_dist([0, 3], [0.0, 0.0], E)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=9)
        self.assertTrue(float(p.std()) < 1e-12)  # uniform fallback


class TestWeightAwareMetrics(unittest.TestCase):
    def pm(self, ta, tb, **kw):
        return pairwise_metrics(ta, tb, E, LAYERS, 2, **kw)

    def test_identical_traces(self):
        t = make_trace(std_cells())
        m = self.pm(t, t, weight_aware=True)
        self.assertEqual(m["topk_overlap"], 1.0)
        self.assertEqual(m["mean_cell_mass_intersection"], 1.0)
        self.assertEqual(m["mean_cell_emd"], 0.0)
        self.assertEqual(m["mean_cell_bhattacharyya"], 1.0)
        self.assertEqual(m["matched_weight_mae"], 0.0)
        self.assertEqual(m["matched_weight_cosine"], 1.0)
        self.assertEqual(m["matched_cells"], m["cells_common"])

    def test_marginal_flip_is_cheap(self):
        # The k-th expert (mass 0.02) differs; set metrics charge a full miss,
        # mass metrics charge only the flipped mass.
        ta = make_trace(std_cells(experts=(0, 3), weights=(0.7, 0.02)))
        tb = make_trace(std_cells(experts=(0, 5), weights=(0.7, 0.02)))
        m = self.pm(ta, tb, weight_aware=True)
        self.assertEqual(m["topk_overlap"], 0.0)
        self.assertGreaterEqual(m["mean_cell_mass_intersection"], 0.95)
        self.assertLess(m["mean_cell_emd"], 0.1)
        self.assertEqual(m["matched_cells"], 0)
        self.assertIsNone(m["matched_weight_mae"])

    def test_same_set_different_emphasis(self):
        ta = make_trace(std_cells(experts=(0, 3), weights=(0.8, 0.1)))
        tb = make_trace(std_cells(experts=(3, 0), weights=(0.45, 0.45)))
        m = self.pm(ta, tb, weight_aware=True)
        # Set identity is perfect ...
        self.assertEqual(m["topk_overlap"], 1.0)
        self.assertEqual(m["matched_cells"], m["cells_common"])
        # ... but the emphasis differs, and the fidelity metrics see it.
        self.assertGreater(m["matched_weight_mae"], 0.1)
        self.assertLess(m["matched_weight_cosine"], 0.95)
        self.assertLess(m["mean_cell_mass_intersection"], 0.9)

    def test_disjoint_top1(self):
        ta = make_trace(std_cells(experts=(0,), weights=(0.9,)), top_k=1)
        tb = make_trace(std_cells(experts=(1,), weights=(0.9,)), top_k=1)
        m = self.pm(ta, tb)
        m = pairwise_metrics(ta, tb, E, LAYERS, 1, weight_aware=True)
        self.assertEqual(m["topk_overlap"], 0.0)
        self.assertLess(m["mean_cell_mass_intersection"], 0.3)
        self.assertGreater(m["mean_cell_emd"], 0.5)

    def test_flag_off_by_default(self):
        ta = make_trace(std_cells())
        tb = make_trace(std_cells())
        m = self.pm(ta, tb)
        self.assertNotIn("mean_cell_mass_intersection", m)
        m2 = self.pm(ta, tb, weight_aware=True)
        self.assertIn("mean_cell_mass_intersection", m2)
        self.assertIn("matched_weight_mae", m2)


class TestRepeatNoiseFloor(unittest.TestCase):
    def test_floor_from_repeats(self):
        t1 = make_trace(std_cells(experts=(0, 3), weights=(0.7, 0.02)))
        t2 = make_trace(std_cells(experts=(0, 3), weights=(0.7, 0.02)))
        cells = std_cells(experts=(0, 3), weights=(0.7, 0.02))
        cells[(0, 3)] = ([0, 5], [0.7, 0.02])  # one noisy cell
        t3 = make_trace(cells)

        floor = repeat_noise_floor([t1, t2, t3], E, 2)
        self.assertEqual(floor["n_traces"], 3)
        self.assertEqual(floor["n_pairs"], 3)
        mi = floor["metrics"]["mean_cell_mass_intersection"]
        self.assertGreaterEqual(mi["mean"], 0.95)
        self.assertLess(mi["sd"], 0.05)
        self.assertIn("mean_cell_emd", floor["metrics"])
        self.assertIn("topk_overlap", floor["metrics"])

    def test_too_few_traces(self):
        floor = repeat_noise_floor([make_trace(std_cells())], E, 2)
        self.assertEqual(floor["n_pairs"], 0)
        self.assertEqual(floor["metrics"], {})

    def test_load_repeats_filters_by_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "traces").mkdir()
            records = []
            for i, role in enumerate(("repeat", "repeat", "category")):
                t = make_trace(std_cells())
                t.save(d / "traces" / f"t{i}.json")
                records.append({"uid": f"u{i}", "role": role,
                                "trace": f"traces/t{i}.json"})
            with open(d / "manifest.json", "w") as f:
                json.dump({"records": records}, f)
            reps = load_repeats(d)
            self.assertEqual(len(reps), 2)
            self.assertTrue(all(r.meta.backend == "test" for r in reps))


class TestZScore(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(z_score(0.99, {"mean": 0.97, "sd": 0.01}), 2.0)

    def test_zero_sd_is_none(self):
        self.assertIsNone(z_score(0.98, {"mean": 0.97, "sd": 0.0}))


if __name__ == "__main__":
    unittest.main()
