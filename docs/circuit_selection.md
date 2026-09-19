# Phase 6 — exact circuit selection: the solver agrees with the greedy

`src/eval/bmatching.py`, `scripts/bmatching_vs_greedy.py`, results in
`outputs/affinity/bmatching_vs_greedy.qwen36.json`.

## The claim being tested

`docs/ocs_moe_program.md` V11 states that circuit selection is **exactly
solvable**, because the degree-bounded b-matching polytope is integral and so the
LP relaxation is exact. Two things make that non-obvious here:

* the rank graph is the **complete graph**, not bipartite, and integrality of the
  degree-constrained subgraph polytope is a bipartite result;
* the plan also carries a **cardinality constraint** (`sum z <= n_circuits`),
  which is not part of that matroid structure.

So it was measured rather than inherited: `linprog` (LP) and `milp` (integer) are
both implemented on identical semantics, and the LP solution is checked for
fractional edges.

## Result

| source | placement | method | covered | gain % | circuits | fractional edges |
| --- | --- | --- | --- | --- | --- | --- |
| hash | linear | greedy | 0.0690 | 4.755 | 16 | — |
| hash | linear | LP | 0.0691 | 4.755 | 16 | **0** |
| hash | linear | MILP | 0.0691 | 4.755 | 16 | — |
| hash | affinity_coord | greedy | 0.0664 | 2.528 | 16 | — |
| hash | affinity_coord | LP | 0.0664 | 2.528 | 16 | **0** |
| measured_packed_4 | linear | greedy | 0.1399 | 9.462 | 8 | — |
| measured_packed_4 | linear | LP | 0.1404 | 8.824 | 8 | **0** |
| measured_packed_4 | affinity_coord | greedy | 0.1298 | 8.669 | 8 | — |
| measured_packed_4 | affinity_coord | LP | 0.1298 | 8.669 | 8 | **0** |

Three findings:

1. **V11's integrality claim holds empirically** — zero fractional edges on all four
   instances, and the LP objective equals the MILP objective exactly (e.g. 20608.0
   both ways). The complete graph and the cardinality constraint did not break it.
   Caveat: *these instances*, so it is evidence, not a proof.
2. **The greedy leaves almost nothing on the table.** Best case, the exact solver
   improves coverage by 0.1404 / 0.1399 = **+0.36 %**; worst case 0.14 %. The
   ½-approximation's worst-case bound is not realised here, which is why greedy is
   the right deployed choice.
3. **Better coverage is not better time.** In the `measured_packed_4 | linear` cell
   the exact solver covers *more* traffic (0.1404 vs 0.1399) and yet delivers a
   *worse* OCS gain (8.824 % vs 9.462 %). Maximising covered traffic is a proxy for
   the bottleneck, not the bottleneck: a plan can add a circuit on a pair that does
   not bind while the binding pair loses one.

## What this changes

* **Circuit selection is not the bottleneck of this program.** The exact solver is
  available, cheap and confirms the deployed greedy — so the remaining headroom is
  not in the planner. Per `docs/ownership_measurement.md`, it is in **plan-window
  stability** (Jaccard 0.0000 for `per_sequence`, 1.0000 for measured-spread).
* The false oracle noted in `ocs_moe_program.md` (`value_of_prediction
  = −0.099 %`) is now explained: the "oracle" is greedy, and greedy is within 0.4 %
  of exact — so the oracle was never really an oracle, it was the same planner with
  hindsight.
