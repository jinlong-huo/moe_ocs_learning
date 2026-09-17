# Plan — making OCS pay for MoE inference (three papers × this repo)

Status: plan only, no code changed. Replaces the earlier "certification surface"
draft: the objective is now **a positive OCS result**, and the applicability
boundary is demoted to scope, not headline. Companions:
`docs/research_assessment.md`, `docs/proposal_harvest_schedule.md`,
`docs/optitrainsim_final_comment.md`.
Papers: **Harvest** (2026), **TE-CCL** (SIGCOMM'24), **OptiTrainSim** (2026).

---

## 0. The one equation this plan maximises

The cost model is explicit: a circuit may replace a tier **only if that tier is
oversubscribed**, and `Tier.OPTICAL` bandwidth is the **NIC rate**
(`src/eval/cost_model.py`, `FabricConfig.bandwidth`). So an OCS **removes
contention; it does not create bandwidth**. With σ = oversubscription of the
promotable tier and f = fraction of critical-path (bottleneck) bytes sitting on
promotable pairs:

```
Δ_oracle    = f · (1 − 1/σ)                  # ceiling: promote everything promotable
Δ_realized  = c · Δ_oracle − α_r / N         # c = capture under the port budget,
                                             #     N = token passes the plan is reused
```

Every next step attacks **f**, **σ**, **c**, or **N**. Nothing else moves the
number.

---

## 1. Why the current configuration reports no win — and why that is not "OCS is useless"

Three concrete, fixable facts, all in code:

| # | fact                                                                  | where                  | consequence                                                                                                                                      |
| - | --------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1 | `Topology.promote_from = (Tier.CROSS_POD,)`                         | `cost_model.py` L143 | only cross-pod pairs can carry a circuit                                                                                                         |
| 2 | `hierarchy_for("realistic")` = 32 nodes/pod = **256 GPU/pod** | `cost_model.py` L202 | an EP ≤ 256 deployment never crosses pods ⇒`n_candidate_promotable_pairs = 0`, `ocs_comparison` returns `applicable: False`              |
| 3 | `FabricConfig.pod_oversubscription = 1.0`                           | `cost_model.py` L88  | even the intra-pod tier that a small-EP deployment actually uses is modelled as uncontended ⇒`promotable` set is empty even if (1) is relaxed |

So today f = 0 **by configuration**, not by physics. The negative result is a
statement about one regime; the plan below goes and finds the regimes where f > 0
and then maximises what can be captured.

---

## 2. The four terms, and how to attack each

### T1 — f: put traffic on promotable pairs

Levers, cheapest first:

1. **EP / pod ratio** — the decisive axis. `world_size > pod capacity` makes
   cross-pod pairs exist: EP ≥ 256 under `realistic`, EP > 16 under `multi_pod`
   (2 nodes/pod), EP 512–1024 for training-scale MoE. This is where the paper's
   "cluster-scale" story lives and where OptiTrainSim's multi-job contention also
   appears.
2. **Model / window size** — E, K, batch: `logs/workload/{qwen15 (E=60,K=4), qwen36 (E=256,K=8), whittle (27B-A17.8B)}`; larger batches raise bytes per
   layer and hence f.
3. **Multi-tenant co-batching** — N tenants whose aggregate placement exceeds a
   pod; this is A5's measured regime (`logs/multi_tenant/run_burst_4t`) *and*
   OptiTrainSim's contention regime (their −30 % of −62 % comes from 1 → 2 OCSes).
   It is the single most promising corner because contention is already measured
   to exist.
4. **Dispatch mode** — `replicated` duplicates bytes (placement-invariant volume):
   the duplicated fan-out is precisely the traffic a circuit can serve; `dedup_*`
   concentrates ingress instead. Sweep all three.
5. **Promote-aware placement (the co-design lever, and our unique contribution).**
   Today placement minimises *unpromoted* bottleneck bytes (dedup ingress skew).
   Change the objective to the bottleneck **remaining after the circuit plan is
   applied** — i.e. minimise bytes left on pairs the plan cannot promote. Because
   `plan_circuits` is degree-bounded and weight-greedy, placement changes both the
   edge weights *and* the degree pressure, so placement and plan must be optimised
   together. This is the mirror image of the repo's existing F1 result and it
   reuses `src/eval/placement_opt.py`.

**Test per lever:** `ocs_comparison(fit, ev, placement, topo, cfg)` — it already
returns `baseline_electrical`, `static_ocs`, `oracle_ocs`, capture, and
`value_of_prediction_pct`.

### T2 — σ: use the contention that actually exists

- `core_oversubscription` sensitivity: 2 / 4 (default) / 8.
- `pod_oversubscription` ∈ {1, 2, 3} **plus** `promote_from=(Tier.CROSS_POD, Tier.INTRA_POD)`
  — a one-line change that makes today's pod-local MoE traffic promotable. This is
  the lever that can turn the current negative into a win for *existing* small-EP
  deployments.
- **Defensibility gate (hard):** σ > 1 at the pod tier must be backed by cited
  fabric numbers (rail-optimized pod uplink/leaf-spine ratios). If it cannot be
  defended, this lever is dropped and the win must come from T1 at EP > pod. All
  σ-dependent claims are reported as an **envelope over σ ∈ {1..4}**, never a point.

### T3 — N: amortise α_r (this is where "no reconfiguration" stops being the answer)

- `breakeven(saved_us_per_layer_pass, n_layers, n_changes, cfg)` already computes
  the passes needed to pay for one reconfiguration. Positive statement: **one**
  reconfiguration, then N decode passes at full NIC rate.
- Policy: fit the plan offline on a calibration window and **hold it for the
  epoch**; F3 (plan Jaccard ≈ 0.09 across windows) says chasing per-window optima
  is noise, so the DP's job is to decide *whether to change at all* — and for a
  stable plan the answer is "change once, then reuse".
- Reconfiguration classes to report: `RECONFIG_CLASSES` = 10 ms / 1 ms / 10 µs / 0.
  The claim must name the class where the win survives.
- Runtime form: release only at collective boundaries (OptiTrainSim §3.2 soft
  preemption) via the existing `src/ocs/online_controller.py` seam.

### T4 — c: capture the headroom under the port/radix budget

- Sweep `n_circuits` ∈ {8, 16, 32} × `ports_per_rank` ∈ {1, 2, 4}; report
  `promotable_traffic_covered_fraction` and `port_saturated_ranks`, plus
  static-vs-oracle capture.
- `plan_circuits` is a greedy ½-approximation of a degree-bounded b-matching;
  raising c is cheap with a real b-matching solve (`scipy 1.16.3` is already
  installed). Report greedy vs optimal so the paper can say which is deployable.

---

## 3. Workstreams

| id | workstream                                 | concrete change                                                                                                                                                                                                                   | output                                                                         | gate                                                                                                                                            |
| -- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| P0 | hygiene (0.5 d)                            | complete OptiTrainSim citation fields; verify the`[lit]` figures the assessment still marks "verify" (§4.1): production EP degrees, rail-optimized pod oversubscription, reconfiguration latencies                             | corrected refs, ledger entries                                                 | every number on a slide has one reproducing command                                                                                             |
| P1 | **win surface** (1–2 d)             | `scripts/ocs_win_surface.py`: matrix over T1 × T2 (`world_size` 16→1024, `topology` 4 styles, `mode`, `promote_from`, σ, `n_circuits`, `ports_per_rank`, reconfig class, 3 workloads) calling `ocs_comparison` | `outputs/ocs_win_surface.json` + Δ_oracle heat map (one panel per σ)       | oracle ≥ static ≥ 0; every claimed cell has`applicable: true` and a defensible regime                                                       |
| P2 | **promote-aware placement** (2–3 d) | new objective in`src/eval/placement_opt.py`: minimise bottleneck **after** the plan; fit on window A, score on B                                                                                                          | placement comparison table: linear / affinity-coordinated (F1) / promote-aware | promote-aware must beat linear out-of-sample on**circuit-applied** bottleneck; if it only wins with circuits, that is the co-design claim |
| P3 | **capture & packing** (1 d)          | extend`src/eval/ocs_eval.py`: static/oracle capture vs port budget; b-matching (scipy) vs greedy                                                                                                                                | capture curves c(port budget)                                                  | capture reported at the budget a real OCS class has (n_circuits, radix)                                                                         |
| P4 | **amortisation** (0.5 d)             | `scripts/ocs_amortize.py` around `breakeven`: passes-to-break-even per class, hold-vs-replan                                                                                                                                  | breakeven table (passes, ms at 20 ms/token)                                    | win must survive at the claimed class with N ≪ passes per serving window                                                                       |
| P5 | **multi-tenant** (2–3 d)            | tenant-pair circuit allocation on the A5 setup;`scripts/vllm_serve.py analyze` TTFT/ITL with/without circuits                                                                                                                   | TTFT/ITL growth reduction vs ε-baseline                                       | contention must be measured, not assumed, before/after                                                                                          |
| P6 | flow bound (2 d)                           | `scripts/flow_bound_check.py`: max-concurrent-flow LP (TE-CCL's "no copy ⇒ real variables") vs our bottleneck                                                                                                                  | ordering table + gap                                                           | ordering preserved; reversal ⇒ cost model needs revision before any claim                                                                      |
| P7 | wall clock (3–5 d)                        | `scripts/replay_placement_wallclock.py` through `src/comm/all_to_all.py`                                                                                                                                                      | measured vs predicted ordering                                                 | ≥ 2 of 3 workloads, non-overlapping noise floors                                                                                               |
| P8 | external replay (TBD)                      | OptiTrainSim packet-level engine (on release), OCS on vs off                                                                                                                                                                      | the EPS-vs-OCS number their paper never reports                                | ordering consistent with our cost model                                                                                                         |

Sequence: **P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8.** P1 is the decision
point; P2/P3/P4 are the engineering that turns headroom into a result.

---

## 4. Win criteria — when we are allowed to write "OCS improves MoE inference by X %"

1. **Applicable and defensible**: `n_candidate_promotable_pairs > 0`, and the
   regime's EP degree / pod layout / σ are backed by cited real numbers.
2. **Envelope, not point**: X quoted over σ ∈ {1..4} and over the four
   reconfiguration classes, naming the class and σ where the win lives.
3. **Baseline discipline**: the same placement and topology, with the circuit set
   the only difference; **tier promotion only** (C5) — never
   `α_ocs = α_eps + T_reconfig`.
4. **Out-of-sample**: fit window ≠ evaluation window; report Δ_static, Δ_oracle and
   `value_of_prediction_pct`. A large value of prediction means online control is
   needed — which we explicitly do **not** want to claim, so prefer a static plan.
5. **Amortised**: `breakeven_token_passes` ≪ passes per serving window at the
   claimed class.
6. **Externally validated**: P6 ordering + P7 wall-clock ordering.

**Kill criterion (the only path back to a negative headline):** if in *every*
defensible regime f < ~10 % — i.e. headroom < ~7.5 % at σ = 4 — then OCS must not
headline, and the paper keeps placement as the result with OCS as scope. State it
up front; measure it in P1.

---

## 5. Paper shape (positive)

- **Headline**: "OCS cuts expert-parallel MoE all-to-all by X % at (σ, EP, model),
  with a plan fitted offline and reused across N token passes."
- **Mechanism section**: promote-aware placement — placement chosen so the
  *residual* (off-circuit) bottleneck is minimal; contrast with the current
  objective (F1) which minimises bytes on pairs the plan cannot promote.
- **Figures**: (1) Δ vs EP/pod ratio, one panel per σ; (2) capture c vs port
  budget, greedy vs b-matching; (3) passes-to-break-even per class; (4) measured
  vs predicted ordering (P7).
- **Scope paragraph**: the applicability boundary (from P1) — where f collapses —
  so the claim cannot be over-read. Not a headline.

Expected shape of the answer, to be confirmed by P1: the win is largest where
(a) EP spans pods, (b) σ is high, (c) dispatch is replicated/fan-out heavy, and
(d) the serving window is long enough to amortise one reconfiguration — which is
exactly the decode-heavy multi-tenant serving corner, i.e. OptiTrainSim's regime
plus our placement co-design.

---

## Slide points (concise, positive framing)

> Suggested slide title: **Making OCS pay for MoE inference**

1. Goal is a **win**, not a certificate: cut MoE all-to-all time with optical circuits.
2. A circuit **removes contention, not bandwidth** — so it pays exactly where the fabric is contended: **Δ ≈ f·(1 − 1/σ)**, minus α_r amortised over reuse.
3. Today we measure no win because **f = 0 by configuration**: only cross-pod pairs are promotable, and an EP ≤ 256 MoE never crosses a 256-GPU pod.
4. **Lever 1 — regime:** EP > pod size, larger E/K/batch, multi-tenant co-batching (the regime where OptiTrainSim measures contention).
5. **Lever 2 — fabric:** pod-level oversubscription σ > 1 (needs cited numbers) — reported as an **envelope** over σ, never a single point.
6. **Lever 3 — co-design:** *promote-aware placement* — minimise the bottleneck **remaining off-circuit**; the mirror of our placement result.
7. **Lever 4 — capture:** pack the port/radix budget properly (b-matching vs greedy, n_circuits, ports_per_rank).
8. **Lever 5 — amortise:** one reconfiguration per epoch, held over N decode passes; release only at collective boundaries.
9. Target claim: **"OCS cuts MoE all-to-all by X % at (σ, EP, model) with an offline, reused plan"** — validated by an LP flow bound and a wall-clock replay.
10. Decision rule: if headroom stays < ~10 % in every defensible regime, placement stays the headline — and that becomes the applicability scope, not the result.
