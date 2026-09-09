# Proposal — trace-guided OCS schedules for MoE (Harvest machinery, no MILP)

Status: proposal + additive prototype (`src/eval/harvest_sched.py`,
`scripts/harvest_schedule_demo.py`). Nothing existing is modified; the demo is
runnable on captured workloads today.

---

## 1. The thread this ties together

Three pieces already exist on this machine:

1. **Harvest** (`~/Downloads/Projects/harvest`, arXiv:2602.09188): given a
   fixed *step sequence* `<m_i · M_i>` and photonic costs (α, δ, β, α_r),
   synthesise *when* to reconfigure the optical interconnect and *which
   topology* runs until the next reconfiguration.  Machinery: per-interval
   subproblem (best static topology for steps a..b), DP over segment
   boundaries (Algorithm 1), Theorem-1 sweep over the rewire count k
   (`min_k DP[0][k] + k·α_r`), baselines = never-rewire / rewire-every-step.
   Its paper solves the general interval subproblem with MILP; the
   reproduction here (and its authors) escapes it via closed forms and a
   restricted pool.  **We are the restricted case — no MILP.**
2. **TE-CCL** (arXiv:2404.16460, SIGCOMM'24): collective = multi-commodity
   flow over time (chunks × epochs × buffers).  Two lessons for us:
   (a) *integer variables exist only to track in-network copy* — no-copy
   demands (all-to-all) drop to an LP; structure decides the solver, so the
   aggregate rate model we already use needs no integer program; (b) scale by
   *partitioning in time* (A*-rounds) — the time-domain analogue of Harvest's
   segment DP, with a user-tunable optimality gap.
3. **This repo's evidence chain**: routing is a pure function of
   (input, weights) — bit-exact across placement/topology (Q1).  Hence the
   *per-layer dispatch demand of a window is knowable from traces before any
   scheduling decision*.  Measured negatives: aggregate matrix is ~rank-1; at
   realistic pod sizes there is no cross-pod traffic to promote; circuit-plan
   Jaccard ≈ 0.09–0.11 across request/layer windows, so *dynamic* plans are
   weight-tie noise, and a static fit-only plan captures a small share of the
   critical path.

**The synthesis:** the MoE forward pass over one workload window *is* a step
sequence — layer ℓ's dispatch(+combine) round is step ℓ with demand M_ℓ
recovered from the routing cells.  Run the Harvest DP over *that* sequence,
with the "topology pool" restricted to circuit plans this repo already builds
(EPS-only, window-fit, per-layer), scoring DCT per (step, plan) with this
repo's own bottleneck cost model, and paying α_r per actual circuit-set change.
No MILP appears anywhere.  The DP *computes* where the earlier negatives hold
(k\* = 0 → static certified) and where they do not (k\* > 0 → per-window
schedule, only when α_r and message sizes make rewiring pay).

## 2. The model, mapped onto Harvest's notation

```
step ℓ   : layer ℓ dispatch(+combine) demand matrix M_ℓ over ranks
           (from routing cells — placement/topology-independent, bit-exact)
topo  G  : EPS fabric + degree-bounded circuit set (a plan_circuits output)
DCT_ℓ(G) : evaluate(by_layer(ℓ), …, topo_G)["bottleneck_us"]   (repo model)
tc(a,b)  : Σ_{ℓ=a..b} DCT_ℓ(G_{a,b})      (G_{a,b} = argmin over the pool)
DP[a][t] : min over cut b of tc(a,b−1) + DP[b][t−1]            (Alg. 1)
answer   : min over t of DP[0][t] + t·α_r                      (Thm 1)
```

Pool construction is deterministic and solver-free: `eps`, `fit_static`
(greedy plan on the window's summed traffic), and `plan.L<layer>` (greedy plan
of each single layer) — deduped by circuit set.  Per-layer plans are the
"step-matching" candidates: the DP decides whether chasing the measured
per-layer churn is worth its rewires, which is precisely the question the
Jaccard ≈ 0.09 number raises but never answers.

Costs are the repo's own `evaluate()` bottleneck numbers, so the DP cannot
disagree with the evidence chain about what a layer costs under a circuit set;
the physical-cost convention of the Harvest reproduction is kept (adjacent
segments reusing one circuit set merge; α_r is charged per actual change).

## 3. What the prototype does

`python scripts/harvest_schedule_demo.py --workload logs/workload/smoke
--world-size 20 --topology multi_pod`

1. loads a captured workload into a `CellTable` (existing loader);
2. builds the per-layer step sequence (existing `traffic_matrix`, one call per
   layer — new `per_layer_steps`);
3. builds the restricted pool (existing `plan_circuits` + `with_circuits`);
4. scores every (step, member) with `evaluate` (`step_costs`);
5. runs the Harvest DP sweep over k (`solve`), plus baselines `eps_static`,
   `static_best`, `bvn_per_step`;
6. writes a full JSON report under `logs/workload/harvest_schedule/`.

Demonstrated on the smoke workload (Qwen1.5-MoE, E=60, K=4, W=20, 2 pods):

| switch class | α_r     | rewires | harvest | vs EPS-static | verdict        |
|--------------|---------|---------|---------|---------------|----------------|
| mems_10ms    | 10 ms   | 0       | 742 µs  | −30.0 %       | k*=0: static plan certified |
| mems_1ms     | 1 ms    | 0       | 742 µs  | −30.0 %       | k*=0: static plan certified |
| fast_10us    | 10 µs   | 0       | 742 µs  | −30.0 %       | k*=0: static plan certified |
| ideal_0      | 0       | 16      | 726 µs  | −31.6 %       | per-layer chasing pays ~nothing |

On `realistic` pod sizes the pool degenerates to `['eps']`, harvest = static =
EPS, k\* = 0: the repo's "no cross-pod traffic" negative, now *derived by the
optimizer* rather than asserted.  Both behaviours are the point: the numbers
decide, and under MEMS-class switches the DP independently certifies the
static-plan conclusion that the Jaccard analysis already reached.

## 4. Deliverables and gates before this becomes a claim

Additive files only:
- `src/eval/harvest_sched.py` — steps/pool/costs/DP/baselines/orchestration;
- `scripts/harvest_schedule_demo.py` — CLI demo + report writer;
- this document.

Gates to turn the prototype into an evidence-chain result:
1. **Out-of-sample plans.**  The pool is currently built and scored on the
   same window.  Split fit windows (plan construction) from evaluation windows
   (scoring) so k\* is not overfitted, mirroring the existing leave-out
   discipline.  Expect Jaccard-style churn to push k\* → 0; *that is the
   result*, quantified.
2. **Alpha-r realism.**  α_r must sit against a *layer's* dispatch time, not a
   whole pass: rewires happen between layers.  The per-layer DCT column of the
   report is the honest comparison scale (10 µs–100 µs layer rounds vs 10 ms
   MEMS → k\* = 0 before even running the DP; SOA-class switches + large
   batches open the k\* > 0 regime).
3. **Dispatch-mode sweep.**  REPLICATED vs DEDUP_RANK vs DEDUP_NODE change
   volumes and fan-out; the TE-CCL lesson says only multicast-bearing demands
   would need integer machinery, and our aggregate model never does.  Sweep
   the mode axis and show the DP outcome per mode.
4. **Schedule replay.**  The segment schedule (rewire at layer boundaries) is
   the input the runtime would need; no execution exists yet.  Position this
   as the plan for the OCS-feasibility section, matching the repo's staging.

## 5. Honest limits

- Aggregate `evaluate()` bottleneck ≈ rate model, not a chunk-tracking LP;
  per-layer slices average a window's tokens into one matrix per step.  The
  DP inherits the repo's semantics by construction — it cannot contradict
  them, only rearrange them in time.
- Steps collapse a whole window's tokens into one round; real serving runs
  many microbatches and could pipeline layers.  Pipelining only strengthens
  the k\* = 0 regime (rewire gaps shrink), so the conclusion direction is
  safe, but numbers should be reported as "per window pass".
- Plan stability across windows (Jaccard ≈ 0.09) is *why* schedules should be
  recomputed per window and reused for many passes (α_r amortised over token
  passes via `breakeven`), or replaced by k\* = 0 — never chased online.
