# Code map — what lives where, and what it does

A navigation document for this repository: the three subsystems, the data
contracts between them, every module's job, every entry point, and the traps a
reader will hit. Written after the four-cell work, so it marks clearly what is
**pre-existing** and what was **added** in that effort.

Companion documents, by question:

| question | document |
| --- | --- |
| Where is the code for X? | **this file** |
| What are the results, and what is assumed vs measured? | `docs/four_cell.md` |
| Which assumptions does the testbed rely on, and are they verified? | `docs/assumptions.md` |
| How do the claims line up with the field? | `docs/research_assessment.md` |
| What is the trace format? | `docs/routing_schema.md` |
| How does the Harvest-schedule prototype work? | `docs/proposal_harvest_schedule.md` |
| What is planned next? | `docs/plan_next_steps.md` |
| How does the MoE implementation align with production frameworks? | `docs/alignment.md` |

---

## 1. Three subsystems, one direction of flow

The repository contains **three worlds** that must not be confused. They share
only one thing: the routing trace format (`src/data/routing_schema.py`).

```
  WORLD 1 — CAPTURE                  WORLD 2 — ANALYSIS (no GPU)        WORLD 3 — RUNTIME (legacy)
  src/data/, src/serving/            src/eval/                          src/runtime/, src/comm/,
  scripts/capture_workload.py        scripts/verify_live_invariance.py  src/ocs/, src/model/
  scripts/vllm_serve.py              scripts/ocs_four_cell.py           src/launcher.py
  scripts/run_vllm.py                scripts/make_figures.py            scripts/run_preset_pipeline.sh
  ---------------------------------  ---------------------------------  ---------------------------------
  real model, real forward pass      pure functions over a trace        torch.distributed workers,
  emits RoutingTrace JSON            emits reports / figures / JSON     real sleep-based delay injection
                                     <- ALL RESEARCH CLAIMS             -> superseded for claims,
                                                                           kept for wall-clock replay
```

Rules of thumb:

* **If a claim is going in a paper, it comes from World 2.** It is deterministic,
  needs no GPU, and its assumptions are enumerated in `docs/assumptions.md`.
* **World 1 must run on real weights.** It exists so World 2 never has to invent
  routing. `README.md` §"cross-cutting rules" is explicit that the engine must
  not contaminate a trace with what it merely executed.
* **World 3 is not a simulator of the same thing.** It runs real
  `torch.distributed` collectives with injected sleeps. It is the only place
  wall-clock time is measured, and `README.md` marks it superseded for research
  claims (see also `docs/assumptions.md` A6, "superseded for research claims").

### 1.1 Data contracts (the interfaces that matter)

```
RoutingTrace (JSON on disk)                      src/data/routing_schema.py
   │  one file per prompt, validated on save
   ▼
CellTable                                        src/eval/trace_ir.py
   │  immutable columnar IR: run, layer, pos, tok, phase, experts[N,K], weights
   │  the ONLY thing World 2 reads; selection ops only (select, by_runs, by_layer,
   │  by_category, decode_only) — never mutate
   ▼
Placement + Topology                             src/eval/cost_model.py
   │  Placement: expert -> rank ([E] global or [n_layers, E] per layer)
   │  Topology: rank -> node -> pod, FabricConfig (bandwidth/latency per tier),
   │           circuits, promote_from
   ▼
TrafficMatrix                                    src/eval/cost_model.traffic_matrix()
   │  [n_dp, W] dispatch message counts, dedup semantics per DispatchMode
   ▼
cost report (dict)                               src/eval/cost_model.evaluate()
      bottleneck_us, cross_pod_bytes, optical_bytes, rank1_energy, …
```

Everything in World 2 is a pure function of those four objects. That is what
makes the invariance claim (A1) structural rather than empirical: the cost model
cannot change a routing decision because it never sees the router.

---

## 2. World 1 — capture

| module | job | notes |
| --- | --- | --- |
| `src/data/routing_schema.py` (332) | `RoutingTrace` / `RunMeta` / `LayerRoute` / `TokenRoute` + `validate()` | the on-disk contract; every backend validates before saving |
| `src/data/mlx_capture.py` (126) | MLX routing capture (`RoutingCapture`) | used by `capture_workload.py`; the fastest path, no vLLM needed |
| `src/data/vllm_capture.py` (639) | vLLM + vllm-metal capture and **steering** (`VllmSteering`) | hook installation, patched gate forwards, Metal text-model patch |
| `src/data/live_capture.py` (158) | `capture_live()` — realtime inference capture | thin wrapper over the vLLM path |
| `src/data/model_utils.py` (111) | architecture-agnostic introspection, MPS patches | `ModelLayout`, `enable_router_logits` |
| `src/serving/suite.py` (443) | **the 112-prompt factorial suite** and the two splits | categories, paraphrase, lexical controls, length ladders, repeats |
| `src/serving/capture.py` (491) | multi-tenant hook installation on vllm-metal | used by `vllm_serve.py run` |
| `src/serving/engine.py` (440) | multi-tenant serving loop (concurrent + sequential baseline) | TTFT/ITL growth measurements |
| `src/serving/workload.py` (315) | arrival generation, prompt pools | builds the multi-tenant load |
| `src/serving/analyze.py` (220) | session analysis, expert contention | `analyze_session` |
| `src/serving/affinity.py` (576) | cross-tenant affinity metrics, weight-aware | has its own test file |
| `src/serving/schema.py` (221) | `MultiTenantSession` / `StepRecord` etc. | session IR |

Entry points: `scripts/capture_workload.py` (batch capture → `logs/workload/<model>/`),
`scripts/vllm_serve.py {run,analyze,affinity}`, `scripts/run_vllm.py`,
`scripts/moe_run.py`.

---

## 3. World 2 — the analytic evaluation chain (where the research happens)

Dependency order, top to bottom. **Bold** = added in the four-cell work.

```
   src/eval/trace_ir.py            CellTable — the immutable IR
        ├── src/eval/affinity_graph.py      6 affinity definitions + load-preserving null
        │        └── src/eval/specialization.py   does the WORKLOAD determine routing?
        ├── src/eval/cost_model.py          placement + topology + tiers -> bottleneck
        │        ├── src/eval/ocs_eval.py        circuit planning, breakeven, stability
        │        │        ├── src/eval/harvest_sched.py   Harvest-style DP over layers
        │        │        └── **src/eval/promote_aware.py**  the co-design objective
        │        ├── **src/eval/completion.py**  bottleneck -> TTFT / ITL / throughput
        │        └── src/eval/placement_opt.py   11+ placement generators
        │                 └── **src/eval/promote_aware.py** (lazy import, one branch)
        └── **src/eval/milp_bound.py**      optimality bounds (scipy LP/MIP)
```

### 3.1 Module-by-module

| module | lines | what it does | key API | reads | produces |
| --- | --- | --- | --- | --- | --- |
| `trace_ir.py` | 369 | canonical routing IR | `CellTable`, `load_workload`, `routing_identical` | manifest + trace JSON | `CellTable` |
| `affinity_graph.py` | 343 | 6 affinity definitions, structure tests, **load-preserving null** | `affinity_matrix`, `layer_affinities`, `pooled_affinity`, `structure_test` | `CellTable` | affinity matrices |
| `specialization.py` | 337 | does routing carry workload structure? | `category_decoding`, `expert_category_mi`, `semantics_vs_lexis` | `CellTable` | decoding rates, MI |
| `cost_model.py` | 523 | **the physical model**: tiers, bandwidths, dispatch semantics, per-rank bottleneck | `FabricConfig`, `Topology`, `hierarchy_for`, `DispatchMode`, `Placement`, `traffic_matrix`, `evaluate` | `CellTable` + `Placement` + `Topology` | cost report dict |
| `placement_opt.py` | 607 | placement generators + their objectives | `make_placement(kind, …)`, `IngressOracle`, `PLACEMENT_KINDS` | fit slice + (for one kind) topology | `Placement` |
| `ocs_eval.py` | 320 | circuit planning, reconfiguration economics, temporal stability | `plan_circuits`, `with_circuits`, `breakeven`, `stability`, `ocs_comparison` | `CellTable` + `Placement` + `Topology` | comparison dict |
| `harvest_sched.py` | 468 | Harvest-style reconfiguration DP across per-layer steps | `per_layer_steps`, `circuit_pool`, `solve`, `schedule_report` | `CellTable` + `Placement` + `Topology` | schedule + report |
| `completion.py` | 135 | **bottleneck → serving metrics** | `ServingModel`, `prefill_pass`, `first_decode_step`, `comm_us`, `timing_report`, `speedup` | `CellTable` slices | TTFT / ITL / throughput |
| `promote_aware.py` | 389 | **co-design objective**: score a placement on the bottleneck *remaining after* promotion; alternating plan ↔ placement | `PostCircuitOracle`, `promote_aware_placement`, `Drain` | fit slice + `Topology` + `OcsConfig` | `Placement` + circuit plan + history |
| `milp_bound.py` | 511 | **optimality bounds**, each labelled with what it proves | `linear_minmax`, `union_minmax`, `combinatorial_union_bound`, `optimality_report` | fit slice (one layer) | bounds + gaps |

### 3.2 The three placement axes a reader must keep apart

This is the most common source of confusion in the repo, because three different
"placements" exist across the two worlds:

| name | module | meaning |
| --- | --- | --- |
| `cost_model.Placement` | `src/eval/cost_model.py` | **expert → rank**, `[E]` or `[n_layers, E]`. Used by all of World 2. |
| `runtime.placement.Placement` | `src/runtime/placement.py` | **expert → rank *and* rank → physical location** for the live runtime. Different class, same name. |
| `Topology` | `src/eval/cost_model.py` | rank → node → pod + `FabricConfig` + `circuits` + `promote_from` |

If you are reading a stack trace and see `Placement`, check which module it came
from before assuming it holds a rank→location table.

### 3.3 What "the x-axis" means in `ocs_eval`

`ocs_comparison(fit, ev, placement, topo, cfg, cost, mode, seed)` returns:

| key | meaning |
| --- | --- |
| `baseline_electrical` | EPS cost of `placement` on the **eval** slice |
| `static_ocs` | circuits planned from the **fit** slice only → honest out-of-sample |
| `oracle_ocs` | circuits planned from the **eval** slice — documented as an upper bound, but **not one in practice** (see §6) |
| `value_of_prediction_pct` | `oracle − static`; can be negative, which is how the false-oracle problem was found |
| `applicable` | `false` when no contended rank pair carries traffic — a statement about the regime, not about OCS |

---

## 4. World 3 — runtime / data plane (superseded, kept for wall-clock replay)

| module | lines | job |
| --- | --- | --- |
| `src/comm/transport.py` | 287 | `Transport` — `torch.distributed` wrapper with delay injection and circuit pre-establishment |
| `src/comm/all_to_all.py` | 387 | the MoE dispatch/combine primitive (`scatter_tokens`, `gather_tokens`, `combine_expert_outputs`) |
| `src/comm/topology.py` | 264 | hierarchical topology, `LinkTier`, `RankLocation` |
| `src/comm/path_resolver.py` | 229 | per-pair path choice: OCS vs EPS |
| `src/comm/timeline.py` | 57 | Chrome Trace / Perfetto export |
| `src/model/qwen_experts.py` | 418 | real Qwen SwitchGLU expert/gate modules in PyTorch |
| `src/model/router_replay.py` | 210 | replays captured routing decisions inside the runtime |
| `src/ocs/circuit.py` | 222 | **the legacy α-β OCS cost model** (`FixedDelayCircuitPool`) — see A6 |
| `src/ocs/online_controller.py` | 267 | adaptive circuit management during inference |
| `src/ocs/placement.py` | 279 | `ExpertAffinityTracker` |
| `src/ocs/preconfig.py` | 354 | training/inference trace → affinity → circuit plan → JSON |
| `src/ocs/topology.py` | 93 | OCS-aware topology config + port pool |
| `src/runtime/worker.py` | 532 | per-rank execution loop |
| `src/runtime/scheduler.py` | 746 | micro-batch scheduler: `run_serial`, `run_overlap`, `run_ocs_pipeline`, `run_ocs_dbo`, `run_ocs_preset`, `run_ocs_online` |
| `src/runtime/placement.py` | 361 | expert→rank and rank→location tables, `build_placement_manifest` |
| `src/launcher.py` | 96 | multi-process launcher; `--config configs/*.yaml` |

Entry point: `python3 -m src.launcher --config configs/<name>.yaml`, or the
convenience wrapper `bash scripts/run_preset_pipeline.sh`.

---

## 5. Entry points — what to run for what

| I want to… | command | world | cost |
| --- | --- | --- | --- |
| capture routing for a model | `python3 scripts/capture_workload.py --model models/<m> --out logs/workload/<name>` | 1 | 5–20 min |
| re-run the Q1–Q5 evidence chain | `python3 scripts/verify_live_invariance.py --workload logs/workload/qwen36 --world-size 32` | 2 | minutes |
| **run the four-cell experiment** | `python3 scripts/ocs_four_cell.py --workload logs/workload/qwen36 --world-size 32` | 2 | ~2–10 min (cache-warm) |
| **only the dispatch-mode axis** | `… --stage dispatch` | 2 | ~1 min |
| **only the bounds** | `… --stage milp --milp-layers 3` | 2 | ~4 min |
| **audit token-ownership assumptions** | `python3 scripts/source_model_sensitivity.py --workload logs/workload/qwen36 --world-size 32` | 2 | ~2 min |
| **tables from results** | `python3 scripts/four_cell_summary.py outputs/four_cell/*.ws*.json` | 2 | instant |
| figures for the chain | `python3 scripts/make_figures.py --workload logs/workload/qwen36` | 2 | ~1 min |
| Harvest-style schedule | `python3 scripts/harvest_schedule_demo.py` | 2 | seconds |
| multi-tenant serving + contention | `python3 scripts/vllm_serve.py {run,analyze,affinity} <dir>` | 1 | minutes |
| EPS vs legacy α/β OCS models | `python3 scripts/compare_ocs_models.py` | 3 | minutes |
| wall-clock replay on real weights | `bash scripts/run_preset_pipeline.sh` | 3 | minutes |
| backend invariance (MLX vs vLLM-metal) | `python3 scripts/compare_backend_traces.py --a … --b …` | 1→2 | seconds |
| model-dependence control | `python3 scripts/compare_model_affinity.py --small … --large …` | 1→2 | seconds |
| visual check of a trace | `python3 scripts/routing_map.py <trace.json>` | 2 | seconds |
| tests | `python3 -m unittest discover -s tests -v` | — | ~1 s |

`ocs_four_cell.py` writes `outputs/four_cell/<model>.ws<N>.json` **after every
stage**, and merges into an existing file when you re-run a single stage, so
iterating on one stage never discards the others.

---

## 6. Structural quirks and traps

Read this section before debugging anything.

1. **`evaluate()` and `ocs_comparison()` force `n_dp = world_size`**
   (`cost_model.py`, `ocs_eval.py`). DP < EP — 24 of 32 ranks holding experts but
   no tokens — is therefore *unrepresentable* through them. `evaluate()` itself
   accepts `n_dp`; `scripts/source_model_sensitivity.py` drives it directly to
   measure that regime. This is the single most surprising limitation in the cost
   model.
2. **`token_rank()` invents the source side of the traffic matrix** with
   `(run * 1_000_003 + pos) % n_dp`. Routing (which experts) comes from the trace;
   *ownership* (which rank sends) is modelled, uniform, uncited, and no gate covers
   it. It is also what makes the traffic matrix look rank-1. See
   `docs/four_cell.md` F8/V.1.
3. **`oracle_ocs` is not an oracle.** `plan_circuits` is a greedy degree-bounded
   b-matching (a ½-approximation), so a static plan can beat the "oracle":
   measured `value_of_prediction_pct = −0.099 %`. Any "static is nearly optimal"
   argument must be re-derived from a real optimum.
4. **Two different `Placement` classes** (§3.2) — `src/eval/cost_model.py` vs
   `src/runtime/placement.py`.
5. **`combine` is charged as a mirror of dispatch.** `evaluate()` multiplies by
   `2 * n_microbatches` when `CostConfig.include_combine` is true, so a
   "bottleneck_us" is dispatch **plus** combine unless you turn that off.
6. **`CellTable.select()` keeps the parent's `runs` list**, so `t.n_runs` stays
   the *whole workload's* run count after `by_runs`/`by_category`. Use
   `completion.n_sequences(t)` (distinct `run` ids) instead — using `n_runs`
   inflated throughput by 5.4× here.
7. **Tier promotion is the only OCS mechanism modelled in World 2.** A circuit
   removes oversubscription for the pair it serves; it never adds bandwidth. So
   `promote_from` + `oversubscription > 1` gate everything: with the defaults
   (`promote_from=(CROSS_POD,)`, `pod_oversubscription=1.0`) an EP ≤ 256
   deployment reports `applicable: false`, which is why the old Q5 verdict read
   as "OCS is useless" rather than "this configuration has nothing to promote".
8. **Two function-local imports hold the `placement_opt` ↔ `promote_aware` pair
   together.** `placement_opt.make_placement` imports `promote_aware_placement`
   inside its `promote_aware_layer` branch (`placement_opt.py:506`), and
   `promote_aware.promote_aware_placement` imports `make_placement` inside the
   function (`promote_aware.py:297`). Both are deliberate: either one moved to
   module level creates a circular import. Verified: importing
   `src.eval.placement_opt` does **not** pull in `promote_aware`.
9. **`logs/` is gitignored**; `outputs/four_cell/cache_*/` is gitignored
   (rebuildable placement caches). Tracked results live in `outputs/four_cell/*.json`.
10. **Two cost models coexist.** World 2 uses tier promotion; World 3 uses the
    legacy α-β fixed-delay model (`src/ocs/circuit.py`), which is explicitly
    **superseded for research claims** (`docs/assumptions.md` A6) because
    `α_ocs = α_eps + T_reconfig` makes a hot circuit exactly as fast as electrical.
    Never quote a World 3 OCS number as a research result.

---

## 7. Added in the four-cell work (this session)

Everything below is new unless marked *modified*. Nothing else in World 2 was
changed; World 1 and World 3 were not touched.

| file | lines | what it is | depends on | used by |
| --- | --- | --- | --- | --- |
| `src/eval/promote_aware.py` | 389 | the co-design cell: `PostCircuitOracle` scores a *per-layer* placement on the bottleneck that remains after the circuit plan, reproducing `cost_model.evaluate`'s bottleneck term exactly; `promote_aware_placement` alternates plan → placement → plan and returns the best round seen | `cost_model`, `ocs_eval`, `placement_opt` (lazy) | `make_placement("promote_aware_layer")`, `ocs_four_cell.py` |
| `src/eval/milp_bound.py` | 511 | scipy LP/MIP + a combinatorial bound on dedup ingress, each labelled with what it does and does not prove | `trace_ir`, `placement_opt` | `ocs_four_cell.py --stage milp` |
| `src/eval/completion.py` | 135 | composition of prefill and decode pass costs into TTFT / ITL / throughput; `compute_us_per_layer` is an explicit parameter | `cost_model`, `trace_ir` | `ocs_four_cell.py --stage timing` |
| `scripts/ocs_four_cell.py` | 735 | the experiment: 2×2 + **synergy**, regime envelope, dispatch modes, completion time, bounds; incremental JSON write-back | all of the above | — |
| `scripts/source_model_sensitivity.py` | 258 | token-ownership × DP audit; monkey-patches `cost_model.token_rank` for the duration and restores it | `cost_model`, `ocs_eval`, `placement_opt` | — |
| `scripts/four_cell_summary.py` | 242 | JSON → markdown tables (one reproducing command per number) | — | — |
| `tests/test_promote_aware.py` | 432 | 33 tests: oracle fidelity vs `evaluate`, the four-resource drain, bound validity (averaging floor), completion maths | — | — |
| `src/eval/placement_opt.py` | *modified* | **additive**: `promote_aware_layer` branch + 3 keyword-only params defaulting to `None` | — | — |
| `README.md`, `.gitignore` | *modified* | module table + doc pointer; ignore placement caches | — | — |
| `docs/four_cell.md` | 694 | the results and the assumption audit | — | — |

Outputs: `outputs/four_cell/*.json` (raw), `outputs/four_cell/REPORT.md`
(generated), `logs/four_cell/*.log` (run logs, gitignored).

### 7.1 How the new pieces slot into World 2

```
CellTable ──► make_placement(kind, fit, …) ──► Placement
                  │                            │
                  │  kind="promote_aware_layer" │
                  └──────────► promote_aware.py ┤
                                   │            │
                          Topology + OcsConfig  │
                                   ▼            ▼
                            PostCircuitOracle  cost_model.evaluate ──► bottleneck_us
                                   │                    ▲
                                   │                    │
                            ocs_eval.plan_circuits ──────┘
                                   │
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
      ocs_four_cell.py     completion.py          milp_bound.py
      (2x2, regime,        (TTFT/ITL/            (bounds on the
       dispatch, milp)      throughput)           placement objectives)
              │
              ▼
   outputs/four_cell/*.json ──► four_cell_summary.py ──► REPORT.md
```

### 7.2 Where to look for a given finding

| finding (see `docs/four_cell.md`) | code that produces it |
| --- | --- |
| F1 the 2×2 and the synergy verdict | `ocs_four_cell.two_by_two` + `stage_four_cell` |
| F2 the substitution gradient | `stage_four_cell` (per-placement cells) |
| F3 the co-design cell and its control | `promote_aware.promote_aware_placement`, `placement_opt` branch, `stage_four_cell["co_design"]` |
| F4 objective mismatch, bounds | `milp_bound.optimality_report`, `stage_milp` |
| F5 reconfiguration vs ports | `ocs_eval.breakeven`, `stage_regime` port sweep |
| F6 σ sensitivity | `stage_regime` regimes |
| F8 ownership / DP audit | `source_model_sensitivity.py` |
| F9 TTFT / ITL | `completion.timing_report`, `stage_timing` |
| F10 dispatch modes | `stage_dispatch` |

---

## 8. Configs and tests

| config | world | what it runs |
| --- | --- | --- |
| `qwen_eps_baseline.yaml` | 3 | EPS baseline |
| `qwen_ocs_lite.yaml` / `qwen_ocs_pipeline.yaml` | 3 | small / 32-expert OCS pipeline |
| `ocs_alpha_model.yaml` / `ocs_beta_model.yaml` | 3 | legacy α (fast switch) / β (single-port MEMS) models |
| `ocs_affinity_placement.yaml` | 3 | affinity-driven expert→rank **and** rank→location, with online circuits |
| `qwen_replay.yaml` | 3 | replay of captured routing |

Tests: `tests/test_affinity_weights.py` (pre-existing, weight-aware affinity
metrics) and `tests/test_promote_aware.py` (added). Both are pure unit tests —
no captured data, no GPU, sub-second. Run with
`python3 -m unittest discover -s tests -v`.

---

## 9. Reading order for someone new

1. `README.md` §"The result" — what the repo claims.
2. `docs/assumptions.md` — what it is allowed to claim (A1 is the load-bearing one).
3. `src/eval/trace_ir.py` → `src/eval/cost_model.py` — the two files that define
   everything else's vocabulary.
4. `src/eval/ocs_eval.py` — how a circuit is modelled and priced.
5. `docs/four_cell.md` Part V — where the founding assumption does *not* reach.
6. Only then World 3, and only if you need wall-clock numbers.
