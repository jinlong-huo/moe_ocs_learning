# The four-cell experiment, and an audit of the assumption it rests on

**What this document is.** A complete record of the work done on top of the
existing testbed: what was already there, what I added and why each piece was
necessary, every measured finding, and — the part that matters most — an audit
of where the results are *not* aligned with the founding assumption the whole
repo is built on (`docs/assumptions.md` A1: routing is a pure function of
(input, weights)).

**How to read it.** Every number below is tagged with how it was obtained:

* **[M]** measured by a command in section 10.
* **[C]** read from the code (a call-site or a constant), not measured.
* **[A]** assumed — a modelling choice with no measurement behind it. These are
  collected in Part V, because they are where the results are soft.

The distinction is the whole point of the document: the experiment is
reproducible and its arithmetic is checked, but several quantities that decide
the answer are **[A]**, and two of them are outside the reach of A1.

---

## Part I — The starting point

### 1.1 The founding assumption

`docs/assumptions.md` A1 **[C]**:

> In exact arithmetic, gate math depends only on input tokens and model weights.
> The engine (MLX / HF / vLLM-metal), the physical topology (pods × nodes × ranks,
> latencies, BW), and the node distribution of experts must not change *which*
> expert a token hits — only the *cost* of getting there.

This is the `f = (inputs, model)` assumption. Everything else in the ledger is
derived from it:

| # | assumption | what it licenses |
| --- | --- | --- |
| A1 | routing = f(inputs, weights) | traces can be replayed under any placement/topology |
| A2 | affinity is model-specific | presets are re-derived per model |
| A3 | per-cell routing is noisy; trust distributions | aggregate metrics only |
| A4 | placement is a *free, cost-side* variable | affinity may decide where experts live |
| A5 | multi-tenant co-batching creates contention | the serving regime is measurable |
| A6 | legacy α-β OCS cost model | **superseded** for research claims (C5) |

The research-grade cost path is tier promotion (`cost_model.py` + `ocs_eval.py`),
not A6. That path is what this work exercises.

### 1.2 What the pipeline already computed, and what each piece takes as given

The four cells of the design were, in code terms, mostly already there:

| design axis | existing artefact | what it silently assumes |
| --- | --- | --- |
| EPS substrate | `evaluate()` on a plain `Topology` | dispatch semantics (`DispatchMode`) **[A]** |
| OCS substrate | `ocs_eval.ocs_comparison()` → `static_ocs`, `oracle_ocs` | `plan_circuits` is a *good* plan **[A]** |
| no affinity | `make_placement("linear"/"random"/"load_balanced")` | — |
| affinity-guided | `make_placement("affinity_layer"/"affinity_coordinated_layer"/…)` | the objective is the right one **[A]** |
| the crossing | **absent** — `stage_q5` in `verify_live_invariance.py` takes one placement and crosses it with topology × reconfig class, never placement × OCS | — |

Two structural facts about the existing cost model matter for everything below
**[C]**:

1. `cost_model.token_rank()` assigns every routing cell to a DP rank by
   `(run * 1_000_003 + pos) % n_dp`. The *destination* of traffic comes from the
   trace (A1 protects it); the **source** is invented, uniformly at random.
2. `evaluate()` and `ocs_comparison()` both default `n_dp` to `world_size`, so
   **the pipeline cannot express DP < EP**: 24 of 32 ranks owning experts but no
   tokens is a standard MoE deployment and is unrepresentable.

Both are assumptions *outside* A1, and Part V shows the first one decides the
OCS result.

---

## Part II — What I added, and why

### 2.1 The crossing (deliverable A)

`scripts/ocs_four_cell.py` — nested loop over placements × {EPS, static OCS,
oracle OCS} under a regime, plus four supporting stages (regime envelope, port
budget, completion time, bounds). It imports the existing modules and changes
none of them. It also computes the number the design actually needs:

```
synergy = Δ(A→D) − [ Δ(A→B) + Δ(A→C) ]
```

`D` being the minimum cell is *not* evidence for the design's claim; the sign of
`synergy` is. This is the single most important addition, because it is the
difference between "the cell wins" and "the mechanisms cooperate".

### 2.2 The co-design cell (deliverable B)

`src/eval/promote_aware.py` (+ a registration branch in `placement_opt.py`).

The problem it exists to break: every existing generator optimises an
*electrical* objective and then a circuit plan is fitted to whatever traffic it
produced. That ordering cannot test co-design, because the placement never sees
the plan it will be paired with — the pair `(placement*, plan*)` is never jointly
selected. `promote_aware_layer` alternates plan → placement → plan, scoring each
placement on the bottleneck that **remains after** promotion.

Fidelity was the risk, so it is machine-checked rather than asserted:
`PostCircuitOracle.bottleneck_us` reproduces `cost_model.evaluate`'s bottleneck
term to float32 precision, across topologies and with/without circuits
(`tests/test_promote_aware.py`). Two model details are load-bearing and both were
got wrong on the first attempt (Part IX):

* a rank's NVLink drain and NIC drain are **separate resources** — the critical
  path is a max over four vectors, never a sum;
* the byte contraction must return `[src, dst]`, not its transpose.

The generator also returns the best round seen, because alternating optimisation
is not monotone (a round re-plans against a placement whose traffic changed, so
its objective is measured against a different circuit set).

`bottleneck_search_layer` is the control: the same search with `n_circuits=0`.
Without it, "co-design lost" cannot be distinguished from "this search is
weaker than the optimiser it started from".

### 2.3 Bounds (deliverable C)

`src/eval/milp_bound.py`. The design asks for a MILP over the affinity graph. Two
things are true and both matter:

* A **union cardinality** ("how many distinct destination ranks does this token
  reach") is not linear in the assignment, so no compact MILP reproduces the
  bottleneck the cost model charges.
* What *can* be bounded is the min-max **message count** per rank, plus a
  rigorous (loose) combinatorial bound on exact dedup ingress:
  `max(ceil(K/cap)·N/W, max_e reach_e)`.

The bound that is actually quoted is an **LP relaxation**, because it always
exists. That is not a stylistic choice: on the real instance the union MIP found
no feasible assignment within 60 s and scipy reports no dual bound in that case,
and the union LP did not solve in 60 s at 300 expert sets. **A bound that exists
only when the solver succeeds is not a bound.**

### 2.4 Completion time

`src/eval/completion.py`, because the design's axis (4) asks for TTFT and
throughput and the repo had no such composition. `TTFT = prefill_comm + compute +
reconfig/N`, `ITL = decode_comm + compute + reconfig/N`. `compute_us_per_layer` is
an **explicit parameter, never a measurement** — with the default 0 the report is
the communication component, which is the only honest headline, since an OCS can
only remove communication.

### 2.5 Reporting

`scripts/four_cell_summary.py` turns the JSONs into markdown tables, so every
number in a slide has one reproducing command (the repo's P0 rule).
`scripts/source_model_sensitivity.py` is the audit tool of Part V.

---

## Part III — Method

* **Inputs [M]**: real captured routing, three models, chosen for structural
  diversity: Qwen3.6-35B (E=256, K=8, 40 MoE layers, load skew 10.5×), Whittle-27B
  (E=64, K=16, 64 layers, K/E=25 %), Qwen1.5-MoE (E=60, K=4, 24 layers, skew 1.64×).
* **Out-of-sample everywhere [M]**: placements and the static plan are fitted on
  one category split, scored on a disjoint one (leave-categories-out). `oracle_ocs`
  fits its plan on the evaluation slice; the gap to it is the value of prediction.
* **Decode cells** for the communication comparison (the existing chain's
  convention); prefill and single-decode-step slices are used **only** by the
  completion stage.
* **Reference regime**: 2 pods × 16 GPUs (`multi_pod`, σ_core=4, σ_pod=1),
  16 circuits, 2 ports/rank, static plan.
* **Applicability gate [C]**: a regime with no contended rank pair carrying
  traffic returns `applicable: false`. `single_pod` and `realistic` (256 GPU/pod)
  are reported that way rather than dropped.

Cells, exactly:

| cell | meaning |
| --- | --- |
| A | EPS bottleneck, `linear` placement (the deployed default) |
| B | EPS bottleneck, `affinity_coordinated_layer` |
| C | A's placement + static OCS plan fitted on the fit slice |
| D | B's placement + static OCS plan fitted on the fit slice |

---

## Part IV — Findings

### F1 — D is the minimum cell in all three models [M]

| model | E | K | A (µs) | B (µs) | C (µs) | D (µs) | affinity | OCS | both | synergy | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen1.5 | 60 | 4 | 2412.8 | 2222.7 | 2366.9 | **2137.7** | +7.88 % | +1.90 % | +11.40 % | **+1.62 %** | complements |
| Qwen3.6 | 256 | 8 | 3956.2 | 2944.1 | 3768.0 | **2869.7** | +25.58 % | +4.76 % | +27.46 % | **−2.87 %** | substitutes |
| Whittle | 64 | 16 | 12639.2 | 10120.3 | 11569.4 | **9825.5** | +19.93 % | +8.46 % | +22.26 % | **−6.13 %** | substitutes |

The design's headline survives: D wins every time. Its mechanism does not: on the
two models where placement does heavy lifting the two mechanisms are
**substitutes**, and the synergy is *more negative the more affinity buys*
(+1.62 % where affinity buys 7.9 %; −6.13 % where it buys 19.9 %).

Two caveats found later and folded in below: the Qwen1.5 "complements" reading is
**fragile** and does not survive a change of dispatch mode (F10), and D's margin
over B — the value of adding circuits at all — depends on the token-ownership model
by up to 148× (F8).

### F2 — The substitution is a monotone gradient [M]

Qwen3.6, σ=4, same topology, ordered by the placement's own EPS bottleneck:

| placement | EPS (µs) | static OCS gain | promotable traffic covered |
| --- | --- | --- | --- |
| `affinity_coordinated_layer` | 2944 | **+2.53 %** | 6.6 % |
| `promote_aware_layer` | 3118 | +5.53 % | 6.8 % |
| `bottleneck_search_layer` (control) | 3173 | +4.22 % | 6.6 % |
| `load_balanced` | 3703 | +2.60 % | 6.6 % |
| `linear` | 3956 | +4.76 % | 6.9 % |
| `random` | 4406 | +8.45 % | 6.9 % |
| `load_balanced_layer` | 5412 | +8.44 % | 6.6 % |
| `affinity_layer` (naive, 2× worse than random) | 8082 | **+8.32 %** | 8.7 % |

The three best placements average a **4.1 %** OCS gain; the three worst average
**8.4 %**. OCS is worth roughly **twice as much to a badly-placed deployment**.
Coverage stays at ~7 % throughout, so the mechanism was never bandwidth-limited —
a better placement changes *which* rank pairs exist to promote, not how many bytes
are cross-pod (see F7).

### F3 — The co-design cell lost, and the control says why [M]

`promote_aware_layer` vs the best electrical placement, both under static OCS:

| model | promote-aware on OCS | its own circuit gain | control on OCS | control on EPS |
| --- | --- | --- | --- | --- |
| Qwen3.6 | −2.66 % | +5.53 % | −5.89 % | −7.76 % |
| Whittle | −0.88 % | +8.51 % | −10.13 % | −7.79 % |
| Qwen1.5 | −3.13 % | +4.56 % | −6.07 % | −7.02 % |

The circuit-aware objective works on its own terms — it raises the circuit gain
(2.53 % → 5.53 %) and on Whittle nearly recovers the whole deficit (−0.88 %).
But the control shows the search machinery is itself 7–8 % behind
`affinity_coordinated_layer` **on the EPS substrate**, in all three models. So
the correct statement is *"this co-design search does not win yet"*, not
*"co-design does not work"*. The control has to be beaten first.

### F4 — LPT is optimal for the wrong objective [M]

Three layers of Qwen3.6, not just layer 0 — a single layer turned out **not** to be
representative (see the method note at the end of this finding):

| layer | objective | proven bound | `load_balanced_layer` | `affinity_coordinated_layer` | `linear` |
| --- | --- | --- | --- | --- | --- |
| 0 | max messages, linearised | 512 (LP) | **+1.8 %** | +190.0 % | +79.5 % |
| 1 | max messages, linearised | 512 (LP) | **+1.8 %** | +184.8 % | +79.7 % |
| 2 | max messages, linearised | 512 (LP) | **+6.4 %** | +277.1 % | +68.6 % |
| 0 | max messages, exact union | 463 | +11.2 % | **+8.0 %** | +91.4 % |
| 1 | max messages, exact union | 498 | **+4.6 %** | +43.8 % | +59.4 % |
| 2 | max messages, exact union | 478 | **+13.0 %** | +80.5 % | +63.0 % |
| 0 | max **cost-model bottleneck** | — | 5412 µs | **2944 µs** | 3956 µs |
| Whittle (1 layer) | max messages, linearised | 1669 | **+0.0 %** | +54.9 % | +97.2 % |
| Qwen1.5 (1 layer) | max messages, linearised | 454 | **+0.0 %** | +27.8 % | +36.3 % |

Two things are true and they must not be conflated:

* **Per-layer LPT is essentially optimal for the per-layer count objective** —
  within 1.8–6.4 % of the linearised bound and 4.6–13 % of the union bound — and it
  uses **no affinity graph at all**.
* **That bound cannot arbitrate between the two placements**, because they
  optimise different things. `affinity_coordinated_layer` is *coordinated*: it
  minimises the cross-layer accumulated critical path (`max_rank Σ_l ingress_l`),
  not a per-layer maximum, so a poor per-layer reading is not evidence against it —
  it is the documented design. `load_balanced_layer` is the only apples-to-apples
  entry, and it is the one the bound flatters.

The metric that *does* arbitrate is the cost model's bottleneck, and there affinity
wins by 45 % (Qwen3.6) and 89 % (Whittle) over per-layer LPT — because LPT
balances **messages** while ignoring **which pair** each message crosses, and the
fabric charges by pair (intra-node ≈ 9× intra-pod; cross-pod a further 4× worse at
σ=4). So the repo's placement claim sharpens to: what matters is balancing
tier-aware bytes, not counts, and neither per-layer alone nor pooled affinity alone
is enough. `load_balanced_layer` is now in the four-cell table because it is the
baseline the claim has to beat.

**Method note.** The single-layer version of this table was misleading. On layer 0
affinity appeared to match LPT on exact ingress (8.0 % vs 11.2 %); on layers 1–2
LPT is 39 and 67 points *better* on that metric. Per-layer bound claims in this
repo should be reported over several layers, and never used to rank a coordinated
placement against a per-layer one.


### F5 — Reconfiguration is not the binding constraint; ports are [M]

Qwen3.6, `affinity_coordinated_layer`, 16 circuits / 2 ports:

| reconfig class | reconfig | static gain | break-even token passes |
| --- | --- | --- | --- |
| MEMS 10 ms | 10 000 µs | +2.53 % | **3.36** |
| MEMS 1 ms | 1 000 µs | +2.53 % | **0.34** |
| fast 10 µs | 10 µs | +2.53 % | 0.0 |
| ideal | 0 | +2.53 % | 0.0 |

(The same at Whittle: 0.53 passes; Qwen1.5: 4.9 passes.) A serving window is
orders of magnitude longer, so the static-plan reading is the relevant one, and
the repository's framing — that the switch's reconfiguration time decides
feasibility — is **not what binds in these regimes**.

What binds is the **port/circuit budget**:

| circuits | ports/rank | static gain | oracle gain | covered |
| --- | --- | --- | --- | --- |
| 8 | 2 | +0.41 % | +4.55 % | 3.4 % |
| 16 | 2 | +2.53 % | +4.55 % | 6.6 % |
| 32 | 2 | **+8.57 %** | +8.74 % | 13.0 % |

Tripling the budget gives a 3.4× larger gain. So the substitution in F2 is partly
a *budget* artefact: a richer switch can afford to promote the pairs a good
placement left behind.

### F6 — σ moves the naive placements, not the good ones [M]

Static OCS gain by regime (Qwen3.6 / Whittle):

| regime | `random` | `linear` | `load_balanced_layer` | `affinity_layer` | `affinity_coordinated_layer` |
| --- | --- | --- | --- | --- | --- |
| σ=2 | 5.37 / 5.16 | 4.80 / 5.19 | 5.29 / 5.16 | 5.15 / 5.10 | 2.89 / 2.90 |
| σ=4 | 8.45 / 8.43 | 4.76 / 8.46 | 8.44 / 8.47 | 8.32 / 8.39 | 2.53 / 2.91 |
| σ=8 | 10.27 / 9.61 | 4.73 / 10.39 | 10.32 / 10.43 | 10.20 / 10.33 | 2.32 / 2.92 |
| pod σ=2 | 7.60 / 7.20 | 4.56 / 7.62 | 7.57 / 2.54 | 2.80 / 2.54 | 2.54 / 2.80 |

σ roughly doubles the gain for the badly-placed deployments (5.4 % → 10.3 %) and
barely moves the well-placed one (2.89 % → 2.32 %). A deployment that has already
removed its cross-pod hotspots has nothing left for σ to amplify: the
substitution again, seen from the fabric side.

### F7 — What is *not* σ-dependent, and why that matters

`f`, the fraction of bytes on contended cross-pod pairs, is essentially constant
across every source model and every placement tested: **0.490–0.525 [M]**. Yet
the OCS gain varies by a factor of **54** between source models (F8). So `f` alone
cannot predict the gain — the missing variable is how *concentrated* those
promotable bytes are on the busiest rank ports, which is what `plan_circuits`
coverage measures. The design document's `Δ = f·(1 − 1/σ)` is therefore
incomplete: see V.6.

### F8 — The OCS result depends on an assumption the founding claim does not cover [M]

This is the audit's headline, and it is now measured on all three models plus a
DP-size sweep. A1 fixes the **expert** side of the traffic matrix. It says nothing
about which rank **owns** each token, and `cost_model.token_rank` invents that with
a uniform hash. Replacing it with real serving layouts, everything else fixed
(DP = world size, DEDUP_RANK):

| model | ownership | EPS `linear` | EPS `affinity_coord` | OCS gain `linear` | OCS gain `affinity_coord` |
| --- | --- | --- | --- | --- | --- |
| Qwen3.6 | **hash** (repo default) | 3 956 µs | 2 944 µs | **4.76 %** | 2.53 % |
| Qwen3.6 | **per_sequence** | 13 654 µs | 11 020 µs | **0.088 %** | 0.109 % |
| Qwen3.6 | token_roundrobin | 3 967 µs | 2 942 µs | 6.47 % | 5.92 % |
| Whittle | hash | 12 639 µs | 10 120 µs | 8.46 % | 2.91 % |
| Whittle | **per_sequence** | 43 282 µs | 38 286 µs | **0.028 %** | 0.031 % |
| Whittle | token_roundrobin | 12 539 µs | 10 182 µs | 8.38 % | 3.41 % |
| Qwen1.5 | hash | 2 413 µs | 2 223 µs | 1.90 % | **3.83 %** |
| Qwen1.5 | per_sequence | 4 989 µs | 4 400 µs | **5.11 %** | 3.42 % |
| Qwen1.5 | sequence_block | 4 989 µs | 4 400 µs | 8.43 % | 8.89 % |
| Qwen1.5 | token_roundrobin | 2 520 µs | 2 298 µs | 0.48 % | 2.71 % |

And the same axis swept by DP size (Qwen3.6), which the pipeline cannot express at
all through `ocs_comparison` — `evaluate()` and it both default `n_dp` to
`world_size` **[C]**, so DP < EP has to be driven through `evaluate` directly:

| ownership | DP | EPS `linear` | OCS gain `linear` | OCS gain `affinity_coord` | covered |
| --- | --- | --- | --- | --- | --- |
| hash | 32 | 3 956 µs | 4.76 % | 2.53 % | 6.6 % |
| hash | 16 | 7 002 µs | 3.19 % | 0.23 % | 6.6 % |
| hash | 8 | 13 539 µs | **8.72 %** | 8.73 % | 13.6 % |
| per_sequence | 32 | 13 654 µs | 0.088 % | 0.109 % | 10.3 % |
| per_sequence | 16 | 20 352 µs | **0.059 %** | 0.073 % | 10.8 % |
| per_sequence | 8 | 20 352 µs | **8.12 %** | 8.66 % | 14.2 % |

**The OCS gain spans 0.059 % → 8.73 %, a factor of 148**, across configurations that
differ only in unmeasured assumptions — and the dependence is non-monotone in DP
(fewer token owners makes each rank pair hotter, which *helps* circuits, until the
source axis disappears altogether). Nothing in A1 or in any gate distinguishes these
cases, and there is no measurement anywhere in the repo of which one is real.

**Corrected robustness statement.** My earlier reading — "the placement conclusion is
robust, the OCS conclusion is not" — was too generous, and the correction matters:

* The OCS result is **not robust at all**: 148× across ownership and DP.
* The placement result is robust **in sign but not in size**, and it **reverses** in
  one configuration. `affinity_coordinated_layer` is the best-EPS placement in 11 of
  12 measured configurations; the exception is Whittle under `per_sequence`, where
  `load_balanced_layer` wins by 2.1 % (37 501 vs 38 286 µs). More important than that
  reversal is the collapse in *magnitude*: affinity's edge over per-layer LPT is
  **47 % under the hash model and 2.1 % *worse* under per-sequence ownership** on the
  same model. A 47 % headline that becomes −2 % under a different, equally plausible
  token-ownership model is not a result that can be reported without naming the
  ownership model.

### F9 — Completion time shows the same substitution [M]

Communication-only, focus regime:

| model | placement | EPS TTFT | OCS TTFT | TTFT gain | EPS ITL | OCS ITL | ITL gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6 | affinity-coordinated | 2278.9 | 2211.1 | +2.98 % | 190.3 | 168.0 | +11.73 % |
| Qwen3.6 | linear | 2785.7 | 2576.1 | +7.52 % | 347.9 | 335.9 | +3.45 % |
| Whittle | affinity-coordinated | 13792.6 | 13239.1 | +4.01 % | 898.4 | 886.4 | +1.34 % |
| Whittle | linear | 14835.8 | 13904.3 | +6.28 % | 1033.3 | 930.8 | +9.91 % |
| Qwen1.5 | affinity-coordinated | 1172.2 | 1059.4 | +9.62 % | 91.5 | 79.5 | +13.11 % |
| Qwen1.5 | linear | 1398.1 | 1268.2 | +9.30 % | 95.9 | 83.4 | +13.02 % |

The placement buys ~20 % of decode ITL on its own; circuits then buy *either*
~2–12 points on the well-placed deployment *or* ~3–10 points on the default —
never both. Qwen1.5 is the exception, matching its positive synergy.

### F10 — The verdict is stable across dispatch modes, except on the marginal model [M]

`DispatchMode` is documented in `cost_model` as changing the conclusion
*qualitatively*, and A1 says nothing about it, so the substitute-vs-complement
verdict was only a claim about `DEDUP_RANK`. Re-running the four cells under all
three modes:

| model | mode | total-byte spread across placements | affinity | OCS | both | synergy | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.6 | REPLICATED | **0.000000 %** | +5.01 % | +7.62 % | +9.96 % | −2.67 % | substitutes |
| Qwen3.6 | DEDUP_RANK | 33.79 % | +25.58 % | +4.76 % | +27.46 % | −2.87 % | substitutes |
| Qwen3.6 | DEDUP_NODE | 26.33 % | +18.05 % | +6.84 % | +23.42 % | −1.47 % | substitutes |
| Whittle | REPLICATED | **0.000000 %** | +9.59 % | +8.47 % | +12.01 % | −6.05 % | substitutes |
| Whittle | DEDUP_RANK | 17.78 % | +19.93 % | +8.46 % | +22.26 % | −6.13 % | substitutes |
| Whittle | DEDUP_NODE | 12.66 % | +9.91 % | +8.63 % | +16.75 % | −1.79 % | substitutes |
| Qwen1.5 | REPLICATED | **0.000000 %** | **−1.31 %** | +2.58 % | +1.48 % | +0.21 % | additive |
| Qwen1.5 | DEDUP_RANK | 10.46 % | +7.88 % | +1.90 % | +11.40 % | +1.62 % | complements |
| Qwen1.5 | DEDUP_NODE | 6.09 % | +6.25 % | +5.55 % | +11.22 % | −0.58 % | substitutes |

Three things follow.

* **The model's own invariance claim is confirmed numerically**: the placement-to-
  placement spread in total bytes is *exactly* zero under REPLICATED and non-zero
  under both DEDUP modes. That is a real validation of `DispatchMode`'s semantics,
  not an assumption.
* **The substitute verdict is robust on the two models where it matters**: all three
  modes agree on Qwen3.6 and Whittle. So V.7 does not undermine F1's main reading.
* **The complement reading on Qwen1.5 is fragile.** It is +1.62 % under
  `DEDUP_RANK`, +0.21 % ("additive") under `REPLICATED`, and −0.58 % ("substitutes")
  under `DEDUP_NODE`. It is also mechanistically explicable: under REPLICATED the
  volume is placement-invariant, and affinity placement then *costs* 1.31 % of
  bottleneck time, because it can only relocate bytes between tiers and does so
  worse than `linear` on this model. A single positive synergy number from one
  dispatch mode should not be reported as "these mechanisms complement each other".

---

## Part V — Alignment audit: where this work is *not* aligned with the founding assumptions

### V.1 A1 covers the destination of traffic; the cost model needs the source as well

**The gap.** A1 is a statement about which experts a token reaches. The quantity
an OCS is priced against is the **rank-pair byte matrix**, which requires knowing
which rank holds the token. `token_rank()` answers that with a hash **[C]** — an
assumption with no gate, no citation, no measurement. F8 shows it decides the
answer: the OCS gain moves **148×** across plausible (ownership, DP) configurations,
and the placement conclusion is robust in sign but not in magnitude — on Whittle the
affinity placement's 47 % edge over per-layer LPT **reverses to 2.1 % worse** under
per-sequence ownership.

**Why it is easy to miss.** A1's own phrasing encourages the elision: "only the
*cost* of getting there" sounds like cost is a pure function of (routing,
placement, topology). It is — but *routing* in that sentence is a token→expert
map, and cost needs token→*rank*. The inference step "tokens are sharded across
DP ranks independently of their content" is stated in `cost_model.token_rank`'s
docstring as a modelled assumption and described as "load-bearing: it is *why* the
traffic matrix turns out to be near rank-1". It is load-bearing for more than
that: it is what makes the promotable pairs hot enough to be worth a circuit.

**Consequence for the write-up.** Any OCS claim must name the ownership model *and*
the DP size it assumes. The honest regime is the one where a whole sequence sits on
one rank (plain data-parallel serving), where the current static-plan gain is
0.028–5.11 % depending on model — and 0.088 % on Qwen3.6, which is not a result
worth headlining. The hash model is the *favourable* case for circuits (it makes
rank pairs artificially hot) and the paper should say so explicitly. Note also that
the placement paper is not exempt: it must report its affinity margin under at least
the hash and per-sequence ownership models, because on Whittle that margin changes
sign.

### V.2 A1 is a statement about one input; the exploited object is a distribution

A1 is per-input: `f(x, θ)`. Every placement is fitted on a sample and scored on
another, which means the object being exploited is `E[f(x, θ)]` over an input
distribution. The binding assumption is therefore **A1′: routing is f(inputs,
model) *and* the serving input distribution matches the calibration
distribution**. The repo has strong evidence that the input dimension matters a
great deal (Q2: 93.75 % category decoding on Qwen3.6), which makes that
distribution-matching clause a *requirement*, not a footnote. My protocol
(leave-categories-out) tests exactly this and passes, so the assumption holds here
— but it should be stated as an assumption, because Q2 is simultaneously evidence
that C1 is *detectable* if it fails.

### V.3 "Only the cost changes" hides that *which* cost you choose decides the winner

A1 and the derived A4 treat cost as a single well-defined quantity on the cost
side, so a placement can be chosen "freely". Part IV F4 falsifies the practical
form of that: on **max messages** the ranking is
`load_balanced_layer` < `affinity_coordinated_layer` ≪ `linear`; on the
**cost model's bottleneck** it is `affinity_coordinated_layer` < `linear` <
`load_balanced_layer`. The two objectives are both "communication cost" and they
disagree in sign.

So A4's "placement is a free cost-side variable" is true, but its *optimum is not
a function of affinity alone*: the placement objective must include the topology
(and the circuit budget) to be well-posed. A topology-free affinity objective
optimises a proxy that F4 shows to be near-optimal and simultaneously 84 % off the
metric the repo headlines. **This is the most actionable misalignment for the
placement paper: name the objective, and report the other one as a control.**

### V.4 Assumptions the code states that the measurements contradict

| stated in code | measurement |
| --- | --- |
| `ocs_comparison` docstring: `oracle_ocs` is "an upper bound no controller can beat" | `plan_circuits` is a greedy ½-approximation, not an optimal b-matching, so the "oracle" is not an upper bound. Measured `value_of_prediction_pct = **−0.10 %**` (static beat "oracle" at 32 circuits / 2 ports, `linear` placement) **[M]**. Every "static nearly reaches the oracle, so no online control is needed" argument must be re-derived from a real matching optimum (a b-matching solve, as `docs/plan_next_steps.md` P3 already proposes). |
| README: the rank×rank matrix is "99.9 % rank-1 … there is no pairwise structure to engineer around" | rank-1 energy is 0.993–0.999 across *every* ownership model and placement **[M]**, yet OCS gains span 0.088 %–9.1 % in the same set. Rank-1-ness of the count matrix does not imply a promotion gain is impossible, because the gain depends on the *tier-weighted* structure, not the spectral one. The structural argument and the OCS verdict are not as tightly coupled as the README implies. |
| The reconfiguration timescale is the feasibility question (Q5, `breakeven`) | break-even is 0.34–4.9 token passes at the 10 ms class **[M]**. Reconfiguration is *not* binding here; the port budget is (8→32 circuits: 0.41 % → 8.57 %) **[M]**. |

### V.5 Assumptions my own additions introduce

Honesty requires listing these alongside the ones I criticised.

* **`promote_aware.py` optimises against a plan it may not be allowed to hold.**
  It re-derives the circuit set on every round, while the repo's own stability
  measurement says plan churn across request windows is high (Jaccard ≈ 0.09).
  The search therefore treats reconfiguration as free. Any future win from this
  generator must be re-checked against the amortisation account (F5) before it can
  be claimed as deployable.
* **The MILP bounds are bounds *within the repo's placement family*.** Both
  formulations impose exactly `E/W` experts per rank, matching every generator,
  but a real deployment may hold unequal per-rank expert counts — and since reach
  is skewed, unequal counts are plausibly better. So "LPT is within 2 % of
  optimal" means *within this family*, and the family constraint itself may be the
  binding one.
* **`completion.py` treats the captured workload as one homogeneous batch** (one
  prefill pass + one decode step). Batch composition is a scheduler property, not
  a function of (inputs, model), so TTFT/ITL here are per-batch-of-N quantities
  with N taken from the split, not from a serving configuration.
* **Phase transfer.** The completion stage applies a placement fitted on decode
  routing to prefill traffic. A1 does not cover that: it says routing is a
  function of the input, and prefill inputs are different inputs.

### V.6 `f` in `Δ = f·(1 − 1/σ)` is not a workload property

`docs/plan_next_steps.md` writes the OCS headroom as `Δ = f·(1 − 1/σ)` with `f` the
fraction of critical-path bytes on promotable pairs, and treats `f` as something
the *workload* supplies (hence the levers: EP/pod ratio, model size, batch). The
measurements say `f` is nearly invariant (0.49–0.53 **[M]**) across placements and
ownership models, while the *gain* varies 148×. `f` therefore cannot be the
predictive quantity: the second factor is the concentration of promotable traffic
on the busiest ports, which depends on placement, ownership model, and dispatch
mode — the first two of which the plan document treats as fixed backdrop.

A corrected form would read `Δ ≈ c(f, concentration) · (1 − 1/σ)`, where `c` is
the plan's captured share of the critical path — which is the quantity
`plan_circuits` already returns as `promotable_traffic_covered_fraction`, and
which is 3.4–13 % in every measurement here.

### V.7 Dispatch semantics: measured, and the verdict survives — except where it was marginal

`DispatchMode` (REPLICATED / DEDUP_RANK / DEDUP_NODE) is documented as changing the
conclusion *qualitatively*: under REPLICATED the total dispatch volume is
placement-invariant, so affinity can only relocate bytes between tiers; under DEDUP
it can also remove them. Every four-cell run in this document uses `DEDUP_RANK`, and
A1 says nothing about dispatch semantics — so until F10 was run, "affinity and OCS
are substitutes" was a claim about one mode.

F10 closes it, with three results worth separating:

* the mode semantics themselves are **validated** (byte spread across placements is
  exactly 0.000000 % under REPLICATED, non-zero under both DEDUP modes);
* the **substitute** verdict holds in all three modes on Qwen3.6 and Whittle, so the
  main reading does not depend on a dispatch artefact;
* the **complement** reading on Qwen1.5 does not survive: +1.62 % under DEDUP_RANK,
  +0.21 % under REPLICATED, −0.58 % under DEDUP_NODE. Under REPLICATED, affinity
  placement is outright *harmful* on that model (−1.31 %), which is exactly what the
  documented mechanism predicts — with volume fixed, a placement that only relocates
  bytes can lose to one that relocates them better.

The general lesson for the write-up: a synergy number must be reported per dispatch
mode, because the sign of the interaction is not a property of the workload alone.

---

## Part VI — Threats to validity

1. **The ownership model and DP size** (V.1/F8). The single largest threat: the OCS
   gain spans 148× (0.059 %–8.73 %) across configurations that differ only in
   unmeasured assumptions, and none of them is measured anywhere in the repo. The
   pipeline cannot even express DP ≠ EP through `ocs_comparison` **[C]**.
2. **No measured source locality.** Real serving has locality the hash destroys
   (prefix caching, continuous batching, same-tenant prompt reuse). Any of these
   would change pair concentrations in a direction nothing here can predict.
3. **σ is a literal, not a citation** **[C]**: `FabricConfig.core_oversubscription
   = 4.0`. All gains scale with it (F6).
4. **Bounds are per-layer, and the two placements have different objectives.** F4 now
   reports three layers, and the layer-0-only version was misleading. The bounds
   arbitrate only among per-layer objectives; a coordinated placement is not
   competing on that metric, so a single-layer bound must never be used to rank the
   two families. Still one split and one seed.
5. **No measured compute.** TTFT/ITL are communication-only until someone supplies
   kernel timings; the sensitivity rows exist for that reason.
6. **Single-tenant traces.** A5's co-batching regime — the one `plan_next_steps.md`
   argues is most promising — is *not* covered by these four cells; the hash-based
   ownership model is a poor stand-in for real co-batching.
7. **The control is one search at one budget.** `bottleneck_search_layer` is two
   sweeps and 1500 evals/layer; a stronger search could change F3's sign. That is
   exactly why F3 is worded as "this search does not win yet".
8. **The kernel-level compute term is absent.** Every completion-time number is the
   communication component; the sensitivity rows make the omission visible rather
   than hiding it in an assumption.
9. **The co-design search assumes reconfiguration is free** (V.5), so even its
   partial recovery in F3 is not yet a deployable claim.

---

## Part VII — What would change the conclusion, ranked

1. **Measure the ownership model**, or report the envelope over ownership models.
   Nothing else in this list moves the OCS number as much (148× across
   ownership × DP; F8).
2. **Fix the "oracle".** Replace greedy `plan_circuits` with a real degree-bounded
   b-matching optimum, then re-run the static-vs-oracle argument. `value_of_
   prediction` can be negative today, so that argument is currently unfounded.
3. **Sweep `DispatchMode`.** Cheap, and it is the axis on which affinity's
   mechanism either exists or does not.
4. **Strengthen the co-design search** (F3): the control has 7–8 % of headroom on
   the EPS substrate that a winning co-design must capture first.
5. **Port budget as a stated assumption** (F5): the substitution is partly an
   artefact of 2 ports/rank. Report the envelope, as with σ.
6. **Move to the untested corner**: EP > pod with multi-tenant co-batching, where
   `f = 0` stops being structural and where A5 already measured contention.

---

## Part VIII — Bugs found (appendix)

Each was caught by a check that now exists as a test.

| # | bug | how it was caught |
| --- | --- | --- |
| 1 | byte contraction returned `[dst, src]`, swapping egress/ingress bottleneck | oracle vs `evaluate` mismatch on an asymmetric instance |
| 2 | NVLink and NIC drain summed instead of maxed | mismatch on a rank hot on both resources |
| 3 | MILP coefficient array mis-ordered vs its column order → bound of 93 against a provable floor of 512 | averaging invariant `Σ_r load_r = Σ_e reach_e` |
| 4 | `CellTable.select()` keeps the parent's `runs` list, so `n_runs` (87) inflated throughput by 5.4× | throughput sanity check |
| 5 | per-layer eval budget starved later layers; alternating loop could return a worse round than it started | history logging |
| 6 | a crashed late stage discarded every earlier stage's results | incremental write-back |

---

## Part IX — Reproducing

```bash
# the four cells + regime/port/reconfig envelope + completion time + bounds
python3 scripts/ocs_four_cell.py --workload logs/workload/qwen36  --world-size 32
python3 scripts/ocs_four_cell.py --workload logs/workload/whittle --world-size 32
python3 scripts/ocs_four_cell.py --workload logs/workload/qwen15  --world-size 30

# the audit of V.1/F8: ownership model x DP size (all three models)
python3 scripts/source_model_sensitivity.py --workload logs/workload/qwen36  --world-size 32
python3 scripts/source_model_sensitivity.py --workload logs/workload/whittle --world-size 32
python3 scripts/source_model_sensitivity.py --workload logs/workload/qwen15  --world-size 30

# the audit of V.7/F10: dispatch semantics
python3 scripts/ocs_four_cell.py --workload logs/workload/qwen36 --world-size 32 --stage dispatch

# multi-layer bounds (a single layer is not representative; see F4)
python3 scripts/ocs_four_cell.py --workload logs/workload/qwen36 --world-size 32 \
    --stage milp --milp-layers 3

# markdown tables from the JSON
python3 scripts/four_cell_summary.py outputs/four_cell/qwen36.ws32.json \
    outputs/four_cell/whittle.ws32.json outputs/four_cell/qwen15.ws30.json

# tests: objective fidelity vs cost_model.evaluate, MILP bounds, completion maths
python3 -m unittest discover -s tests -v
```

| artefact | contents |
| --- | --- |
| `scripts/ocs_four_cell.py` | the experiment (stages: `fourcell`, `regime`, `dispatch`, `timing`, `milp`) |
| `scripts/source_model_sensitivity.py` | the V.1/F8 audit (monkey-patches `token_rank`; modifies nothing) |
| `scripts/four_cell_summary.py` | JSON → markdown |
| `src/eval/promote_aware.py` | the co-design objective and alternating search |
| `src/eval/milp_bound.py` | LP/MIP and combinatorial bounds, with their caveats |
| `src/eval/completion.py` | TTFT/ITL/throughput composition |
| `tests/test_promote_aware.py` | 33 tests; no captured data required |
| `outputs/four_cell/*.json`, `REPORT.md` | raw results and generated tables |
| `outputs/four_cell/source_sensitivity.*.json` | the F8 ownership × DP sweep |
| `logs/four_cell/*.log` | full run logs |

`src/eval/placement_opt.py` is the only source file modified: a
`promote_aware_layer` registration branch and three keyword-only parameters that
default to `None`, so every existing caller is unaffected.
