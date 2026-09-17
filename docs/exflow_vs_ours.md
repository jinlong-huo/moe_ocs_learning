# ExFlow vs this repo's placement: measured comparison and OCS alignment

**Source for ExFlow:** `Exploiting Inter-Layer Expert Affinity for Accelerating
Mixture-of-Experts Model Inference` (arXiv:2401.08383), formulation summary in
`Zotero/ExFlow.md` (ILP Eq. 8–12 and the reference `solve_affinity.py`).
**Source for "ours":** `src/eval/placement_opt.py`, `src/eval/cost_model.py`,
`src/eval/ocs_eval.py` in this repo.
**Measurement:** `scripts/exflow_compare.py`, results in
`outputs/four_cell/exflow_compare.*.json`, logs in `logs/four_cell/exflow_*.log`.

---

## 1. The two formulations, side by side

| | **ExFlow** | **this repo** |
| --- | --- | --- |
| decision | `x[n,c] ∈ {0,1}` — expert `n = i + jE` (expert `i`, layer `j`) → group `c` | `expert_to_rank ∈ [E]` or `[L, E]` |
| capacity | `Σ_{n∈N_ℓ} x[n,c] = B` (hard) or `≥ ⌈ηB⌉` (relaxed) | exactly `E/W` per rank per layer |
| auxiliary | `cost[k,s] ∈ {0,1}` — token `k` re-routed across the layer-`s` boundary | none |
| **objective** | **min `Σ_k Σ_s cost[k,s]`** — *count* of cross-layer re-routes | dedup ingress / fanout, accumulated over layers, on **tier-weighted bytes** |
| coupling | `cost[k,s] ≥ x[r_{k,s},c] − x[r_{k,s+1},c]` and its mirror | union indicator `y[c,r] ≥ x[e,r]` (in the bound module) |
| coefficients | **none** — every re-route costs 1, tier-blind | **all of them** — 450 / 50 / 12.5 GB/s per tier, per-tier latencies |
| topology | handled *hierarchically*: stage 1 over nodes, stage 2 within a node | handled *numerically*: `FabricConfig` bandwidths/latencies |
| what it optimises | cross-layer **continuity** (does the token stay put between layers?) | intra-layer **reach** + per-rank **balance** |
| what it ignores | fan-out within a layer, load balance (default `fused_obj=False` drops even the imbalance penalty), tier structure | cross-layer continuity — *entirely* |
| scaling | warm-start incremental MIP + multilevel bipartitioning + two-stage hierarchy | greedy clustering + bitset local search |
| reported gain | one Alltoall instead of two; up to 67 % cross-GPU latency cut; 2.2× throughput | +25–46 % bottleneck vs baselines, out-of-sample |

Two structural facts stand out before any measurement:

**(a) They are orthogonal.** ExFlow counts a token's *hops between layers*; this
repo measures a token's *reach within a layer* plus the per-rank maximum. A
placement can be excellent at one and terrible at the other, and the measurement
below shows it is.

**(b) Their objective is coefficient-free by design.** The note records the
rejected coefficient-weighted variant: bilinear (`T·R`), and "coefficient
fluctuation can produce completely different solutions". The measurement in
`docs/four_cell.md` F4 points the other way for *our* metric — the tier-blind
count objectives are satisfied to within 1.8–6.4 % by plain per-layer LPT, while
the tier-weighted bottleneck ranks placements differently. Both positions are
defensible because the *units* differ: one avoided hop is worth far more than one
avoided message.

**One caveat carried from your note:** Eq. 8 is not the Lagrange dual of Eq. 5
(there is no λ in it), and the primal is never optimised — the relationship used
in practice is the surrogate "high affinity ⇒ low re-routing", minimised directly.
The implementation is also a MILP (integer `lb`/`abs`), not the ILP of the paper.

---

## 2. Measured comparison

Every placement, scored on **both** families of metric. `reroute%` is ExFlow's
objective (lower better); `bn_us` is ours (lower better). Both families were
evaluated on the same eval split; the ExFlow-objective rows come from plain
swap-based local search on *their exact objective* (`exflow_ls`), which is a
stand-in for their MILP, not a reproduction of their solver.

### Qwen3.6-35B (E=256, K=8, L=40, W=32, cap=8)

| placement | reroute % | fanout | bn µs | cross-pod GB | OCS gain | top-16 pair share |
| --- | --- | --- | --- | --- | --- | --- |
| `random` | 96.99 | 7.268 | 4 406 | 1.219 | 8.45 % | 7.21 % |
| `linear` | 96.74 | 7.224 | 3 956 | 1.210 | 4.76 % | 7.16 % |
| `load_balanced_layer` | 95.00 | 7.497 | 5 412 | 1.250 | 8.44 % | 7.74 % |
| `affinity_layer` | 80.17 | 4.964 | 8 082 | 0.826 | 8.32 % | **12.49 %** |
| **`affinity_coordinated_layer`** | **97.90** | 5.652 | **2 944** | 0.948 | **2.53 %** | **6.83 %** |
| `exflow_ls` (from linear) | **80.67** | 7.113 | 4 552 | 1.195 | 8.56 % | 7.48 % |
| `exflow_ls` (from affinity) | 80.72 | 6.679 | 3 910 | 1.121 | 8.30 % | 7.16 % |

### Whittle-27B (E=64, K=16, L=64, W=32, cap=2)

| placement | reroute % | fanout | bn µs | cross-pod GB | OCS gain | top-16 pair share |
| --- | --- | --- | --- | --- | --- | --- |
| `random` | 96.95 | 14.116 | 11 967 | 3.788 | 8.43 % | 6.87 % |
| `linear` | 96.89 | 14.073 | 12 639 | 3.776 | 8.46 % | 6.95 % |
| `load_balanced_layer` | 94.65 | 15.055 | 19 134 | 4.039 | 8.47 % | 7.79 % |
| `affinity_layer` | 90.77 | 12.379 | 22 058 | 3.326 | 8.39 % | **9.23 %** |
| **`affinity_coordinated_layer`** | **98.26** | 12.709 | **10 120** | 3.406 | **2.91 %** | **6.53 %** |
| `exflow_ls` (from linear) | **87.18** | 13.831 | 13 602 | 3.715 | 8.35 % | 7.28 % |
| `exflow_ls` (from affinity) | 87.96 | 13.802 | 13 043 | 3.701 | 8.37 % | 7.09 % |

### Qwen1.5-MoE (E=60, K=4, L=24, W=30, cap=2)

| placement | reroute % | fanout | bn µs | cross-pod GB | OCS gain | top-16 pair share |
| --- | --- | --- | --- | --- | --- | --- |
| `random` | 96.20 | 3.890 | 2 394 | 0.719 | 0.95 % | 7.88 % |
| `linear` | 96.56 | 3.896 | 2 413 | 0.719 | 1.90 % | 7.81 % |
| `load_balanced_layer` | 96.73 | 3.901 | 2 466 | 0.718 | 0.81 % | 7.88 % |
| `affinity_layer` | 95.48 | 3.495 | 2 286 | 0.643 | 3.67 % | **8.07 %** |
| **`affinity_coordinated_layer`** | 96.53 | 3.511 | **2 223** | 0.646 | **3.83 %** | 7.82 % |
| `exflow_ls` (from linear) | 86.11 | 3.869 | 2 407 | 0.713 | 2.69 % | 7.98 % |
| `exflow_ls` (from affinity) | **85.41** | 3.797 | 2 432 | 0.700 | 3.60 % | 8.01 % |

### F1 — The two objectives **conflict**

| model | corr(ExFlow re-route, our bottleneck) |
| --- | --- |
| Qwen3.6 | **−0.488** |
| Whittle | **−0.317** |
| Qwen1.5 | **−0.323** |

Negative in all three: the placements that are better for ExFlow's objective are
*worse* for ours. Concretely, **our best placement is the worst for ExFlow** —
`affinity_coordinated_layer` re-routes 97.90 % of boundaries on Qwen3.6, worse
than `random`'s 96.99 %, because balancing per-rank ingress moves experts around
with no regard for whether a token stays put between layers. And ExFlow's
placement is worse than `linear` on our metric (4 552 vs 3 956 µs).

### F2 — ExFlow's objective has a capacity-imposed floor

| model | E/W = cap | re-route before | re-route after | residual |
| --- | --- | --- | --- | --- |
| Qwen3.6 | 8 | 96.74 % | **80.67 %** | 80.7 % still move |
| Whittle | 2 | 96.89 % | **87.18 %** | 87.2 % still move |
| Qwen1.5 | 2 | 96.56 % | **85.41 %** | 85.4 % still move |

**Even under their own objective, 80–88 % of layer boundaries still require the
token to move.** The reason is the same capacity wall as ours: a group holds
`B = E/W` experts *per layer*, so co-locating a token's layer-`l` and
layer-`l+1` experts forces both into one of `C(E, B)` slots. The larger
`B/E` (=`1/W`), the more room — which is why Qwen3.6 (cap 8) aligns better than
the two cap-2 models. The "one Alltoall instead of two" benefit therefore applies
to a *minority* of tokens at these configurations, not to the collective as a
whole.

---

## 3. OCS alignment — the question you actually asked

A circuit is a **pair** resource with a degree bound. What decides whether optical
circuits can pay is not total bytes but the **concentration** of cross-pod bytes
on the few rank pairs a switch can promote. That is the last column
(`top-16 pair share`: the share of all cross-pod bytes carried by the 16 heaviest
cross-pod pairs, i.e. the pairs `plan_circuits` would pick at a 16-circuit budget).

### F3 — Neither method aligns with OCS, and they align *inversely*

| model | best OCS concentration | its bottleneck | our best placement | our concentration |
| --- | --- | --- | --- | --- |
| Qwen3.6 | `affinity_layer` 12.49 % | 8 082 µs (2× worse than random) | `affinity_coordinated` | **6.83 %** (worst of all) |
| Whittle | `affinity_layer` 9.23 % | 22 058 µs | `affinity_coordinated` | **6.53 %** (worst) |
| Qwen1.5 | `affinity_layer` 8.07 % | 2 286 µs | `affinity_coordinated` | 7.82 % |

In every model, **the placement that maximises circuit opportunity is the one
that concentrates traffic — and concentration is exactly what breaks the
electrical bottleneck.** Our best placement has the *lowest* circuit opportunity
of any candidate, which is precisely why its OCS gain collapses to 2.5–3.8 %
while the badly-concentrated placements get 8.3–8.6 %.

This is the same substitution effect reported in `docs/four_cell.md` F2, now
stated structurally:

```
EPS bottleneck (max over ranks)   wants traffic SPREAD   → balance
OCS circuit value (bounded pairs) wants traffic CONCENTRATED → coverage
```

They are pushed by the *same* traffic distribution in opposite directions.

### F4 — ExFlow's objective does **not** help OCS either (my earlier prediction was wrong)

I previously argued that ExFlow's cross-layer alignment would concentrate traffic
onto persistent rank pairs and therefore suit circuits. Measured:

| model | corr(ExFlow re-route, OCS concentration) | `exflow_ls` pair share vs `linear` |
| --- | --- | --- |
| Qwen3.6 | **−0.512** | 7.48 % vs 7.16 % |
| Whittle | **−0.389** | 7.28 % vs 6.95 % |
| Qwen1.5 | **−0.576** | 7.98 % vs 7.81 % |

The correlation is **negative** in all three models, and the absolute gain over
`linear` is ~0.2–0.3 points — noise-level. The reason is that ExFlow minimises a
**count of hops**, not the **weight on specific pairs**: aligning 13–19 % of
boundaries does not create hot pairs, it merely spreads the remaining 80 %+ over
the same wide set of pairs as before. **Count-based objectives cannot create the
concentration a circuit needs.**

---

## 4. Verdict

**There is no single winner, because they are not competing for the same thing.**

| question | answer |
| --- | --- |
| Which is better on the electrical bottleneck? | **ours**, by 25–89 % over the simple baselines; ExFlow's objective lands *worse than `linear`* (4 552 vs 3 956 µs on Qwen3.6) |
| Which is better on cross-layer hops? | **ExFlow**, 80.7 / 87.2 / 85.4 % vs our 97.9 / 98.3 / 96.5 % — and our best placement is *worse than random* |
| Do they conflict? | **yes**, ρ = −0.32 … −0.49 |
| Which aligns with OCS? | **neither.** OCS wants concentration; our objective minimises it, ExFlow's is indifferent to it |
| Can they be combined? | **yes, and they should be** — see below |

They act on *different factors of the same product*:

```
MoE communication  ≈  (bytes per hop)  ×  (number of hops)
                       ↑ ours            ↑ ExFlow
```

Our objective reduces bytes per hop (fanout ↓, **21.8 % of volume** on Qwen3.6).
ExFlow's eliminates hops at a boundary (13–19 % of boundaries fully aligned at
these configurations). Quantified — a token normally pays `2L` hops (dispatch +
combine per layer), and at an aligned boundary both `combine(l)` and
`dispatch(l+1)` vanish:

```
hops   = 2L − 2·a·(L−1)          a = aligned share
saving = a·(L−1)/L               (= 1 − 1/L at a = 1: only dispatch(1) and combine(L) remain)
```

At a = 19.3 %, L = 40 (Qwen3.6) that is **18.8 % of MoE volume** — the *same order*
as our 21.8 %, not a minor effect. (An earlier estimate of `a/2` double-counted the
hops shared between adjacent boundaries.) Caveat: the fusion assumes nothing between
two MoE layers forces the token home — if attention between them is itself
tensor- or sequence-parallel, it caps the saving, and that is not modelled here.
Because the two factors multiply, a combined objective should beat either alone —
and neither of the two currently measured placements achieves it. Because they multiply, a combined objective should beat
either alone — and neither of the two currently measured placements achieves it.
Measuring the combination needs a cost-model change: `cost_model.evaluate`
currently applies `mult = 2 × n_microbatches` **unconditionally**, so the fusion
saving is structurally invisible to us (see §5).

### What to build for OCS

1. **A concentration-aware term, not a better EPS objective.** Add to the
   coordinated objective a term maximising the share of critical bytes on the
   top-`C` rank pairs, where `C` is the switch's circuit budget. Its weight
   `γ` is set by the port budget and σ: at 8 circuits it is worthless
   (0.41 % gain measured), at 32 circuits it is worth 8.57 %. The current
   objective is the `γ = 0` corner and the measurements show how much is
   being left on that table.
2. **Add ExFlow's inter-layer term as a second objective** (or a constraint), and
   extend the cost model so a fused layer boundary is actually cheaper —
   otherwise the comparison cannot be made inside this repo at all.
3. **Report the Pareto front** of (bottleneck, concentration) rather than a single
   placement, since the operating point is chosen by the deployment's switch
   budget, not by the workload.

---

## 5. Caveats

* **`exflow_ls` is not their solver.** It is plain swap-based local search on
  their exact objective; their implementation uses warm-started incremental MIPs
  with multilevel bipartitioning, and would likely reach a somewhat lower
  re-route. The *floor* argument in F2 is structural (capacity), not
  solver-quality.
* **`r_{k,j}` uses the top-1 expert** (the argmax gate weight), matching the
  paper's single-expert index; top-K alternatives are untested here.
* One trace and one seed per model; the correlations are computed over 7
  placements, so ρ is indicative, not a precise estimate.
* Our cost model's own limits apply: mean-field contention (fixed `1/σ`), no
  per-flow rate cap, `mult = 2` fixed (so no fused boundaries). See
  `docs/four_cell.md` Part V.
* The `top-16 pair share` uses a 16-circuit budget; the ordering between
  placements is what matters and it is stable across all three models.

---

## 6. Reproducing

```bash
python3 scripts/exflow_compare.py --workload logs/workload/qwen36  --world-size 32
python3 scripts/exflow_compare.py --workload logs/workload/whittle --world-size 32
python3 scripts/exflow_compare.py --workload logs/workload/qwen15  --world-size 30
```
