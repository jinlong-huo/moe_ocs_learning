# Research validity assessment — routing → affinity → placement → OCS

Status: written against measured results from two real MoE models captured in this repository. Every number below is reproducible with the commands in §7. Where a claim rests on external literature rather than measurement it is
marked **[lit]** and flagged if it needs verification.

Artifacts:

* `logs/workload/qwen15/` — Qwen1.5-MoE-A2.7B-Chat-4bit, E=60, K=4, 24 MoE
  layers, 112 sequences, 308,544 routing cells
* `logs/workload/qwen36/` — Qwen3.6-35B-A3B-4bit, E=256, K=8, 40 MoE layers,
  87 sequences, 356,440 routing cells

---

## 1. Verdict in one paragraph

The direction is defensible and there is a real, large effect to report — but **not the effect the original framing claimed**. The rank×rank all-to-all traffic matrix is 99.9 % rank-1, so there is essentially no *pairwise* traffic structure for a topology optimiser to exploit; the expert affinity graph does not shape the traffic matrix. What the affinity graph *does* do is reduce total dispatch volume by coalescing each token's destinations, and that is worth **32.8 % of the all-to-all critical path out-of-sample** on the 256-expert
model — but only under a formulation the original code did not have (per-layer
placement, jointly balanced across layers). The naive version of the same idea
is **2× worse than random**. OCS, meanwhile, is inapplicable at the scale these
models occupy under realistic pod sizes, and where it is applicable its circuit
plan is too unstable (Jaccard 0.09 across request windows) for a
millisecond-class switch to track. The publishable contribution is therefore a
**placement** result with a carefully bounded OCS feasibility analysis — not an
OCS architecture result.

---

## 2. Which original assumptions survive

### 2.1 Correct

**A1 — Logical routing is decoupled from placement and topology.** Holds, and
now holds *structurally*: `src/eval/cost_model.py` and
`src/eval/placement_opt.py` consume an immutable `CellTable` and have no path
to alter a routing decision. Verified across 15 (placement × topology)
configurations: token→expert match rate 1.000, cell count constant, while the
bottleneck moves.

**A2 — The empirical boundary of invariance is gate determinism, and it is
tight.** Four identical-prompt repeats under greedy decoding give a
**bit-exact** match rate of 1.000 on both models. This matters: it means the
noise floor is zero, so every similarity number below is signal, not
quantisation flutter. (The pre-existing comment in `src/serving/affinity.py`
claiming a "~5–6 % marginal-expert flip on Metal" is not reproduced by the MLX
capture path used here; that tolerance can be dropped for this backend, but the
repeat set must stay in the suite because it is the only thing that
*establishes* the floor rather than assuming it.)

**A3 — Placement changes the cost of a fixed routing.** Holds. Bottleneck
spread across placements and perturbations: 3.7 % (Qwen1.5, EP=15), 13.3 %
(Qwen3.6, EP=32) for arbitrary placements; up to 46 % between the best and
worst *optimised* placements.

**A4 — Workload similarity correlates with routing similarity.** Holds, and far
more strongly than the original two-prompt comparison could show. Leave-one-
sequence-out nearest-centroid decoding of the workload category from the
routing signature alone:

| model       | classes | accuracy          | permutation null | naive chance | p     |
| ----------- | ------- | ----------------- | ---------------- | ------------ | ----- |
| Qwen1.5-MoE | 12      | **62.5 %**  | 6.4 %            | 8.3 %        | 0.005 |
| Qwen3.6-35B | 12      | **93.75 %** | 6.0 %            | 8.3 %        | 0.005 |

**A5 — The driver is semantics, not surface form.** This is the controlled
result the two-prompt design structurally could not produce. Routing-signature
cosine on Qwen1.5:

| contrast                                                            | cosine          | reading                      |
| ------------------------------------------------------------------- | --------------- | ---------------------------- |
| identical prompt (noise floor)                                      | 1.000           | measurement floor            |
| **paraphrase** (same meaning, minimal shared words)           | **0.916** | 63 % of the way to the floor |
| length ladder (same topic, 3 lengths)                               | 0.917           | not a length artifact        |
| same category                                                       | 0.822           | —                           |
| **lexical control** (same template & words, different domain) | **0.817** | only 20 % of the way         |
| different category                                                  | 0.771           | baseline                     |

Paraphrase (0.916) ≫ lexical control (0.817) with non-overlapping 95 % CIs.
Verdict: `semantic_dominant` on both models. Routing tracks meaning.

### 2.2 Conditionally correct — the conditions are load-bearing

**B1 — "Affinity-aware placement reduces communication cost."** True, but only
under two conditions the original code violated:

*Condition 1: per-layer scope.* Expert ids are per-layer namespaces. Measured
cross-layer Pearson r between per-layer expert-load vectors: **0.0037**
(Qwen1.5, 276 layer pairs), **0.0106** (Qwen3.6, 780 pairs). Layer-pooled
("global") affinity placement buys 3.2 %–7.4 %; per-layer buys 5–10× more.

*Condition 2: the layers must be balanced jointly.* The all-to-all critical
path is `max_s Σ_layers ingress_l(s)` — a rank's ingress accumulated over every
MoE layer. Optimising each layer independently minimises
`Σ_l max_s ingress_l(s)`, a different quantity, and independent optimisers pile
their hot spots onto the same ranks. Measured consequence on Qwen3.6 (EP=32,
held-out categories):

| placement                                   | fan-out         | network bytes     | ingress imbalance | bottleneck vs random |
| ------------------------------------------- | --------------- | ----------------- | ----------------- | -------------------- |
| random (the correct null)                   | 7.268           | 3.86 GB           | 1.258             | 0 %                  |
| linear (the deployed default)               | 7.241           | 3.85 GB           | 1.158             | +4.6 %               |
| load-balanced (LPT on pooled load)          | 7.269           | 3.87 GB           | 1.078             | +11.1 %              |
| affinity, layer-pooled                      | 6.579           | 3.50 GB           | 1.267             | +7.4 %               |
| **affinity, per-layer, independent**  | **4.846** | **2.58 GB** | **3.679**   | **−100.6 %**  |
| load-balanced, per-layer, independent       | 7.480           | 3.98 GB           | 1.554             | −31.2 %             |
| **affinity, per-layer, coordinated**  | —              | —                | —                | **+32.8 %**    |
| direct bottleneck optimisation, coordinated | —              | —                | —                | +31.1 %              |
| adversarial (max fan-out)                   | 7.325           | 3.90 GB           | 2.810             | −129.9 %            |

Read the fifth row carefully: naive per-layer affinity clustering cuts fan-out
by 33 % and volume by 33 %, and is **twice as slow**, because it co-locates the
popular experts (they co-occur with everything) and triples the ingress
imbalance. This is precisely the configuration the original `payoff` section
would have reported as a success, because its metric —
`intra_rank_affinity_fraction` — rises monotonically with exactly the
clustering that destroys the collective.

**B2 — "Affinity reduces traffic."** Conditional on the dispatch kernel. Under
**replicated** dispatch (one copy per selected expert; classic DeepSpeed-MoE /
Megatron all-to-all) total dispatch volume is `N·K·H·dtype` **exactly
independent of placement** — verified: 0.000 % spread across all placements and
perturbations. Placement can then only relocate bytes between tiers and ranks.
Only dedup-capable kernels (one copy per destination rank/node) let placement
change volume at all. Any "affinity reduces traffic" claim must state the
kernel assumption.

**B3 — Load balancing matters, and is *not* affinity.** Qwen3.6 per-layer load
skew: max expert load **10.54×** uniform, Gini 0.59, top ⅛ of experts carry
**49.3 %** of tokens. Pooled across layers those same numbers read 1.86× and
18.8 % — a 5.7× understatement, purely an artifact of pooling independent
namespaces. Load balancing alone buys 11.1 %, i.e. a third of the total
achievable gain, with no affinity graph at all. Any affinity result must be
reported against this baseline, not against `linear`.

**B4 — OCS applicability.** Conditional on cross-domain oversubscription
existing. See §4.

### 2.3 Incorrect — must be dropped or restated

**C1 — "The expert affinity graph is the workload signal that shapes the
network traffic matrix." This is false.** Measured `rank1_energy` (fraction of
the traffic matrix's spectral energy in its first singular value): **0.9991 –
0.9996** across every placement tested, on both models. The matrix is an outer
product of (tokens per DP rank) × (dedup reach per EP rank).

The reason is structural, not incidental: batch schedulers shard tokens across
DP ranks independently of token content, so the source side of the matrix
carries no routing information, and `E[T(d,s)] = N_d · reach(s)`. Expert–expert
co-activation is invisible in the matrix.

Consequence for framing: the mechanism is **not** traffic-matrix-aware topology
engineering. It is **destination coalescing** — reducing how many distinct
ranks each token must reach. The affinity graph is a useful *pairwise
relaxation* of that hypergraph objective, nothing more. A paper framed as
"convert the affinity graph into a network traffic pattern and optimise the
topology for it" would be making a claim the data contradicts.

**C2 — Layer-pooled expert statistics are invalid.** Pooling L independent
expert-id namespaces averages L unrelated distributions and drives every
distributional metric toward uniform. This fully explains the original result
that two semantically distant prompts had a Jensen–Shannon divergence of
**0.0054 bits** — the metric was saturated, not the routing identical. Affected
and to be discarded: `load_entropy_norm`, `top5_expert_share`,
`layer_diversity_mean_js`, `affinity_strength_offdiag`, pooled `co_activation`,
and the `js_divergence` used in `pairwise_metrics`. Pooled affinity remains
legitimate for exactly one purpose: as the input to a *layer-shared* placement.

**C3 — `plan_hit_rate = 1.0` was saturation.** With K=8 over E=256 and ~2500
cells, a single 68-token prompt touches 32–70 % of experts *per layer* and
100 % *pooled*. "Trace A's expert set covers trace B's cells" is therefore
trivially 1.0 and carries no information.

**C4 — Routing entropy as a model-comparison metric.** Pooled entropy is near
its maximum by construction (C2), so it cannot separate models. Measured:
pooled `max_over_uniform` = 1.11 on Qwen1.5 — i.e. "perfectly balanced" — while
per-layer it is 1.64, and on Qwen3.6 1.86 pooled vs **10.54** per-layer. The
original `model_variation` test declared divergence on differences of 0.005 in
this quantity; that is noise in a saturated statistic. Replaced by: per-expert
category-KL (Qwen1.5 **1.9 %** of the log₂(12) bound, Qwen3.6 **23.7 %** — a
12× real difference between the two models, which is the kind of separation a
model comparison should be built on), plus out-of-sample category decoding.

**C5 — The OCS cost model made OCS unable to win.** `src/ocs/topology.py`
documents `alpha_ocs = alpha_eps + T_reconfig`, `beta_ocs = beta_eps`. A hot
circuit is then *exactly* as fast as electrical and a cold one strictly slower,
so no experiment built on it can show a benefit that is not an artifact of the
scheduler hiding reconfiguration. Replaced by the physically grounded
mechanism: a circuit **removes oversubscription** for the pair it serves (tier
promotion CROSS_POD → OPTICAL), and is worthless for any pair that is not
contended. `Topology.tier_matrix` now enforces exactly that.

**C6 — Per-peer byte accounting was wrong by ~W.** The old
`Transport._inject_delay` passed the *entire* padded send buffer's byte count
to every destination and took the max, overstating the bandwidth term by
roughly `world_size`, and growing with padding. Replaced by a genuine per-pair
byte matrix derived from routing cells.
**FIXED** in the legacy data plane: `scatter_tokens`/`gather_tokens` now pass
a per-destination byte map through `Transport.all_to_all(pair_bytes=...)`, and
every delay mode (mixed EPS+OCS, OCS-only, topology) charges each destination
only its own bytes.

**C7 — Bandwidth unit error, 8× on two of three tiers.** `TopologyConfig`
fields are named `*_bandwidth_gbps`, the docstrings cite Gb/s ("400 Gb/s per
port", "200 Gb/s per port"), and `get_pairwise_delay` divides by
`bw * 1000.0`, i.e. treats them as GB/s. Inter-node tiers were therefore
modelled 8× faster than the hardware cited. Same constants are duplicated in
`src/runtime/placement.py:357` and `src/runtime/worker.py:129`.
**FIXED** everywhere: fields renamed to `*_bandwidth_gbs` with honest GB/s
values (900 / 50 / 25), configs and the placement-manifest serializer
(`bandwidth_gbs`) updated, `src/ocs/circuit.py` flat path renamed and
re-defaulted.

**C8 — No congestion, capacity, or collective structure.** Cost was a
stateless per-pair `α + β·n` maxed over peers — an infinitely wide NIC. An
all-to-all cannot complete before its busiest egress *and* ingress port drains.
This is not a refinement: it is the difference between the naive per-layer
affinity placement looking like a 33 % win (volume) and being a 100 % loss
(critical path). Replaced by a per-rank egress/ingress bottleneck over per-tier
capacity.

**C9 — `Topology.get_max_tier` bug.** `my_rank = participating_ranks[0]`, and
its only caller passes `list(range(world_size))`, so `my_rank` is always 0 and
every rank is charged the worst tier *from rank 0's viewpoint*. The `my_rank`
argument to `get_delay` is dead.
**FIXED**: `get_max_tier` takes an explicit `viewpoint_rank` (or computes the
max over all pairs when none is given), and `get_delay` now passes its own
`my_rank`.

**C10 — Everything was in-sample.** The original `payoff` fitted the affinity
graph and evaluated the placement on the same trace. Measured generalisation
gaps here are small (0.85–1.54 % on fan-out) but that is a *finding*, not
something to assume. All placement numbers now come from fit/eval splits, and
two are reported: within-category (easier) and leave-whole-categories-out
(harder).

**C11 — Balancing expert selection counts is the wrong balance objective.**
Under dedup dispatch a rank owning one very popular expert receives a message
from nearly *every* token regardless of what else it owns, so arriving message
counts saturate while selection counts do not. LPT on per-layer selection
counts made the Qwen3.6 bottleneck **31 % worse** than random. The correct
quantity is `dedup_ingress` (`src/eval/placement_opt.py`), which is why the
coordinated optimiser works.

**C12 — `pre_config` wastes the circuit budget.** `src/ocs/circuit.py:136`
does not filter `src == self.rank`, unlike `OnlineAffinityController`. Since
the plan is global, a rank burns its port budget on circuits `(other, dst)`
that its own transport never queries. With `max_circuits=1` the entire budget
can be wasted. Also bypasses `establish()`, so preset-mode reconfiguration cost
is exactly zero rather than merely off the critical path.
**FIXED**: `pre_config` filters `src == self.rank` (the pool now knows its
owning rank), respects the budget per own circuits, and routes through
`establish()` so the reconfiguration time is accounted in the pool metrics
while staying off the inference critical path.

**C13 — Two prompts cannot separate three hypotheses.** The original design
compared "why MoE needs routing" against "how gradient descent works". These
differ in semantics *and* lexis *and* length simultaneously, so no outcome
could distinguish semantic routing from surface-form routing. This is what
§2.1/A5 fixes with paraphrase and lexical-control sets.

---

## 3. The claim that is actually defensible

> For a fixed model state, routing configuration and input sequence, the logical
> token→expert assignment is independent of expert placement and network
> topology; placement and topology determine only the communication cost that
> routing induces. That cost is **not** governed by pairwise expert affinity —
> the induced rank×rank traffic matrix is empirically rank-1 (≥0.999 spectral
> energy) because tokens are sharded independently of content. It is governed
> by two quantities that *are* recoverable from routing traces: how many
> distinct destination ranks each token must reach (destination fan-out), and
> how unevenly arriving messages concentrate on ranks (dedup ingress skew).
> Expert co-activation is a useful pairwise relaxation of the first, and is only
> beneficial when optimised **per layer** and **jointly balanced across
> layers**; applied naively it is worse than random placement. Optical circuit
> switching can improve only the fraction of this traffic that crosses an
> oversubscribed domain, which at realistic pod sizes is zero for the expert-
> parallel degrees current MoE inference deployments use.

That claim is supported by every number in this document, survives the negative
results, and is more interesting than the original because it identifies *which*
signal in the routing trace is the exploitable one and why the obvious candidate
is not.

---

## 4. Is the OCS motivation realistic?

### 4.1 Where MoE inference actually runs **[lit]**

* Expert placement is quasi-static: changing it means moving expert weight
  tensors (GB per expert-shard), so it is a deployment-time or
  epoch-boundary decision — not something to re-optimise per request. This is
  a point in favour of the placement result: a placement fitted offline on
  traces is exactly the deployable artifact.
* MoE inference is routinely expert-parallel across multiple nodes, with
  reported EP degrees in the tens to low hundreds. **[lit — verify exact
  figures before citing]**
* DeepSeek-V3-class models use 256 routed experts with top-8 gating and
  **node-limited routing**: each token is constrained to at most ~4 nodes,
  explicitly to bound all-to-all cost. **[lit]** This is important corroboration
  — it is the same objective as `mean_fanout` here, already recognised as the
  cost driver in production, and implemented as a *routing* constraint rather
  than a placement one. Our result is the placement-side dual of it.
* Dedup-at-node-granularity dispatch kernels (DeepEP class) forward one copy
  per destination node and fan out on NVLink inside the node. **[lit]** This is
  why `DispatchMode.DEDUP_NODE` is modelled and why B2's condition is
  satisfiable in practice.

### 4.2 The pod-size problem — an honest negative

`hierarchy_for(..., "realistic")` uses 8 GPUs/node and 32 nodes/pod, so a pod
holds 256 GPUs. Measured tier composition:

| EP degree | nodes | pods (realistic) | cross-pod pairs |
| --------- | ----- | ---------------- | --------------- |
| 15        | 2     | 1                | **0**     |
| 32        | 4     | 1                | **0**     |
| 64        | 8     | 1                | **0**     |
| 256       | 32    | 1                | **0**     |

At realistic pod sizes, **no expert-parallel degree these models can reach
produces any cross-pod traffic at all**, so there is nothing for an optical
circuit to promote. `ocs_comparison` returns `applicable: False` with that
reason rather than manufacturing a number. This is the single most important
correction to the OCS motivation, and it is why §1 calls this a placement paper.

To study OCS at all we must adopt `multi_pod` (8 GPUs/node, 2 nodes/pod = a
16-GPU pod). That is a deliberate modelling choice representing a
cost-constrained or rail-partitioned cluster where crossing a domain is
expensive — **it is not a description of a hyperscale pod**, and must be
labelled as such in any write-up. Under it, Qwen3.6 at EP=32 gives 4 pods and
512 cross-pod rank pairs carrying 1.98 GB of the 3.86 GB total.

### 4.3 The timescale problem — the second honest negative

Reconfiguration reference points **[lit]**: MEMS beam-steering
cross-connects of the class deployed in production datacenter fabrics
reconfigure in the **millisecond** range and are used at job or slice
granularity; research designs (tunable-laser/AWGR, SOA) reach **sub-microsecond**
but are not deployed. One MoE all-to-all in our cost model is **tens to
hundreds of microseconds**.

Measured circuit-plan stability on Qwen3.6 (EP=32, multi_pod):

| window            | traffic-matrix cosine | **circuit-plan Jaccard** |
| ----------------- | --------------------- | ------------------------------ |
| per-request group | high                  | **0.095**                |
| per-layer         | high                  | **0.109**                |

Read this pair carefully, because the two columns say opposite things and both
are true. The *traffic matrix* is stable across windows — consistent with C1, it
is a rank-1 object determined by aggregate expert reach, which barely moves. But
the *circuit plan* is not, because a degree-bounded greedy matching over a
near-flat weight distribution is chaotically sensitive: many candidate pairs
have almost identical weight, so tiny fluctuations reshuffle which ones win a
port. Plan persistence of 0.09 means ~91 % of circuits would be torn down and
rebuilt every window.

The correct engineering conclusion is *not* "OCS is infeasible because traffic
is unstable". It is:

* A **dynamic** affinity-chasing controller is unjustified: it would pay
  millisecond reconfigurations to chase a plan whose churn is driven by weight
  ties, not by real traffic shifts.
* A **static** plan is the right design, and it loses little: static
  (fit-only) OCS achieves 8.87 % bottleneck reduction at zero reconfiguration
  cost, and `value_of_prediction_pct` — the gap to an oracle that sees the
  evaluation workload — is what quantifies how little prediction buys. Circuit
  selection should be *stabilised* (hysteresis / weight thresholds), not made
  more reactive.

### 4.4 What this means for the OCS contribution

OCS survives as a **bounded feasibility analysis**, not a headline. Defensible
statements: OCS can only address the oversubscribed fraction of MoE all-to-all
traffic; that fraction is zero at realistic pod sizes for current EP degrees;
where it is non-zero a static affinity-informed plan captures ~9 % of the
critical path; and dynamic reconfiguration is not justified because plan churn
is dominated by weight ties rather than workload shift. Every one of those is a
result. None of them is "we apply OCS to MoE and it is faster".

---

## 5. Revised experimental design

### 5.1 Workload — `src/serving/suite.py`, 112 sequences

Factorial, so that semantics and surface form can be separated:

| role                        | n  | purpose                                                       |
| --------------------------- | -- | ------------------------------------------------------------- |
| 12 semantic categories × 6 | 72 | between-category contrasts; category decoding                 |
| 5 paraphrase sets × 3      | 15 | same meaning, minimal shared words → tests semantics         |
| 4 lexical controls × 3     | 12 | same template & words, different domain → tests surface form |
| 3 length ladders × 3       | 9  | rules out sequence-length artifacts                           |
| identical-prompt repeats    | 4  | **measurement noise floor**                             |

Categories: code generation, code debugging, math reasoning, science QA,
history QA, factual recall, instruction following, creative writing,
summarisation, translation, technical explanation, conversational. Prompts vary
in wording, length, difficulty, subject and syntactic form within each category,
so a category is not a single template.

The repeat set is not optional. Without it, "paraphrases route 0.92 alike" is
uninterpretable; with it, we know the ceiling is 1.00 and 0.92 is 63 % of the
available span.

### 5.2 Pipeline — routing trace as the canonical IR

```
model ──► capture_workload.py ──► traces/*.json + manifest.json
                                        │
                                        ▼
                                  CellTable            (src/eval/trace_ir.py)
                                  immutable; per-layer expert namespaces
                                        │
              ┌──────────────┬──────────┴──────────┬───────────────┐
              ▼              ▼                     ▼               ▼
        affinity_graph  specialization       placement_opt      cost_model
        (+ load null)   (+ perm nulls)       (fit slice only)   (topology,
              │              │                     │             dispatch mode)
              └──────────────┴──────────┬──────────┴───────────────┘
                                        ▼
                                    ocs_eval
                              (tier promotion, ports,
                               reconfig, stability)
                                        ▼
                            verify_live_invariance.py  Q1..Q5
```

Placement and topology modules take `CellTable` as input and never write to it,
which is what makes Q1 structural rather than asserted.

### 5.3 Experiment matrix — each axis independently variable

| dimension        | values                                                          |
| ---------------- | --------------------------------------------------------------- |
| model            | Qwen1.5-MoE (E=60,K=4), Qwen3.6-35B (E=256,K=8)                 |
| workload         | 12 categories + 3 control families                              |
| EP degree        | any divisor of E (2 … 64 swept)                                |
| dispatch         | REPLICATED, DEDUP_RANK, DEDUP_NODE                              |
| placement        | 11 kinds incl. random null and adversarial upper bound          |
| perturbation     | swap_two, swap_many, full_permute, move_hottest, split_top_pair |
| topology         | single_node, single_pod, multi_pod, realistic                   |
| OCS switch class | mems_10ms, mems_1ms, fast_10us, ideal_0                         |
| split            | in-sample, within-category, leave-categories-out                |
| time window      | per-request, per-token-range, per-layer                         |

### 5.4 Statistical discipline

* **Permutation nulls at the run level, never the cell level.** A 130-token
  sequence contributes ~3000 dependent cells; permuting cell labels inflates
  the effective sample size ~1000× and makes anything significant. Every null
  here permutes whole sequences.
* **A load-preserving null for affinity.** Popular experts co-occur because
  they are popular. `null_experts` resamples each cell's top-k (without
  replacement, via Gumbel top-k) from that layer's empirical load, preserving
  marginals and destroying only the joint. Affinity is reported as excess over
  this null: Qwen3.6 excess ratio **1.60×**, z well beyond 2.
* **`random`, not `linear`, is the null placement.** Measured: linear and
  random are within 0.4 % of each other on fan-out. Reporting a gain against
  `linear` and calling it a gain against "no optimisation" is fine; calling it
  evidence that affinity matters is not.
* **`adversarial` gives the range.** −129.9 % on Qwen3.6 tells us +32.8 % is a
  real fraction of a wide achievable span, not a rounding artifact.
* Fixed seeds throughout; multiple random placements (n=4 default) and multiple
  null draws (n=15–20); 95 % CIs on all similarity contrasts.

---

## 6. Results summary

### Q1 routing invariance — **holds**

Structural across 15 configurations (match rate 1.000, cost spread non-zero).
Gate bit-exact across identical-prompt repeats on both models: noise floor 1.000.

### Q2 routing structure — **holds**

Category decoding 62.5 % / 93.75 % vs ~6 % permutation null (p=0.005).
Driver `semantic_dominant` on both. Affinity exceeds the load-preserving null
(excess 1.60× on Qwen3.6). Per-expert specialisation 1.9 % (Qwen1.5) vs 23.7 %
(Qwen3.6) of the log₂(12) bound — a genuine 12× model difference, replacing the
saturated entropy comparison. Cross-layer load r = 0.004 / 0.011 confirms
per-layer namespaces.

### Q3 placement changes cost — **holds**

Routing bit-identical across all placements and 5 perturbation families;
bottleneck spread 3.7 % / 13.3 %. Total volume exactly placement-invariant
under REPLICATED (0.000 % spread) — the structural fact behind B2.

### Q4 affinity value — **holds, with the formulation caveat**

Qwen3.6, leave-categories-out: `affinity_coordinated_layer` **+32.8 %**,
direct bottleneck optimisation +31.1 %, load balancing alone +11.1 %,
layer-pooled affinity +7.4 %, naive per-layer affinity **−100.6 %**.
Qwen1.5, leave-categories-out: `fanout_layer` +19.2 %, `affinity_layer` +16.7 %
(no imbalance penalty at this scale because K=4 and skew is only 1.64×).
Affinity graph transfers across held-out categories: Pearson **0.909**,
top-1 % edge Jaccard 0.576, fan-out generalisation gap 0.85–1.54 %.

### Q5 OCS — **applicable only under a small-pod assumption; dynamic control not justified**

Zero cross-pod traffic at realistic pod sizes for every EP degree tested.
Under `multi_pod`: static fit-only OCS 8.87 % bottleneck reduction at zero
reconfiguration cost. Circuit-plan Jaccard 0.095 (per-request) / 0.109
(per-layer) against a *stable* traffic matrix → churn is weight-tie noise, so
the design conclusion is a stabilised static plan.

### The model-dependence finding

The exploitable structure depends on gating sparsity K/E and on EP degree.
Qwen3.6 (K/E = 3.1 %) shows a large effect; Qwen3.8-Whittle (E=64, K=16,
K/E = 25 %) shows **~0 %** for *global* affinity placement across every EP
degree from 2 to 64, because fan-out is already saturated at `min(K,W)` and
there is no room to coalesce.

**Update (full workload chain on Whittle, EP=32, 87 sequences, 778k cells —
`logs/workload/whittle/evidence_chain.json`):** the "~0 %" applies to the
layer-POOLED placement only. The coordinated per-layer formulation retains
**+19.3 %** out-of-sample at 25 % sparsity (direct bottleneck optimisation
+19.1 %, pure load balancing +9.5 %, naive per-layer affinity −78.8 %,
pooled affinity +1.7 %), with Q1–Q3 PASS (gate bit-exact, category decoding
89.6 % vs 5.8 % null, spread 3.2 %) and Q5 conditional exactly as for
Qwen3.6 (static OCS 8.6 % under the small-pod assumption, plan persistence
0.21/0.14, inapplicable at realistic pod sizes). At high K/E the win comes
from joint ingress balancing rather than destination coalescing — the
regime caveat binds the pooled *method*, not the coordinated *formulation*.
Since DeepSeek-V3-class models sit at K/E ≈ 3 % **[lit]**, the regime where
this work applies is the one production models occupy — but the boundary
must be stated, not hidden.

---

## 7. Reproducing

```bash
# 1. capture routing over the full 112-sequence suite (MLX; no vLLM needed)
python3 scripts/capture_workload.py \
    --model models/Qwen1.5-MoE-A2.7B-Chat-4bit \
    --out logs/workload/qwen15 --max-tokens 96              # ~5.5 min

python3 scripts/capture_workload.py \
    --model models/Qwen3.6-35B-A3B-4bit \
    --out logs/workload/qwen36 --max-tokens 64 \
    --per-category 4 --n-repeats 3                          # ~6.5 min

# 2. the staged evidence chain
python3 scripts/verify_live_invariance.py \
    --workload logs/workload/qwen36 --world-size 32 \
    --topology single_pod --topology multi_pod --topology realistic

python3 scripts/verify_live_invariance.py \
    --workload logs/workload/qwen15 --world-size 15 \
    --topology single_pod --topology multi_pod --topology realistic

# single stage, faster iteration
python3 scripts/verify_live_invariance.py \
    --workload logs/workload/qwen36 --stage q4 --world-size 32

# 3. figures
python3 scripts/make_figures.py --workload logs/workload/qwen36
python3 scripts/make_figures.py --workload logs/workload/qwen15
```

Outputs: `traces/*.json` (raw routing), `manifest.json` (design matrix),
`evidence_chain.json` (all metrics), `figures/*.png`.

---

## 8. What to do next

1. **Lead with placement, not OCS.** The 32.8 % out-of-sample critical-path
   reduction is the result. OCS is a bounded feasibility section.
2. **Make the coordinated per-layer formulation the technical core.** The
   contrast between −100.6 % (naive) and +32.8 % (coordinated) is the paper's
   most instructive finding and it is a general lesson about per-layer
   placement in MoE, independent of OCS.
3. **Verify the [lit] claims** in §4.1 against primary sources before citing —
   especially exact EP degrees, the node-limited-routing constant, and OCS
   reconfiguration figures.
4. **Extend the model sweep along K/E.** Two points (3.1 % and 25 %) establish
   that the effect is sparsity-dependent; a proper curve would make the
   applicability boundary a contribution rather than a caveat.
5. **Close the loop to wall-clock.** Every cost number here is from an
   analytical bottleneck model. The honest next step is to replay a fitted
   placement through the existing `src/comm/all_to_all.py` data plane and
   confirm the predicted ordering survives on real hardware. *(The blocking
   defects C6, C7, C9, C12 are now fixed in that data plane.)*
6. **Do not repair the discarded metrics.** `load_entropy_norm`,
   `top5_expert_share`, `layer_diversity_mean_js`, `affinity_strength_offdiag`
   and pooled `js_divergence` are saturated by construction (C2). They should be
   deleted, not recalibrated. *(Done: `scripts/compare_model_affinity.py` now
   reports per-layer statistics only; the ledger's A2 entry is updated.)*
