# Hot-spine-only OCS — the plan, and the falsifiers that come first

Status: **plan only. Nothing is changed by this document** — no module, config,
result or existing document is modified. Companions: `docs/plan_next_steps.md`
(the P0–P8 program), `docs/ocs_moe_program.md` (what is verified),
`docs/circuit_selection.md`, `docs/ownership_measurement.md`,
`docs/contention_fixes.md`, `docs/astra_ordering.md`.

Marker convention follows the repo: **[V]** verified by a command, **[C]** read
from code, **[A]** assumed.

---

## 0. The idea, stated precisely — because as stated it has two readings

> "Replace only the popular / frequently congested spines with the OCS, and do not
> guarantee that negative optimisation is impossible: the replaced spines will
> definitely not be the last to finish, and that is enough. No training/inference
> composition."

Two readings, and they are not the same claim:

| # | reading | statement | status in this repo | needed? |
| - | ------- | --------- | ------------------- | ------- |
| **A** | **identity** | no promoted object is ever the last-to-finish (makespan argmax) | **false, and false by construction** — a circuit changes a pair's *rate*, and only ~51 % of the binding rank's bytes sit on promotable pairs (`f = 0.51` **[V]** `bottleneck_attribution`), so a promoted rank can remain the argmax, just faster | **no** |
| **B** | **time** | promoting does not increase the makespan: `M(S) <= M(empty)` | not yet testable — see §2.1 | **yes** |

Reading **B** is what "that's enough" has to mean, and it is the right one: the
makespan (max over rank ports of drain time) is the only end-to-end quantity that
survives *without* composing a training/inference model, which is exactly the
scope you asked for (§6).

The real content of the idea is therefore not "hot-only" (the planner is already
that, §1) but two things:

1. **a screening rule with a threshold** — promote a resource only if the traffic
   on it pays for what promoting it costs;
2. **a robustness clause** — "**frequently** congested", i.e. hot across windows,
   not hot once.

Both survive as separate, testable conditions (§3, P3).

---

## 1. What already exists, so the new part is not "pick the hot pairs"

| piece | where | rule |
| --- | --- |
| promotion | `src/eval/cost_model.py:141,176-187` | a circuit promotes one **unordered rank pair** from CROSS_POD to OPTICAL; the pair's whole volume moves, tier-level, no per-byte fraction |
| promotion benefit | `cost_model.py:99-106` | rate is a pure function of tier: `CROSS_POD = nic/sigma = 12.5`, `OPTICAL = nic = 50` GB/s — **the x-sigma multiple is identical for every pair, independent of how loaded that pair is** |
| deployed planner | `src/eval/ocs_eval.py:82` `plan_circuits` | greedy **top-K by dispatch bytes**, degree bound `ports_per_rank`, cardinality `n_circuits`; objective = covered promotable **bytes** |
| exact planner | `src/eval/bmatching.py:96,113` | LP / MILP, same objective; greedy is within **0.36 %** of exact **[V]** |
| metric | `cost_model.py:451-473` | `bottleneck_us = alpha + max(egress_nic, ingress_nic, egress_nvl, ingress_nvl)` — a **max over rank ports**, not over pairs |

So "replace the popular pairs" **is** today's planner: it already ranks candidates
by bytes and refuses to exceed the port budget. The idea is not a new selector —
it is a new **admissibility criterion** on the selected set, plus the spine
granularity you name.

---

## 2. The two gaps that make the idea untestable as the repo stands

### 2.1 There is no harm channel — every circuit set is harmless **[C], to be confirmed by P0**

Read out of `cost_model.evaluate`:

* `drain` is a **sum** over pairs of `bytes / bandwidth(tier)` (l.458-463), and
  `bandwidth` is non-decreasing when a pair is promoted (`12.5 -> 50`);
* `alpha = latency(max tier present)` (l.469-470), and `OPTICAL` (idx 3) carries
  the **lowest** latency (6 us vs 12 us).

Both terms are monotone non-increasing in the circuit set, so:

> **`M(C) <= M(none)` for every circuit set C.** In the current model the negative
> optimisation this idea guards against **cannot be expressed at all**.

Consequence for the plan: an experiment run today would report "hot-only is
harmless" as a property of the arithmetic, not of the rule. **The harm channel has
to be built before the rule can be tested** — that is the single most important
line in this document.

### 2.2 "Spine" is not an object, and sigma is a literal

`spine` appears in `src/` only in comments (`cost_model.py:28,58,73`);
`src/comm/topology.py:53` describes a cross-pod "spine/core fabric" but models no
switch. There is **no per-spine load, no per-spine congestion, no spine-to-pair
map**. And the tier's contention is the global literal `core_oversubscription =
4.0`, which `docs/eps_baseline.md` §3 already flags as **an uncited knob** that
fails its own defensibility gate.

So "popular spine" currently has no referent, and "congested spine" no measurement
— the census does not exist (P1 below).

---

## 3. The plan

Seven phases, all additive (the repo's pattern: new modules
*import* the reference model, never modify it, and reproduce `evaluate()` exactly
when the new features are off — as `src/eval/contention.py` does, `rel_err <
1e-9`).

### P0 — premise check + freeze the claim

1. **Assert §2.1.** A 20-line test: for each of the three workloads, 200 random and
   200 adversarial circuit sets, assert `M(C) <= M(none) + 1e-9`. Gate: if it
   fails, a harm channel already exists somewhere and the plan changes before
   anything is built. If it passes, it becomes the paper's "why this rule needed a
   new model" paragraph, *measured*.
2. **Write the claim down** in the §0/§5 form: reading B, the screening rule, the
   falsifiers. Nothing gets built before this exists, because the difference
   between the useful and the trivial version of the idea is exactly this
   paragraph.

### P1 — materialise the spine, and census it

**New: `src/eval/spine_fabric.py`** (additive overlay, same pattern as
`contention.py`).

```python
@dataclass
class SpineConfig:
    n_planes: int = 4                # parallel planes of the core
    plane_capacity_gbytes_per_s: float = 50.0
    assign: str = "ecmp"             # how a rank pair picks its plane
    port_sharing: float = 0.0        # kappa in [0,1]: 0 == today's model exactly

def assign_pairs(topo, cfg) -> np.ndarray          # [W, W] -> plane id
def spine_load(counts, topo, cfg) -> dict          # per-plane bytes, pairs, sigma_s
def evaluate_spined(t, placement, topo, cost=None, mode=..., n_dp=None,
                    seed=0, circuits=None, sc=None) -> dict
```

The physical statement: in a Clos core a rank-pair flow takes **one** plane (ECMP
picks it), so a plane's load is the sum of the bytes of the pairs hashed to it, and
a pair's rate on that plane is `plane_capacity / n_pairs(plane)` — hence a
**per-plane oversubscription sigma_s**, derived from the trace and the radix rather
than assumed. That replaces the uncited global `sigma = 4.0` with a measured
distribution and closes the §3 hole in `eps_baseline.md`.

**New: `scripts/spine_census.py`** -> `outputs/spine/census.<workload>.json`,
reusing `logs/workload/{qwen15,qwen36,whittle}` and the measured ownership model
`measured_packed_4`:

| what | why it decides |
| --- |
| per-plane bytes, pair count, sigma_s, utilization | is there anything to call a spine? |
| concentration (Gini, top-k share of cross-pod bytes) | how few planes carry the load |
| **Spearman(popularity by bytes, congestion by sigma_s)** | **the number that says whether "popular" is a valid proxy for "congested"** |
| congestion frequency per plane over windows | the "**frequently** congested" clause |
| plan Jaccard across windows (extends `plan_window_stability.py`) | is a hot set stable enough to hold? |

Gate: the census must reproduce the known concentration result (16 hottest pairs =
12.5 % of cross-pod bytes concentrated vs 6.8 % spread **[V]**) before any new
number from it is quoted.

### P2 — build the harm channels

Four channels, each independently switchable so every delta is attributable:

| id | channel | mechanism | exists today? |
| -- | ------- | --------- | ------------- |
| **H1** | **port displacement** | OCS circuits and EPS uplinks draw on the **same** ports at a rank. `ports_per_rank` today bounds circuits only (`ocs_eval.py:115`); making it a shared pool means `c` circuits leave `P-c` for everything else, so the unpromoted cross-pod bytes at that rank drain slower | **no** — this is the channel that makes the rule both meaningful and necessary |
| **H2** | reconfiguration | `alpha_r` amortised over N passes (`OcsConfig.reconfig_us = 10 ms`; classes 10 ms / 1 ms / 10 us / 0) | in `completion.py` only, not in `evaluate` |
| **H3** | per-flow BDP cap | a promoted pair needs **600 KB in flight** at the 12 us optical RTT to reach 50 GB/s; 256 KiB binds, 1 MiB does not **[V]** | yes, `contention.py:159-160` |
| **H4** | objective mismatch | covering more bytes is not a better time — measured reversal **[V]** `circuit_selection.md` | yes |

H1 is the honest one to build and the one to be sceptical about: it is a **knob**
(like sigma), so it ships with a sharing factor `kappa in [0,1]` whose `kappa = 0`
case reproduces today's `evaluate()` to 1e-9, and every claim is reported as an
**envelope over kappa**, never a point. That keeps it inside the discipline the
repo already applies to sigma and the latencies.

### P3 — the screening rule, made exact (part of P2's module)

For a pair `p = (a,b)` on a plane with per-pair rate `r_s`, a port pool of `P`
per rank and `c` circuits at that rank:

```text
recaptured(p) = bytes_p * (1/r_s - 1/nic)          # what the circuit buys
displaced(a)  = drain_a(unpromoted cross-pod) * kappa*c/(P - kappa*c)
admit p  <=>  recaptured(p) > displaced(a) + displaced(b) + alpha_r/N
```

and the temporal clause:

```text
admit plane s  <=>  share of windows with sigma_s > 1  >=  q     # "frequently", not "once"
```

This is your rule, made checkable: **a resource is replaced only if the traffic on
it pays for the capacity it displaces, and only if it is congested often enough to
amortise the switch.** The threshold is not a taste parameter — it falls out of
`P`, `kappa`, `alpha_r`, `N` and the plane's own load.

### P4 — policy sweep

**New: `scripts/hot_spine_policy.py`** -> `outputs/spine/hot_spine_policy.json`.

| axis | values |
| --- |
| policy | `ALL`, `HOT-BYTES` (= today's greedy), `HOT-SIGMA` (congestion-ranked), `THRESH` (the P3 rule), `COLD`, `RANDOM` |
| granularity | pair-level (today) vs **plane-level** (your "replace a spine") |
| workloads | qwen36 W=32, qwen15 W=30, whittle W=32 |
| placement | `linear`, `affinity_coordinated_layer` |
| ownership | `hash` (the assumption), `measured_packed_4` (the measurement) |
| fabric | `multi_pod`, `realistic`; sigma in {1,2,4,8} **and** derived sigma_s |
| budget | `n_circuits` in {8,16,32} x `ports_per_rank` in {1,2,4} |
| cost | reconfig class in {0, 10 us, 1 ms, 10 ms} x N in {1, 100, 10000} |
| harm | `kappa` in {0, 0.25, 0.5, 1.0} |

Metrics per cell: `M0`, `M(S)`, delta %, **negative_rate** (`M(S) > M0`),
**admissibility_violation_rate** (promoted pairs failing the P3 rule), argmax
resource and rank before/after, whether the promoted set holds the straggler
(**reading A, as a claim to be falsified, not a requirement**), sigma_s
distribution, congestion frequency, Spearman(popular, congested), plan Jaccard.

Gates: (a) `kappa = 0` reproduces every published number (4.755 / 2.528 / 9.462 /
8.669 / 8.4-9.8 %) to 1e-9 — the regression gate the repo already uses; (b) the
policy ordering must be stable across the three workloads, or the result is a
single-model artifact; (c) `bash scripts/check_env.sh` (33 tests) stays green.

### P5 — falsification — see §5

Deliberately construct the counterexamples rather than hoping the sweep misses
them.

### P6 — report + the deployed decision rule

Update this document with results; add the one-page controller rule ("promote plane
`s` iff `sigma_s > 1` in at least q windows and `recaptured > displaced +
alpha_r/N`; never promote on bytes alone; hold the plan across windows") and wire
it next to the existing seam `src/ocs/online_controller.py:155`, which today
re-runs the bytes-ranked top-K every interval.

### P7 — optional, external

ASTRA-sim cross-check on the **plane** axis: the analytical backend is tier-level,
so a whole-plane promotion can be expressed as a parallel tier with its own
bandwidth, but **per-pair** promotion still needs ns-3 (`configs/astra_sim/README.md`).
This validates ordering only, and only after P4; it is not on the critical path.

---

## 4. Sequencing

| phase | deliverable | gate |
| --- | --- | --- |
| P0 | premise assertion + the claim in writing | `M(C) <= M(none)` holds for all 400 probe sets, else stop |
| P1 | `spine_fabric.py`, `spine_census.py`, census JSON | reproduces the known concentration; gives Spearman(popular, congested) |
| P2 | harm channels H1-H4, kappa-switchable | `kappa = 0` reproduces `evaluate()` to 1e-9 |
| P3 | the screening rule | rule stated, computable, violations countable |
| P4 | policy sweep + JSON | gates (a)(b)(c) above |
| P5 | falsification suite | counterexamples attempted, outcomes recorded either way |
| P6 | report + decision rule | every number has one reproducing command |
| P7 | optional ASTRA-sim plane check | ordering only |

---

## 5. Falsifiers — written before the experiment, not after

| id | what would falsify | how it is measured | what it would mean |
| -- | ------------------ | ------------------ | ------------------ |
| **F1** | the rule is not sufficient | any admissible config with `M(S_hot) > M0` | hot-only does not guarantee non-negativity; the P3 threshold is wrong |
| **F2** | "popular" is the wrong variable | Spearman(bytes rank, sigma_s rank) < 0.5 | the *deployed* planner (`ocs_eval.py:82`, bytes-greedy) selects the wrong resource; "congested" must replace "popular" |
| **F3** | the rule has no content | negative_rate(hot-only) about equal to negative_rate(cold/random) | promoting anything is as good as promoting the hot set: the rule is trivial, and the honest output is that the guarantee is structural (§2.1), not earned |
| **F4** | the mechanism is not the one claimed | promoting the hot set never changes the straggler identity | reading **A** is not doing the work — the gain is arithmetic (rate x bytes), not a critical-path argument; the write-up must say "non-increase", not "not last to finish" |
| **F5** | the effect is a single-model artifact | ordering across qwen15/qwen36/whittle disagrees | no claim; report per-model |

**Kill criterion.** If F3 holds *and* the kappa-envelope peaks below ~1 % absolute
gain, the correct output is the negative result — "with the ports the OCS costs,
only the hottest pairs pay for themselves, and that is a *scope* statement" — and
the placement result stays the headline. That is a publishable outcome, and it is
written here so it cannot be discovered late and quietly dropped.

---

## 6. Out of scope (your instruction, honoured explicitly)

* **No training/inference composition**: no TTFT, no ITL, no throughput, no
  `ServingModel`, no `compute_us_per_layer`. `src/eval/completion.py` is not used.
* **No workload-type distinction**: the trace is treated as an offered traffic
  matrix; whether it was captured in prefill, decode or training does not enter.
* The metric is the **fabric's own**: per-pass makespan and straggler identity.
* **No kernel timings**, and no re-litigation of routing (A1 stays the gate).

---

## 7. Open decisions before P1 starts

1. **Spine granularity.** Plane-level promotion (one OCS *replaces a spine* — your
   phrasing, and the physically deployable unit) or pair-level (today's planner,
   finer and strictly more general)? The plan assumes **both are swept**, with
   plane-level as the headline.
2. **H1 port displacement.** Build it (needed for the claim to have content, §2.1)
   or stay with today's model (in which case the honest headline is "the guarantee
   is structural")? The plan assumes **build it, kappa-enveloped**.
3. **N (passes per reconfiguration).** One plan per serving window (static, matches
   `plan_window_stability`), or per-window re-planning? The plan assumes **static
   with N swept**, since the measured regime shows a stable plan is available.

---

## Reproducing (once P0-P6 land; nothing here exists yet)

```bash
python3.12 scripts/spine_census.py     --workload logs/workload/qwen36 --world-size 32
python3.12 scripts/hot_spine_policy.py --workload logs/workload/qwen36 --world-size 32 --ownership measured_packed_4 --kappa 0.5 --policy thresh
bash scripts/check_env.sh              # 33 tests, must stay green
```
---

## 8. Feasibility verdict (added before any code was written)

**As a research result: low.** Three reasons, in order of severity.

1. **It is vacuous in the model it would be tested in** (§2.1): `M(C) <= M(none)`
   holds for every circuit set, so "hot-only is harmless" is arithmetic, not a
   finding. The plan therefore *builds the harm it then detects*.
2. **The one channel that would give it content is an invented knob.** Port
   displacement (H1) rests on an unverified deployment fact: whether OCS circuits
   are *additive* ports at the ToR (a parallel optical path — then `kappa = 0` and
   the rule is trivial) or *displacing* uplink ports (then the rule has teeth).
   That is the same class of assumption as `sigma = 4.0`, which this program has
   already ruled indefensible, and the plan defers the fact to a swept parameter
   instead of checking it first.
3. **The stated mechanism is inverted.** A circuit pays exactly when its traffic
   *is* on the binding port — the metric is a max over rank ports, and promoted
   bytes still drain through the same NIC. "Will not be the last to finish" is the
   opposite of the operative condition. The repo already holds the correct version
   of the idea: `src/eval/promote_aware.py:274` optimises the **post-circuit**
   critical path (`PostCircuitOracle.objective`), which is a strictly stronger
   selector than any popularity rule.

**As a guardrail: high, and it needs a paragraph, not a program.** "Never spend a
circuit on a resource that is not congested, and never without a plan that survives
the window" is worth one page in the write-up and one condition in
`src/ocs/online_controller.py`.

**Feasibility of the phases as written:** P0, P1, P4, P5 are runnable with what is
in the repo; P2 is runnable for H2/H3/H4 (H3/H4 already exist) but H1 is the
contested piece; P7's premise was **wrong on inspection** — see §9: the
congestion-aware analytical backend is already built and has never been run on our
trace, per-rank wall times are already parsed, and ns-3 is present in the checkout. P4's grid as tabulated is also too
large to run naively; it needs one-layer slices and the repo's existing
`--quick` pattern, or it will not finish.

**The cheap deciders, none of which needs new modelling** — these should run
*before* P1, and each can kill the idea:

| # | check | existing machinery | what kills the idea |
| - | ----- | ------------------ | ------------------- |
| D1 | are OCS ports additive or displacing at the ToR? | reading, not code | additive ⇒ no harm channel ⇒ the rule is trivial |
| D2 | does pair **bytes** rank predict the pair's contribution to the **binding** resource? | `scripts/bottleneck_attribution.py` + `plan_circuits` | high correlation ⇒ "popular" is already the right selector ⇒ nothing new |
| D3 | how often is a pair/plane hot across windows? | `scripts/plan_window_stability.py` | unstable ⇒ "frequently" is decoration; stable ⇒ reconfig is amortised once and the clause is decorative too |

**The salvageable version.** The only reading in which "frequently congested" is
neither already implemented nor vacuous is **tail selection**: choose circuits by
per-window congestion *frequency/volatility* rather than mean bytes, and score on
the **worst window** instead of the mean one. That is a new selector over existing
per-window traces, needs no `kappa` and no `spine_fabric` to be interesting, and
is measurable today — including the honest possibility that it loses to
bytes-greedy.
---

## 9. Route B — do it in ASTRA-sim, on the real LLM trace

Correction to §8 first: I wrote that per-pair promotion needs ns-3 and is therefore
out of reach. On inspection, two of those clauses are simply wrong.

* `~/astra-sim/build/astra_analytical/build/bin/` contains **both**
  `AstraSim_Analytical_Congestion_Aware` and `AstraSim_Analytical_Congestion_Unaware`.
  `scripts/astra_placement_order.py:18` runs the **unaware** one; the aware binary
  has only ever been used on the uniform 16-NPU toy
  (`configs/astra_sim/README.md`). It is the most valuable unused instrument here.
* `scripts/astra_placement_order.py:196` already parses `Wall time: (\d+)` **per
  rank** and keeps the max. Keeping the `argmax` instead yields the **straggler
  rank** — exactly what reading A is about, and what the closed-form model cannot
  produce at all.
* ns-3 is **present** at `~/astra-sim/extern/network_backend/ns-3`, so per-pair
  promotion (per-link JSON) is buildable, not blocked.

What route B buys that §3's plan cannot:

| the plan's soft spot | what ASTRA-sim replaces it with |
| --- | --- |
| `sigma = 4.0`, invented | contention **emerges** from the real per-pair traffic in the congestion-aware backend |
| `kappa`, invented | the cost of a port given to the OCS is bandwidth the EPS fabric **no longer has**, and the consequence is simulated |
| no per-pair / straggler metric | per-rank wall times, hence the makespan **and its argmax** |
| reconfiguration as a subtraction | `COMP_NODE` carries a `runtime` (us) in the custom ET — 10 ms of reconfiguration is **injectable** |

Three encodings of promotion, cheapest first:

1. **Removal + line-rate charge** (today's seam). `chakra_from_traffic.py
   --single-layer` writes the per-pair ET; drop the promoted pairs' send/recv nodes
   and charge them as `bytes/50 + 6 us`. Hybrid makespan = max(ASTRA makespan of
   the remainder, the promoted tails). The promoted part is our arithmetic again —
   but the contention is not.
2. **ns-3 per-link JSON.** Every rank pair gets a link entry: promoted pairs on an
   optical link at NIC rate, the rest on the oversubscribed core. No hybrid, true
   per-pair granularity, and the core is a shared resource inside the simulator.
3. **Reconfiguration.** A `COMP_NODE` with `runtime` = the reconfig class, between
   plan changes.

**The trap.** In any traffic-level simulator, *removing* load from a shared fabric
is monotone-helpful, so encoding 1 alone will find no negative optimisation — it
reproduces §2.1 with a bigger engine. The negative can come only from the two
costs: **the port the circuit consumes** (encoding 2, or a tier-bandwidth cut under
encoding 1) and **the reconfiguration time** (encoding 3). An experiment without
those two is a sanity check, not a test of the rule.

**Scale.** The aggregate trace is 6.4 GB and did not survive; one `linear` layer is
165 MB and did (`159 705` cycles). So: congestion-aware analytical on one-layer
slices first; ns-3 on a handful of configurations as an **ordering gate only** —
the doctrine `docs/hardware_validation.md` already states.

**Smallest run that would settle something** (one layer, real trace, three
configurations): none / hot-set promoted / cold-set promoted, congestion-aware
analytical, reporting makespan, its argmax, and the promoted-set-holds-the-straggler
flag.
---

## 10. Results — first run (real Qwen3.6 trace, one MoE layer)

Commands (all additive; nothing existing modified):

~~~bash
python3.12 scripts/harm_channel_probe.py --workload logs/workload/qwen36 --world-size 32
python3.12 scripts/spine_probe.py --workload logs/workload/qwen36 --world-size 32 --stage astra \
    --configs eps,hot16,cold16 --encode scale --backends Congestion_Unaware \
    --netcfg configs/astra_sim/eps_baseline_2tier.yml
python3.12 scripts/spine_probe.py --workload logs/workload/qwen36 --world-size 32 --stage astra \
    --configs eps,hot16,cold16 --encode scale --backends Congestion_Aware,Congestion_Unaware \
    --netcfg configs/astra_sim/spine_switch32.yml
python3.12 scripts/spine_probe.py --workload logs/workload/qwen36 --world-size 32 --stage all \
    --configs eps,hot16,cold16
~~~

### 10.1 P0: the harm channel does not exist **[V]**

409 circuit sets (empty / all / hot top-k / cold bottom-k / busiest-rank / 200 random /
200 adversarial) x 2 slices, on the real trace, `measured_packed_4`, `linear`:

| slice | candidate promotable pairs | baseline `bottleneck_us` | violations of `M(C) <= M(none)` | worst delta |
| --- | --- | --- | --- | --- |
| layer 0 | 64 | 4 040.37 | **0 / 409** | +0.000000000 us |
| full decode trace | 64 | 148 603.45 | **0 / 409** | +0.000000000 us |

Promoting the hot 16 pairs is worth **-951.5 us**; promoting the cold 16 is worth
**-384.7 us**; promoting everything is worth **-2 699.6 us**. All negative, none
positive: in this model the rule is *better*, never *safer*. §2.1 stands.

### 10.2 The rule's ordering holds in every engine **[V]**

Same traffic (layer 0, 164 978 688 bytes on the wire, 118 directed links), three
configurations: no circuits / hot 16 promoted / cold 16 promoted.

| instrument | eps | hot16 | cold16 | hot vs cold |
| --- | --- | --- | --- | --- |
| closed form `evaluate` (bottleneck_us) | 4 040.4 | 3 088.9 (-23.5 %) | 3 655.6 (-9.5 %) | 2.5x |
| FIFO replay (us) | 3 223.5 | 2 595.0 (-19.5 %) | 2 850.8 (-11.6 %) | 1.7x |
| ASTRA-sim 2-tier, congestion-unaware (cycles) | 159 705 | 146 375 (-8.4 %) | 159 705 (0.0 %) | **cold buys nothing** |
| ASTRA-sim 1-D Switch, congestion-**aware** (cycles) | 943 174 | 814 256 (-13.7 %) | 901 362 (-4.4 %) | 1.8x |
| ASTRA-sim 1-D Switch, congestion-unaware (cycles) | 74 811 | 60 697 (-18.9 %) | 74 811 (0.0 %) | cold buys nothing |

Four independent instruments, one ordering: **hot > cold >= eps, never negative.**
The congestion-aware backend is the interesting column: contention there is
**emergent** (943 174 vs 74 811 cycles for identical traffic, 12.6x) rather than
stated as `sigma = 4`, and the hot-only rule still wins.

### 10.3 Link state: what "not the last to finish" looks like **[V]**

From the FIFO replay (`outputs/spine/{eps,hot16,cold16}.tiers.json`), span-normalised:

| config | EPS core busy | EPS core idle gaps | largest EPS gap | OCS circuits busy | largest OCS gap |
| --- | --- | --- | --- | --- | --- |
| eps | **100.0 %** | 0 | — | — | — |
| hot16 | 84.6 % | 2 | 57.3 us | 17.2 % (16 links) | 143.7 us |
| cold16 | 95.5 % | 1 | 5.6 us | 11.3 % (16 links) | **484.4 us** |

Per-link strips (72 bins, `#` = carrying traffic, `.` = idle) make the mechanism
visible:

~~~text
eps    CROSS_POD  ########################################################################   occ 1.00
hot16  CROSS_POD  ........###################.############################################   occ 0.85
       OPTICAL    ############...##.......................................................   occ 0.17
cold16 CROSS_POD  #####################################################################...   occ 0.95
       OPTICAL    ...............................................####...........###...####   occ 0.11
~~~

The promoted links finish **early** and idle while the EPS core is still draining —
in `cold16` they are idle for 484 us in the middle of a saturated core. That is the
user's sentence rendered as a picture: *the replaced links are not the ones still
finishing.* It is also why cold promotion is nearly worthless: the circuit is
granted to a pair that was never the problem.

### 10.4 The caveat that decides how far the argument can go **[V]**

"Will definitely not be the last to finish" is only usable if *which* node finishes
last is stable. It is not. Rank correlation between ASTRA-sim's per-rank wall time and
the replay's per-rank makespan, 32 ranks, eps, 2-tier, unaware:

| candidate | Spearman vs ASTRA per-rank |
| --- | --- |
| replay makespan (max of egress, ingress) | **+0.147** |
| replay **ingress** | **+0.615** |
| replay **egress** | **-0.569** |
| bytes sent | -0.575 |

ASTRA-sim's per-rank time tracks what a rank **receives** and is *anti*-correlated with
what it **sends**; the replay charges the sender's egress port. ASTRA's top-5
stragglers are ranks 30, 31, 29, 28, 27 (all sinks); the replay's are 3, 20, 30, 31, 2.
The identity of the last-to-finish node therefore flips with the charging convention,
while the *aggregate ordering* between configurations does not.

Consequence, stated plainly: the rule survives as "**promotion does not increase the
makespan**". It does not survive as an identity claim about which node finishes last,
and no argument in this program may rest on the identity.

### 10.4b The regime depends on how many ranks own tokens **[V]**

The same instrument run over four token-ownership models is in
`docs/inference_driven_net.md` §5.4: with 4 owning ranks (the measured small-batch
case) the OCS buys **+0.00 %** at sigma = 4 and +18.2 % at sigma = 32; with tokens
spread over 32 ranks it buys a steady +8 %. The number of owning ranks, not the circuit
policy, is the first-order variable.

### 10.5 Limits of this run

1. **One layer (or four), one placement, one ownership model** (`linear`,
   `measured_packed_4`), 32 ranks.
2. **Promotion is encoded, not simulated.** ASTRA's analytical backends are
   tier-level, so a promoted pair is expressed by shrinking its simulated bytes by the
   promotion factor (50/12.5), with a 64 KiB floor because a zero-cycle message trips
   `EventQueue.cpp:33`. Per-pair promotion proper needs the ns-3 backend (present at
   `~/astra-sim/extern/network_backend/ns-3`, not yet built).
3. **The congestion-aware backend requires a 1-D topology** ("only support 1-dim
   topology"), hence `configs/astra_sim/spine_switch32.yml` and `spine_ring32.yml`.
   Its bandwidth (50 GB/s) is the *undivided* NIC rate, so the per-pair share is
   computed rather than asserted -- the point of using it.
4. **Absolute times are not comparable across instruments** (latency accounting, unit
   base, and per-pair vs per-transfer charging); only orderings and shapes are read here.
5. **No harm channel was exercised**, because neither engine has one. The displacement
   (H1) and reconfiguration (H2) encodings of §9 remain unbuilt, so these runs cannot
   yet say whether the rule is *necessary* -- only that it is *better*, and never worse.

### 10.6 Artifacts

| path | content |
| --- | --- |
| `outputs/spine/harm_channel_probe.qwen36.json` | P0: 409 circuit sets x 2 slices, every delta |
| `outputs/spine/astra_spine.*.json` | per-rank ASTRA wall times, stragglers, bytes |
| `outputs/spine/{eps,hot16,cold16}.messages.csv` | every message: collective, src, dst, bytes, **means (OCS / EPS / pod / NVLink)**, tier, rate, start, end |
| `outputs/spine/{eps,hot16,cold16}.links.json` | per-pair busy intervals, idle gaps, utilisation |
| `outputs/spine/{eps,hot16,cold16}.tiers.json` | per-tier occupancy, idle windows, ASCII strip |
| `outputs/spine/{eps,hot16,cold16}.ranks.csv` | per-rank communication makespan, egress / ingress / NVLink |
| `outputs/spine/comm_makespan_gantt.png` | per-rank Gantt (coloured by means) + core/circuit occupancy |
