# MoE + OCS: where the program stands

Everything below is either **[V]** verified by a command in this repo, **[C]** read
from code, or **[A]** assumed. Where the original narrative and the measurements
disagree, the measurement wins and the correction is stated inline.

---

## Part 1 — The chain, and where it holds

### 1.1 Motivation

LLM MoE serving has one structural cost: a token's expert set is decided by
content, but the experts live on other ranks. So every MoE layer pays a
communication round that is *not* removable by batching or scheduling — it is
decided by **where the experts sit**. [C]

### 1.2 The problem, stated precisely

```
token → experts   S(t,l) ⊆ E, |S(t,l)| = K      measured (A1), never changes with placement
expert → rank     P: [L, E] → W                 the decision
token → rank      O: tokens → W                 ASSUMED today (a hash); see §4.2
traffic           M[s,r] = A_O · B_P            bilinear: needs BOTH
cost              bottleneck = max over ranks of (tier-weighted bytes) + α
```

The fabric charges by **pair**, so what matters is not total bytes but where they
land: intra-node 450 GB/s, intra-pod 50, cross-pod 50/σ = 12.5 at σ=4. [C]

### 1.3 The medium: affinity, in two projections

Both are projections of the same object — the token's expert path across layers:

| | statistic | reduces | measured lever |
| --- | --- | --- | --- |
| **intra-layer** (ours) | `A[e,f]` = #tokens selecting both, same layer | **bytes per hop** | **−21.8 % volume** [V] |
| **inter-layer** (ExFlow) | `P(E_{f,l+1} | E_{e,l})` | **number of hops** | **−18.8 % volume** [V] |

The inter-layer figure comes from `hops = 2L − 2a(L−1)`, `saving = a(L−1)/L`, with
`a = 19.3 %` boundaries aligned and `L = 40`. Both levers are **the same order of
magnitude**.

### 1.4 The hardware: OCS as the spine layer

A circuit is a **tier promotion** for one rank pair: it removes that pair's
oversubscription. It does not add bandwidth. [C]

| knob | value | measured effect |
| --- | --- | --- |
| port / circuit budget | 8 / 16 / 32 | static gain **0.41 % / 2.53 % / 8.57 %** [V] |
| reconfiguration latency | 10 ms / 1 ms | break-even **3.4 / 0.34 token passes** [V] |
| applicability | needs a contended cross-pod pair | `single_pod`, `realistic` → **not applicable** [V] |

**The chain couples in both directions**: placement decides which pairs exist and
how hot they are; the switch budget decides which of those pairs can be served.
Measured, that coupling is a *substitution*, not a synergy (§3).

---

## Part 2 — What is verified

| # | claim | evidence |
| --- | --- | --- |
| V1 | routing is independent of placement and topology | A1 gate, bit-exact (pre-existing) |
| V2 | volume **is** fanout: `bytes = 2·N·fanout·H·dtype` | verified to 1e-6 on 6 placements [V] |
| V3 | co-location reduces volume — mechanically, by collapsing messages | fanout 7.27 → 5.65; affinity_layer → 4.96 [V] |
| V4 | **co-location alone is catastrophic** | `affinity_layer`: −32 % volume, **+83 % bottleneck vs random** [V] |
| V5 | the win needs **balance**, not just co-location | `affinity_coordinated`: −22 % volume, **−33 % bottleneck** [V] |
| V6 | cell `D` (OCS + affinity) is the minimum in all 3 models | synergy +1.62 / −2.87 / −6.13 % [V] |
| V7 | the inter-layer coupling is **strictly 1-hop** | cos 0.614 at lag 1 → 0.23 flat to lag 16 [V] |
| V8 | both statistics are **task-specific**, more so with depth | within-vs-between gap +0.14 → +0.26 [V] |
| V9 | capacity floors are structural | fanout floor `ceil(K·W/E)` = 1 / 8 / 2; 80–88 % of boundaries still move [V] |
| V10 | OCS and good placement are **substitutes** | OCS gain 8.4 % at bad placement → 2.5 % at the best [V] |
| V11 | circuit selection is exactly solvable | degree-bounded b-matching, integral polytope → LP is exact [C] |

### The one result that reorganises everything

```
volume  is a SUM over ranks   → co-location lowers it
bottleneck is a MAX over ranks → concentrating the remainder raises it
```

`affinity_layer` cuts volume 32 % and is **83 % worse than random**. A placement
objective that only maximises co-location is therefore not merely insufficient —
it is actively wrong. This is the correction to "hot experts aligned in the same
pod accelerates communication": alignment is necessary, and balance is what makes
it pay.

---

## Part 3 — What is falsified or corrected

| original statement | correction |
| --- | --- |
| "aligning hot experts in one pod accelerates communication" | alignment ↓volume (true) **but** raises the max unless jointly balanced (V4/V5) |
| "inter- and intra-layer affinity are both provable, take either" | they **conflict**: ρ = −0.32 … −0.49; the correct move is to carry **both** terms |
| "reconfiguration latency is the problem; less reconfig is better" | latency is **not binding** (0.34–4.9 passes); **ports are** (0.41 % → 8.57 %) |
| "dynamic re-planning is unjustified because plan churn is high" | churn is high (Jaccard 0.04–0.07 **[V]**) **but** re-planning is cheap — per-window re-planning is affordable, so churn is not a blocker |
| "an affinity MILP will find a better placement" | the flat MILP finds **no incumbent**; pruning the pair list to make it fit gives a *provably optimal* placement that is **59–80 % worse than greedy** [V] |
| "OCS is a feasibility question, not a headline" | it is a **regime** question: the same model spans 0.059 %–8.73 % gain depending on token ownership and DP size [V] |
| `oracle_ocs` is "a bound no controller can beat" | it is a greedy ½-approximation; measured `value_of_prediction = −0.099 %` [V] |

---

## Part 4 — What is assumed, and what that costs

### 4.1 The coupling is real, but three inputs are not measured

1. **σ (core oversubscription)** — a literal in `FabricConfig` (4.0), not a citation.
2. **Dispatch mode** — every result is `DEDUP_RANK`; under `REPLICATED` the volume
   spread across placements is **exactly 0 %** [V], so the whole volume mechanism
   disappears.
3. **Contention realism** — fixed `1/σ`, no flow counts, no per-flow cap, so a
   promoted circuit is credited with 4× unconditionally.

### 4.2 The biggest one: token → rank ownership

The destination side of the traffic matrix comes from the trace (A1 protects it).
The **source** side is invented: `(run·1_000_003 + pos) % n_dp`. [C]

| ownership / DP | OCS gain |
| --- | --- |
| hash, DP = 32 (today's default) | 4.76 % |
| per-sequence (plain data-parallel), DP = 32 | **0.088 %** |
| per-sequence, DP = 8 (< EP) | **8.12 %** |

**Span: 148×** [V]. The pipeline cannot even express DP < EP (`n_dp` defaults to
`world_size` **[C]**). Every OCS number is conditional on an assumption nothing
measures — and the measurement exists on disk (`logs/multi_tenant/*/session.json`
records 32/33 steps with ≥2 tenants co-batched).

---

## Part 5 — Which way forward: analytical pipeline or testbed?

**Not a choice — a division of labour, in this order.**

| step | what | why first | cost |
| --- | --- | --- | --- |
| 1 | **measure ownership** from the existing multi-tenant sessions | moves the OCS answer 148×; data already on disk | ~1–2 d |
| 2 | **exact circuit selection** — replace greedy `plan_circuits` with the b-matching LP | restores the oracle (currently negative); integral, so exact | ~0.5 d |
| 3 | **Tier-1 cost-model fixes** — fair-share contention `1/min(σ,k)` + per-flow cap `window/RTT` | the two idealisations the OCS claim rests on | ~2 d |
| 4 | **strengthen the co-design search** — beat its own no-circuit control on EPS first (today 7–8 % behind) | a search that can't match the generator can't beat it | ~1–2 d |
| 5 | **packet-level / testbed validation on a sample** (12–20 configs) | validate **ordering** (Spearman), not absolute times | 1–2 wk |

**Why the analytical pipeline stays the engine:** it evaluates a placement in
~0.5 ms, which is what makes the design space sweepable at all. A packet-level
simulator or a testbed predicts *time*; the analytical model predicts *ordering* —
and ordering is what every claim here actually rests on. Use the testbed as a
**gate on ordering**, not as the primary tool.

**What the testbed must record** (your list, kept): TTFT, completion time, achieved
bandwidth per tier, port occupancy, and reconfiguration events — plus, crucially,
**which rank owns each token**, since that is the one input the model invents.

---

## Part 6 — Slides

> Title: **Should optical circuits serve MoE inference? A regime question.**

**1 — Motivation**
- MoE dispatch is content-routed, rank-served: a per-layer communication round that batching cannot remove
- Where experts sit is a deployment-time decision that costs nothing at runtime

**2 — The problem, formally**
- fixed: `S(t,l)` (routing, invariant) · decided: `P` expert→rank · assumed: `O` token→rank
- traffic = `A_O · B_P` — bilinear; cost = max over ranks of tier-weighted bytes

**3 — The fabric charges by pair, not by byte**
- 450 / 50 / 12.5 GB/s (intra-node / intra-pod / cross-pod at σ=4)
- so the question is never "how many bytes" but "which pairs"

**4 — Affinity, in two projections**
- intra-layer `A[e,f]` → fewer **bytes per hop** (−21.8 % volume)
- inter-layer `P(E_{l+1}|E_l)` → fewer **hops** (−18.8 % volume, corrected) (not for fewer hops, but for the pre-establish of the next ocs switch time)
- both measured; they are the same order of magnitude

**5 — Alignment works, and alignment alone fails**

- co-location collapses messages: volume = `2·N·fanout·H·dtype` (verified exactly)
- `affinity_layer`: −32 % volume, **+83 % bottleneck vs random** — the max, not the sum

**6 — So the objective must be balance + alignment**

- `affinity_coordinated_layer`: −22 % volume, −33 % bottleneck (2 944 µs vs 4 406 random)
- affinity is the *initialiser*; the balance term is what wins

**7 — The two projections conflict**
- ρ(inter-layer objective, our bottleneck) = −0.49 / −0.32 / −0.32 across 3 models
- neither alone is a complete objective; carry both, score on one µs metric

**8 — Depth and task: how wide a window?**
- coupling is strictly 1-hop: cos 0.614 → 0.23 (flat to lag 16)
- task specificity grows with depth: within-vs-between gap +0.14 → +0.26

**9 — OCS enters as the spine layer**
- a circuit = tier promotion for one pair; it removes contention, adds no bandwidth
- applicable only where a contended cross-pod pair carries traffic

**10 — What actually binds: ports, not latency**
- 8 → 16 → 32 circuits: **0.41 % → 2.53 % → 8.57 %**
- break-even 0.34–4.9 token passes ⇒ reconfiguration is not the constraint

**11 — The result that reframes the paper**
- same switch, traffic concentrated vs spread: **12.5 % vs 6.8 %** of cross-pod bytes on the 16 hottest pairs
- **placement and circuits are substitutes**: OCS pays most where placement is worst

**12 — And the honest caveat**
- the token→rank ownership model is assumed, unmeasured, and swings the answer **148×**
- measured ownership may land near the low end ⇒ OCS becomes a scope section

**13 — What to do next**
1. measure ownership from the multi-tenant sessions (1–2 d)
2. exact b-matching LP for circuit selection (restores the false oracle)
3. Tier-1 cost model: fair-share contention + per-flow cap
4. validate **ordering** on a sample against a packet-level simulator

**14 — Takeaway**
- placement is the result today; OCS is a regime question whose regime is not yet measured
- the tools exist; the missing input is one measurement, not one model
