# Inference-driven network observation (EPS + OCS)

Status: **new work, new files only.** Nothing under `src/eval`, `src/ocs`,
`src/comm` or `scripts/` existing is modified. This document describes the
instrument and its first measurements.

    new  src/net/__init__.py        the package and why it exists
    new  src/net/fabric.py          EPS pooled core + OCS configuration/epochs/DARK
    new  src/net/collective.py      inference event -> routed flows
    new  src/net/ocs_controller.py  who gets a circuit, and when G changes
    new  src/net/engine.py          the inference loop with the network inside it
    new  scripts/inference_net_trace.py   runner: CSVs, JSON, figure, ASCII trace
    new  docs/inference_driven_net.md     this file

---

## 1. The distinction that motivates it

`src/eval/cost_model.py` answers *"how much traffic does this workload generate,
and what is its bottleneck"*. It takes a static per-pair byte matrix and returns a
scalar. That is the right object for comparing placements and the wrong object for
watching a fabric: it has no time, no queue, no link state, and no notion of a
circuit existing or not existing at an instant. Feeding a packaged workload to a
simulator and reading an average load is the same abstraction with more steps.

This instrument asks the other question: **when this particular collective happens,
what does the network actually do?** The network is inside the inference loop:

`~text
layer -> compute -> collective (dispatch) -> network -> collective (combine) -> next layer
                                 ^
                                 |
                   G(t), queues, port shares, circuit membership
`~

and at every instant of that execution it records who is doing what.

## 2. What is recorded

| family | fields | file |
| --- | --- | --- |
| **GPU** | rank, t, state (COMPUTE / SEND / RECV / SEND+RECV / IDLE), collective, n_send, n_recv — written **on change** | \`<tag>.gpu.csv\` |
| **EPS** | active flows, EPS vs OCS split, core GB/s + **utilisation**, core queue MB, max per-port queue MB | \`<tag>.fabric.csv\` |
| **OCS** | generation, src, dst, epoch start/end, bytes carried, busy ns | \`<tag>.circuits.csv\` |
| **switchover** | collective, t, torn down, established, dark ns | \`<tag>.reconfig.csv\` |
| **collective** | id, phase, start, finish, network ns, bytes, n flows, **EPS vs OCS bytes**, peak core utilisation, peak port queue | \`<tag>.collectives.csv\` |

Anything asked of the network later — "link 2 at 80 %, circuit A→B active at 10.05 ms"
— is a query over these five tables.

## 3. Model semantics

Time is nanoseconds; the integration is exact because between two events the active
flow set is fixed and so is every rate. Per flow the rate is max-min fair over four
resources:

| resource | EPS / OCS flow | NVLink | pod |
| --- | --- | --- | --- |
| egress port at the source | 50 GB/s shared by that rank's active flows | 450 | 50 |
| ingress port at the destination | idem | 450 | 50 |
| path | **EPS**: pooled core \`R*W/sigma\` shared by active core flows | — | — |
| | **OCS**: the circuit, full NIC rate, **not drawn from the pool** | — | — |

Two consequences carry the whole contrast:

* **EPS re-shares continuously.** Its utilisation is \`min(1, demand/capacity)\`: under
  light load a flow runs at full port rate, under saturation it is scaled down. At
  full saturation this reduces exactly to this repo's \`FabricConfig\`
  (\`R*W/sigma\` over \`W\` ranks = \`R/sigma\` = 12.5 GB/s per pair at \`sigma = 4\`);
  under partial load it does **not**, and that difference is a result, not a detail.
* **OCS does not react at all.** A configuration G persists until it is replaced; a
  circuit is ACTIVE or it does not exist; a switchover is a **DARK** interval during
  which every cross-pod flow falls back to EPS. This is the timescale contrast, made
  operational.

## 4. Commands

`~bash
python3.12 scripts/inference_net_trace.py --workload logs/workload/qwen36 \
    --world-size 32 --layers 0,1,2,3 --passes 2 \
    --policies none,static_hot,epoch_hot,oracle_hot \
    --n-circuits 16 --sigma 32 --reconfig-us 10000 --epoch-collectives 4
`~

Policies differ in **what they are allowed to know**: `none` (pure EPS control),
`static_hot` (fit once on the calibration collectives, then hold),
`epoch_hot` (refit every N collectives from traffic actually seen),
`oracle_hot` (refit on the collective about to run — an upper bound, not deployable),
`threshold` (rank by the congestion each pair actually suffered).

## 5. Results

Qwen3.6, W=32, \`measured_packed_4\`, \`linear\`, layers 0–3, 32-rank multi-pod.

### 5.1 The OCS pays only when the core is the binding resource **[V]**

\`oracle_hot\`, 16 circuits, two layers, reconfiguration free (\`outputs/net/sigma_sweep.json\`):

| sigma | core pool | pure EPS | with OCS | delta | bytes on OCS |
| --- | --- | --- | --- | --- | --- |
| 4 | 400 GB/s | 2 790.1 us | 2 790.1 us | **+0.00 %** | 18.2 % |
| 8 | 200 GB/s | 2 790.1 us | 2 790.1 us | **+0.00 %** | 18.2 % |
| 16 | 100 GB/s | 3 524.0 us | 2 975.7 us | +15.56 % | 18.2 % |
| 32 | 50 GB/s | 6 903.9 us | 5 647.6 us | +18.20 % | 18.2 % |
| 64 | 25 GB/s | 13 807.8 us | 11 295.1 us | +18.20 % | 18.2 % |

At \`sigma <= 8\` the pooled core peaks near 30 % utilisation and the binding resource
is the **hot source ranks' own NIC ports** (ranks 0–3 carry 28–31 flows each): moving
18 % of the bytes onto circuits changes the makespan by exactly nothing, because a
circuit removes *core oversubscription*, not port rate. Past \`sigma = 16\` the core
binds and the same circuits buy 15.6 %; from \`sigma = 32\` the benefit **saturates at
the promoted byte share** (18.2 %), which is \`Delta = c*(1 - 1/sigma)\` in the limit.

The static cost model cannot produce either sentence, because it charges every byte
1/sigma whether or not the core is saturated.

### 5.2 Cost of switching versus value of fitting **[V]**

Same trace, 4 layers x 2 passes = 16 collectives, \`sigma = 32\`, **10 ms**
reconfiguration class, first establishment free (setup before the window):

| policy | network total | vs pure EPS | bytes on OCS | switchovers | DARK wall time |
| --- | --- | --- | --- | --- | --- |
| none | 26 984.1 us | — | 0 % | 0 | — |
| **static_hot** | **23 360.2 us** | **+13.43 %** | 13.4 % | 1 | 0 |
| epoch_hot (every 4) | 26 004.5 us | +3.63 % | 3.6 % | 3 | **51.9 % of the run** |
| oracle_hot (every collective) | 26 347.3 us | +2.36 % | 2.4 % | 8 | **89.2 % of the run** |

**Chasing the instantaneous optimum is worse than fitting once and holding**, because
each switchover blinds the OCS for 10 ms while the collectives keep arriving. The
oracle — perfect knowledge of the next collective — ends up with the lowest OCS
utilisation of the four, and the OCS is dark for the entire run. This is the dynamic
form of the plan-stability result in \`docs/ownership_measurement.md\`, and it is a
statement about the *reconfiguration class*, not about the policy.

### 5.3 The trace, as text **[V]**

\`sigma = 32\`, \`none\` policy, 10 instants across the run — the format requested:
instant, which collective is in flight, what each GPU is doing, what the fabric is doing.

`~text
t=      0.0 us  coll  0 dispatch | SEND  0 RECV 28 SR  4 COMPUTE  0 IDLE  0 | core 100.0% q=  69.7MB OCS  12.9GB/s
t=   1202.7 us  coll  0 dispatch | SEND  0 RECV 28 SR  4 COMPUTE  0 IDLE  0 | core 100.0% q=  11.3MB OCS   0.0GB/s
t=   2405.4 us  coll  1 combine  | SEND 28 RECV  0 SR  4 COMPUTE  0 IDLE  0 | core 100.0% q=  22.6MB OCS   7.1GB/s
t=   3608.1 us  coll  2 dispatch | SEND  0 RECV 28 SR  4 COMPUTE  0 IDLE  0 | core 100.0% q=  32.2MB OCS  18.7GB/s
t=   4810.8 us  coll  3 combine  | SEND 28 RECV  0 SR  4 COMPUTE  0 IDLE  0 | core 100.0% q=  41.9MB OCS  19.8GB/s
t=   6013.5 us  coll  4 dispatch | SEND  0 RECV 28 SR  4 COMPUTE  0 IDLE  0 | core 100.0% q=  42.8MB OCS  17.8GB/s
`~

Dispatch and combine alternate and mirror each other (28 receivers, then 28 senders);
the core sits at 100 % once \`sigma = 32\`; the queue builds and drains inside each
collective; and the OCS carries 0-20 GB/s depending on whether the current
collective's demand matches the configuration it is holding.

### 5.4 Who owns the tokens decides the answer **[V]**

The instrument was run with four different token-ownership models, same trace, same
placement, same fabric. Nothing else changed.

| ownership of tokens | ranks that send | directed links | biggest sender | its share of all traffic |
| --- | --- | --- | --- | --- |
| `hash` (spread over all 32) | **32** | 992 | 5.3 MB | 3 % |
| `per_sequence` | **32** | 992 | 5.8 MB | 4 % |
| `measured_packed_4` | **4** | 124 | 42.3 MB | **26 %** |
| `measured_spread_4` | **4** | 124 | 42.3 MB | 26 % |

and the resulting network time and OCS value (one layer, 16 circuits, oracle policy):

| ownership | sigma | pure EPS | with OCS | gain | bytes on OCS |
| --- | --- | --- | --- | --- | --- |
| hash | 4 | 487.4 us | 448.1 us | +8.06 % | 8.3 % |
| hash | 32 | 3 382.6 us | 3 100.5 us | +8.34 % | 8.3 % |
| measured_packed_4 | 4 | 1 429.5 us | 1 429.5 us | **+0.00 %** | 18.2 % |
| measured_packed_4 | 32 | 3 492.1 us | 2 855.2 us | +18.24 % | 18.2 % |

Two regimes, opposite conclusions:

* **Few owners (the measured, small-batch case).** Four ranks carry 26 % of all
  traffic each and fan out to 28-31 peers. Their own ports are the wall (1 429 us
  against 487 us for the same bytes spread evenly), and at normal oversubscription a
  circuit buys **exactly nothing** -- it does not widen a port. Only when the core is
  pushed hard (sigma = 32) does the concentrated traffic make the core bind, and then
  the same circuits buy 18 %.
* **Many owners (the hash / large-batch case).** Load is spread, the fabric is fast,
  and the OCS buys a steady **8 %** at both oversubscriptions -- less than the
  concentrated case at high sigma, because nobody is congested enough for a circuit
  to matter.

Where the "4" comes from, in order: the multi-tenant capture measured **at most 4
sequences alive at once**; the `packed` model assumes one sequence per DP replica;
`DP = EP = 32` and `linear` expert placement then force those few owners to send to
nearly every rank. It is not a mis-set parameter, but it is a *small-batch* setting,
and `ocs_moe_program.md` §4.2 already names the source side of the traffic matrix as
the largest assumption in the program. A serving deployment with hundreds of
concurrent requests would look like the hash row; a four-tenant co-batch looks like
the measured row. The honest statement is therefore a curve over the number of
owning ranks, not a single number -- which is the next measurement, and it is one
flag away (`--source-model`).

## 6. Limits

1. **Fluid, not packet-level.** No packetisation, no per-packet queueing, no
   retransmission; a "queue" is the remaining bytes of a flow. The engine is exact
   for the model it implements, and that model is a fair-share abstraction of EPS.
2. **The EPS core is one pooled resource.** Real Clos fabrics spread each flow over
   parallel planes; the pooled core is the mean-field version of that, and its
   capacity (\`R*W/sigma\`) is the same uncited constant as before — except that it is
   now *not* applied per byte, so the instrument can show when it matters.
3. **Compute is a knob and defaults to 0** (communication-only), matching
   \`src/eval/completion.py\`'s discipline. Raise it with \`--compute-us\` to see COMPUTE
   states and to amortise reconfiguration over a longer window.
4. **One placement, one ownership model, four layers, two passes of one stationary
   workload.** This is a first measurement, not a sweep.
5. **The OCS controller has no cost model of its own**: `oracle_hot` is explicitly a
   bound; `static_hot` and `epoch_hot` observe only traffic they have already seen.
6. **Promotion semantics are the repo's**: a circuit promotes a whole unordered rank
   pair, tier-level. Nothing here models a partially loaded circuit.