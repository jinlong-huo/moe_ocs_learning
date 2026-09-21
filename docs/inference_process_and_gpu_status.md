# The inference process, and what every GPU is doing while the MoE runs

Status: measured on the revision below. Companion to `docs/hot_spine_ocs.md` (the
OCS plan), `docs/ownership_measurement.md` (the measured source side),
`docs/inference_driven_net.md` (the same instrument, from the network's side: the
sigma sweep and the reconfiguration economics), `scripts/inference_net_trace.py`
(the instrument) and `scripts/gpu_state_status.py` (the report this document quotes).

Marker convention follows the repo: **[V]** produced by a command, **[C]** read
from code, **[A]** assumed, **[K]** a knob.

---

## 0. What is real here, and what is not — read this before the numbers

| | status |
| --- | --- |
| the **model** | real: `models/Qwen3.6-35B-A3B-4bit`, the same weights the rest of the repo uses |
| the **routing** | real: captured from the model itself (vLLM backend), 87 prompts, every token, every layer, every expert choice |
| the **prefill/decode split** | real: the traces carry `phase` per routing event (0 = prefill, 1 = decode) |
| the **source side** (which rank owns a token) | measured: `measured_packed_4` — the measured ≤ 4 live owning ranks per step, not the uniform hash |
| the **GPUs** | **modelled: there are no GPUs on this machine.** One Apple M1 Max (32-core GPU, 64 GB). `nvidia-smi` does not exist here |
| the **fabric** | modelled: 50 GB/s per-rank port, σ = 4 core oversubscription, 450 GB/s NVLink, 50 GB/s pod [K] |
| the **compute time** | **a parameter, never a measurement** — swept, and every row says which value produced it |

So "the status of the GPUs" below means: *the state of each of the 32 expert-parallel
ranks of a modelled deployment, driven by this model's real routing*. That is the
strongest statement this repo can make without hardware, and it is exactly the gap
`docs/hardware_validation.md` records.

Revision used (all hashes at the time of the runs; `src/net/*` was still being
edited while these were taken — re-run if a hash moved):

    engine.py 59b8c79a34ce5778277411c99ee70667      inference_net_trace.py d6c38f4e8c8ad83646b6fc2b9c9ed6ac
    fabric.py c29c3f1ea1c9516dda99fe53378a98a5      gpu_state_status.py    ccc6fe57fd81fb1e87cff8ce0b5e5f5d
    collective.py d90de8d83527c4b2655d04d091ca3f21  ocs_controller.py      4b9b5cfc8e725fdcc851eaaebefa137a

---

## 1. The model that is being run

`models/Qwen3.6-35B-A3B-4bit/config.json` **[V]**:

| quantity | value |
| --- | --- |
| layers | 40 (`num_hidden_layers`), all 40 are MoE layers |
| attention | hybrid: 3 × `linear_attention` then 1 × `full_attention`, repeating (`full_attention_interval = 4`) |
| hidden size | 2048 |
| experts | 256, top-8 per token, plus a shared expert (`moe_intermediate_size = 512`, `shared_expert_intermediate_size = 512`) |
| vocab | 248,320 |
| quantisation | affine 4-bit, group 64, with 8-bit tensors where the checkpoint says so |
| extra | one MTP (multi-token-prediction) layer in `mtp.safetensors` |

35B total, ~3B active per token: for every token the router picks 8 of 256 experts, and
only those 8 (plus the shared expert) do FLOPs. That is why the model is cheap to
*compute* and expensive to *move*: the all-to-all is the tax on the sparsity.

---

## 2. The inference process, in the two passes the traces actually contain

### 2.1 What the workload is

`logs/workload/qwen36/manifest.json` **[V]**: 87 prompts (roles: category, paraphrase,
lexical control, length ladder, repeat), prompt length 26-78 tokens (mean 38.4), exactly
64 generated tokens each, 40 layers:

    prefill cells   133,720  =  87 prompts x ~38 prompt tokens x 40 layers   (phase 0)
    decode  cells   222,720  =  87 prompts x     64 tokens     x 40 layers   (phase 1)

Both passes run the same 40 MoE layers; they differ in *how many tokens enter the
collective at once*.

### 2.2 Prefill (phase 0) — one pass over the prompt

For each of the 40 layers:

1. attention over the whole prompt (linear-attention layers are chunked/recurrent,
   full-attention layers build a KV cache);
2. the MoE block: the gate scores 256 experts per token, top-8 are taken with
   weights, and the **dispatch all-to-all** sends each token's hidden state to the
   ranks that own its chosen experts;
3. each rank runs its local experts (grouped GEMM), adds the shared expert;
4. the **combine all-to-all** returns the results to the token's owning rank;
5. residual + next layer.

A prefill pass performs 40 dispatches and 40 combines.

### 2.3 Decode (phase 1) — one token per sequence per step

Identical structure, one token per sequence: same 40 layers, same two collectives per
layer, but the matrices are one row per live sequence instead of one row per prompt
token. The per-step latency of this loop is the ITL/TPOT that a serving report quotes.

### 2.4 How the two passes compose into user-visible numbers

`src/eval/completion.py` **[C]**:

    TTFT = prefill_comm + compute + reconfig/N
    ITL  = decode_comm  + compute + reconfig/N
    throughput = n_sequences / ITL

with `compute` an explicit parameter (default 0 = report the communication part
alone, which is the honest headline) and `reconfig/N` amortising one circuit plan
over N token passes.

Measured end-to-end on this machine, for scale **[V]**
(`logs/multi_tenant/run_burst_3t/session_report.json`, real vLLM backend on the M1 Max,
3 concurrent identical requests, 32 generated tokens):

| quantity | measured |
| --- | --- |
| mean TTFT | 0.7008 s |
| mean TPOT | 0.0662 s/token |
| aggregate throughput | 33.78 tok/s |
| per-layer per-step cost implied | 66.2 ms / 40 ≈ **1.65 ms** |

That 1.65 ms is an *Apple M1 Max* number, not a datacenter-GPU number; it is used
below only as a shape for the compute/communication balance, never as a constant.

---

## 3. The GPU status while the MoE runs

Run: EP = 32 (`--world-size 32`), `multi_pod` hierarchy (8 GPUs/node, 2 nodes/pod),
`linear` placement, `measured_packed_4` source, 16-circuit OCS budget, reconfiguration
charged at 0 (G is held for the run). Layers 0-3, one pass each.

    .venv/bin/python scripts/gpu_state_status.py --workload logs/workload/qwen36 \
        --world-size 32 --layers 0,1,2,3 --policies none,static_hot,oracle_hot \
        --compute-sweep 0,200 --phase decode --out outputs/net_status --reconfig-us 0

### 3.1 The state vocabulary

Per rank, on every change of state **[C]** `src/net/engine.py`:

| state | meaning |
| --- | --- |
| `COMPUTE` | between collectives: attention + expert FFN + shared expert + norm |
| `SEND` | has flows in flight, only outgoing |
| `RECV` | only incoming |
| `SEND+RECV` | both directions at once |
| `IDLE` | nothing in flight |

### 3.2 The picture — decode pass, 4 layers, EPS only **[V]**

Execution window 5,314.5 µs = 8 collectives (4 dispatch + 4 combine) + 4 compute
windows. Strip chart (each line = one rank, left to right = time; C compute, S send,
R recv, B both, . idle):

    r 0 |BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB|
    r 1 |BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB|
    r 2 |BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB|
    r 3 |BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB|
    r 4 |RRRRRRRRRRRRRSSSSSSSSSSSSSSRRRRRRRRRRRRRSSSSSSSSSSSSRRRRRRRRRRRRSSSSSSSSSSSSRRRRRRRRRRRRSSSSSSSSSSSS|
    ...  (r4 - r31 are identical in shape)
    r31 |RRRRRRRRRRRRRSSSSSSSSSSSSSSRRRRRRRRRRRRRSSSSSSSSSSSSRRRRRRRRRRRRSSSSSSSSSSSSRRRRRRRRRRRRSSSSSSSSSSSS|

Two rank classes, and only two:

| class | ranks | status | egress |
| --- | --- | --- | --- |
| **token owners** | 0-3 | `SEND+RECV` **100 %** of the pass — they send their tokens' activations out *and* receive their own experts' results back in every single collective | 175.8 MB each |
| **expert hosts** | 4-31 | `RECV` **50 %** (dispatch) alternating with `SEND` **50 %** (combine) | 20.7 MB each, mean |

* **No rank is idle.** Not one, not for an instant: 0 ranks with idle > 1 %.
* **Egress skew is 11.94x** between the busiest and quietest rank (15.98x in the
  prefill pass). The asymmetry is entirely on the *send* side; **ingress** skew at
  rank granularity is only 1.83x (layer 0 expert load per rank: 542 - 2,553 messages,
  uniform would be 1,392).
* **Fan-out** is 7.24 distinct destination ranks per token (p50 7, max 8 = K).
* Every collective is a **barrier**: it ends when the last flow lands, so all 32 ranks
  finish together — the straggler is not one rank, it is the collective's own tail.

### 3.3 Per collective **[V]**

| coll | phase | start (µs) | network (µs) | flows | senders | receivers | both | idle | MB | peak core util |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | dispatch | 0.0 | 714.7 | 124 | 4 | 32 | 4 | 0 | 164.98 | 0.37 |
| 1 | combine | 714.7 | 714.7 | 124 | 32 | 4 | 4 | 0 | 164.98 | 0.37 |
| 2 | dispatch | 1,429.5 | 680.3 | 124 | 4 | 32 | 4 | 0 | 164.50 | 0.37 |
| 4 | dispatch | 2,790.1 | 630.4 | 124 | 4 | 32 | 4 | 0 | 159.60 | 0.35 |
| 6 | dispatch | 4,050.9 | 631.8 | 124 | 4 | 32 | 4 | 0 | 152.03 | 0.46 |

Dispatch is 4 → 32 (four owners fan out to the expert hosts), combine is the exact
mirror 32 → 4. 124 flows per collective = 4 owners x 31 peers.

### 3.4 Where the time goes when compute is switched on **[V]**

`compute_us_per_layer` is a knob, so it is swept. Mean over the 32 ranks,
decode pass:

| compute (µs/layer) | window (µs) | owners 0-3: COMPUTE / SEND+RECV | ranks 4-31: COMPUTE / SEND / RECV | what it looks like |
| --- | --- | --- | --- | --- |
| 0 | 5,314.5 | 0 % / 100 % | 0 % / 50 % / 50 % | pure communication, GPU pinned on the NIC the whole time |
| 200 | 6,114.5 | 13.1 % / 86.9 % | 13.1 % / 43.5 % / 43.5 % | compute is a thin slice; the NIC still owns the pass |
| 1,650 (= the M1 Max measurement above) | 11,914.5 | 55.4 % / 44.6 % | 55.4 % / 22.3 % / 22.3 % | compute-dominated; the NIC slice is what remains |

Every row keeps `busy% = 100.0`: adding compute time never creates idle time, it only
renames it.

Prefill pass, same procedure **[V]**: 4 collectives (layers 0-1), 98.46 MB per layer per
direction, 420-425 µs per collective, window 1,690.3 µs at compute 0, 2,090.3 µs at
compute 200, 4,990.3 µs at compute 1,650 — with the owners at 66.1 % COMPUTE / 33.9 %
SEND+RECV and ranks 4-31 at 66.1 % / 16.9 % / 16.9 % in the last of those. Same shape as
decode; only the ratio moves.

The pass comparison is worth stating plainly: **per token the two passes cost the same**
(decode 164.98 MB / 5,568 tokens ≈ 29.6 KB/token; prefill 98.46 MB / 3,343 tokens ≈
29.5 KB/token — K = 8 experts x 2048 hidden x 2 B = 32 KB, less rank-level dedup). What
differs is only how many tokens are in the matrix. In *this* captured suite the prompts
are short (≈ 38) and the generations long (64), so the decode phase carries ~1.7x the
prefill traffic. That is a property of the suite, not a law.

### 3.5 The fabric's status at the same instant **[V]**

| policy | bytes on EPS | bytes on OCS | OCS share | EPS core util (mean / max) | core saturated | window |
| --- | --- | --- | --- | --- | --- | --- |
| `none` (pure EPS) | 674.6 MB | 0 | 0 % | 0.317 / 0.461 | 0.0 % of samples | 5,314.5 µs |
| `static_hot` (8 circuits) | 592.0 MB | 82.6 MB | 12.25 % | 0.277 / 0.415 | 0.0 % | **5,314.5 µs** |
| `oracle_hot` | 541.2 MB | 133.4 MB | 19.77 % | 0.249 / 0.384 | 0.0 % | **5,314.5 µs** |

**The cross-pod core is never the binding resource** — it sits at 25-32 % utilisation
and never once reaches saturation. The binding resource is the per-rank NIC port,
shared by all ~31 concurrent flows that a token-owning rank has in flight **[C]**
`engine.py:108-132`: `rate = port_rate / (number of that rank's active flows)`.

---

## 4. The result that falls out: a circuit bought nothing *while the port bound it*

With 12-20 % of the bytes moved onto optical circuits, the finish time is **identical
to the digit**: 5,314.5 µs either way, and the same per collective (1,429.5 µs for
layer 0's dispatch+combine, in both).

That is not an accident of the arithmetic, it is the port **[V]**:

* the engine charges **every** flow to the source rank's single 50 GB/s port
  (`_port_rate` returns `port_gbytes_per_s` for the `OCS` path, and `by_src` counts all
  of a rank's flows together, circuits included);
* so a promoted pair still waits behind the same port, and removing its bytes from the
  pooled core relieves a resource that was never congested;
* promotion therefore changes *where* bytes travel, and nothing about *when* they land.

The repo's **static** cost model disagrees, and the disagreement is precise **[V]** —
same trace, same placement, same fabric constants, same 8 promoted pairs:

| layer | static bottleneck, EPS | static bottleneck, OCS | static says | execution engine says |
| --- | --- | --- | --- | --- |
| 0 | 4,040.4 µs | 3,527.4 µs | **-12.70 %** | 0.00 % |
| 1 | 3,922.2 µs | 3,550.0 µs | -9.49 % | 0.00 % |
| 2 | 3,564.3 µs | 3,281.9 µs | -7.92 % | 0.00 % |
| 3 | 3,900.8 µs | 3,548.2 µs | -9.04 % | 0.00 % |

The static model prices a pair by its **tier alone** (`CROSS_POD = nic/σ = 12.5 GB/s`,
`OPTICAL = nic = 50 GB/s`), i.e. it assumes a pair competes with exactly σ = 4 others.
The engine prices a pair by **the port it shares**, and a real all-to-all has fan-out
≈ 31, not 4. Both are internally consistent; they answer different questions.

### 4.1 The claim, scoped — it is a regime statement, not a verdict on OCS **[V]**

The zero above is the answer at σ = 4, where the core sits at 25-32 % and the port binds.
Raise σ until the *core* is the binding resource and the same circuits pay, exactly in
proportion to the bytes they move:

| σ | core pool | core saturated | pure EPS | `oracle_hot` | delta | OCS byte share |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 400 GB/s | 0.0 % of samples | 5,314.5 µs | 5,314.5 µs | **+0.00 %** | 19.77 % |
| 32 | 50 GB/s | 100 % of samples | 6,903.9 µs | 5,647.6 µs | **+18.20 %** | 18.20 % |

(the σ = 4 row: layers 0-3, this document; the σ = 32 row: layers 0-1,
`outputs/net_status_sigma32/`, reproduced with `--sigma 32`). At σ = 32 `static_hot`
gets 6,113.4 µs with an 11.45 % byte share — i.e. **the saving equals the promoted byte
share to two decimals** whenever the core is the only binding resource, and equals
**zero** whenever it is not. `docs/inference_driven_net.md` §5.1 maps the transition
(0 % at σ ≤ 8, +15.6 % at σ = 16, saturating at the byte share from σ = 32).

So the honest one-line form is: **a circuit converts core oversubscription into
bandwidth, and this deployment has none of the former to convert.**

### 4.2 The other half of the question: what a switchover costs

Everything above charges reconfiguration at 0, i.e. it assumes a configuration can be
changed for free. It cannot, and `docs/inference_driven_net.md` §5.2 measures the
consequence at the 10 ms MEMS class (16 collectives, σ = 32, first setup free): fit once
and hold gives +13.43 % (static_hot), while refitting every 4 collectives gives +3.63 %
and refitting on every collective — the oracle — gives +2.36 %, with the OCS **dark for
51.9 % and 89.2 % of the run** respectively. Chasing the instantaneous optimum loses to
fitting once and holding. That is the dynamic form of the plan-stability result, and it
is quoted here from that document rather than re-measured.

### 4.3 The counterfactual that settles the port question **[V]**

`ports_per_rank = 2` already budgets a *dedicated* port per circuit. If the model
honours that — circuits leave the electrical NIC port instead of sharing it — the same
promoted pairs recover almost exactly what the static model promised:

| policy | shared electrical port (as written) | dedicated optical port | static model |
| --- | --- | --- | --- |
| `static_hot` (8 circuits, 12.25 % of bytes) | 5,314.5 µs (**0.00 %**) | 4,904.2 µs (**-7.72 %**) | -7.9 … -12.7 % |
| `oracle_hot` (19.77 % of bytes) | 5,314.5 µs (0.00 %) | 4,530.7 µs (**-14.75 %**) | — |

Reproduce with `--ocs-dedicated-port`. This is the whole question in one line: **a
circuit is worth nothing as a re-route and worth ~8-15 % as an extra port.** The
current engine says re-route; the config says extra port; the two must be reconciled
before any OCS number in this repo is quoted against the execution model.

---

## 5. Caveats, and what would falsify this

1. **No GPUs.** Everything in §3 is a fluid model of 32 ranks fed by real routing. The
   per-rank *statuses* are exact for the model and unvalidated against hardware
   (`docs/hardware_validation.md`).
2. **One collective = one whole-layer batch.** Each layer's matrix aggregates every
   decode position of all 87 prompts, so the µs figures are *per-pass batch totals*
   under the `completion.py` convention — not per-token latency. For per-token
   latency, slice one position per sequence (`completion.first_decode_step`).
3. **`compute_us_per_layer` is a parameter.** The sweep shows the shape; only
   the 1,650 µs row has a measurement behind it, and it is an M1 Max measurement.
4. **Fabric constants are knobs** (`docs/eps_baseline.md` §3 flags σ as uncited), and
   the results are *not* σ-invariant: at σ = 4 the port binds and circuits are worth
   zero, at σ = 32 the core binds and they are worth their byte share (§4.1). Which
   regime a real deployment is in is exactly the unmeasured input, so no OCS
   conclusion should be quoted without its σ.
5. **Reconfiguration is charged at 0** in these runs. With a 10 ms-class MEMS switch
   the DARK interval dominates everything above; that half of the question lives in
   `docs/hot_spine_ocs.md` §2.1 and the epoch policies in
   `scripts/inference_net_trace.py`.
6. **Falsifiers.** (a) *executed and passed* — raising σ until the core saturates does
   make promotion pay, without any dedicated port (§4.1, `outputs/net_status_sigma32/`);
   (b) still open: a hardware run where a promoted pair does not gain a port and still
   speeds up — that would falsify §4.2's port explanation; (c) still open: an idle-rank
   census at higher EP (EP ≥ 256, where fan-out saturates) — the "GPUs waiting on
   experts" picture this suite does not produce.

---

## 6. Reproducing everything quoted here

    # the report and its CSVs (decode and prefill, with a compute sweep)
    .venv/bin/python scripts/gpu_state_status.py --workload logs/workload/qwen36 \
        --world-size 32 --layers 0,1,2,3 --policies none,static_hot,oracle_hot \
        --compute-sweep 0,200 --phase decode --out outputs/net_status --reconfig-us 0

    .venv/bin/python scripts/gpu_state_status.py --workload logs/workload/qwen36 \
        --world-size 32 --layers 0,1 --policies none,oracle_hot \
        --compute-sweep 0,200,1650 --phase prefill --out outputs/net_status_prefill

    # the counterfactual of section 4.3
    .venv/bin/python scripts/gpu_state_status.py --workload logs/workload/qwen36 \
        --world-size 32 --layers 0,1,2,3 --policies none,static_hot,oracle_hot \
        --compute-sweep 0 --phase decode --ocs-dedicated-port --out outputs/net_status_exempt

    # the raw event stream, the fabric time series and the collective records
    outputs/net_status/raw/<policy>_eps16.{gpu,fabric,collectives,circuits,reconfig}.csv
    outputs/net_status/<policy>_eps16_c<compute>.{occupancy,collective_census}.csv
    outputs/net_status/gpu_state_summary.json

    # the sigma regime of section 4.1 (core saturated: circuits pay their byte share)
    .venv/bin/python scripts/gpu_state_status.py --workload logs/workload/qwen36 \
        --world-size 32 --layers 0,1 --policies none,static_hot,oracle_hot \
        --compute-sweep 0 --phase decode --sigma 32 --out outputs/net_status_sigma32

The one open item this document deliberately does not close: whether a circuit in a
real deployment gives a rank a second port. Every OCS number in this repo is
conditional on that answer.
