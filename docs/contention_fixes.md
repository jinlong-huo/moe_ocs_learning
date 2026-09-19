# Phase 3 — the two contention fixes, and which one actually moves the number

`src/eval/contention.py` (new module), `scripts/contention_before_after.py`,
results in `outputs/ownership/contention_before_after.qwen36.json`.
Additive: the overlay IMPORTS `cost_model` and does not modify it, so
`evaluate()` remains the reference and every delta is attributable.

## The two idealisations

`docs/plan_next_steps.md` step 3 asks for two fixes: a fair-share contention term
`1/min(sigma, k)` and a per-flow rate cap `window/RTT`. The ASTRA-sim cross-check
independently says they matter: on the 16-NPU toy `cycles = busy + 240*latency`,
so a third of the time is **not** bandwidth-proportional (`docs/astra_ordering.md`).

The overlay implements both, each independently switchable, so the delta is
attributable rather than a single tuned number.

### Self-check first

With both fixes OFF the overlay must reproduce `evaluate()` exactly. It does —
`rel_err = 0.0e+00` on all three source models. That check earned its keep: it
caught three real bugs of mine during development (intra-node bytes counted in
the NIC drain; the CORE sigma applied to the POD tier; and a silent
mis-broadcast of `sigma_t (D,W)` against `k_egress (D,)` that produced a 3-D rate
matrix and garbage drains). There is now an `assert rate.shape == net.shape`
guarding the last one.

## Result: fix 1 is inert at these operating points

Qwen3.6, W=32, sigma=4, 16 circuits, static gain (%):

| source model | placement | current | fair_share | +cap@64 KiB | @256 KiB | @1 MiB | @4 MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hash | linear | 4.755 | **4.755** | 4.417 | 4.586 | 4.755 | 4.755 |
| per_sequence | linear | 0.088 | **0.088** | 0.018 | 0.072 | 0.088 | 0.088 |
| measured_packed_4 | linear | 9.462 | **9.462** | 5.871 | 5.895 | 9.462 | 9.462 |
| hash | affinity_coord | 2.528 | **2.528** | 2.253 | 2.490 | 2.528 | 2.528 |
| measured_packed_4 | affinity_coord | 8.669 | **8.669** | 5.356 | 5.386 | 8.669 | 8.669 |

**Fair share changes nothing at all.** That is not a bug — it is a measurement of
the operating point. A rank's share is `nominal / max(1, min(sigma, k))`, and the
binding ranks here have **k_egress = 31 peers** (k_ingress is 3–4). With
k >= sigma the correction *equals* the mean-field rate `nominal/sigma`, so the
formula returns exactly what the model already charged.

The corollary is the useful part: **the mean-field 1/sigma is not the binding
idealisation for this traffic.** A rank that broadcasts to most of the world
really is contended at ~1/sigma. The plan's fix 1 would matter only for ranks
with fewer peers than sigma — and there are none on the critical path here.

## Result: fix 2 bites, with a sharp break-even

The per-flow cap is `window / RTT(tier)`, with RTT = 2 x tier latency:

| tier | latency | RTT | 50 GB/s needs |
| --- | --- | --- | --- |
| cross-pod (EPS) | 12 us | 24 us | 300 KB in flight |
| optical (promoted) | 6 us | 12 us | **600 KB in flight** |

So a promoted pair is BDP-limited below ~600 KB of outstanding data and
unconstrained above it. The measurement brackets it exactly: **256 KiB binds
(gain 9.46 % -> 5.89 %), 1 MiB does not (9.46 %)**.

Read that as a design requirement rather than a defect:

> A circuit only delivers its bandwidth if a single flow can keep ~600 KB in
> flight. Below that, promotion buys about half of what the model claims; above
> it, the model is right and the circuit is uncontended.

That is exactly the "one flow cannot fill a dedicated circuit" objection, now
with a number attached — and it converts a modelling assumption into an
engineering requirement on the flow/window side.

## What this changes

* **Fix 1 (fair share) is demoted**: implement it, but do not expect a number to
  move at DP = EP; it is relevant only for many-peer-poor topologies.
* **Fix 2 (per-flow cap) is promoted to a stated precondition**: every OCS claim
  should name the outstanding-bytes regime it assumes, because the answer changes
  by ~40 % across it.
* **The model is not rescued by either fix at DP = EP**: the gain at a realistic
  window (>=1 MiB) is unchanged, so the measured 9.46 % stands — with the
  precondition attached.

## Limits

* `window_bytes` is a per-flow outstanding-bytes figure; the overlay treats one
  communicating rank pair as one flow. A real deployment may run several flows
  per pair, which would raise the effective window.
* The fair-share term uses **peer count** as `k`. A finer `k` would be the number
  of simultaneously active flows on the port at that instant, which needs a
  timeline the cost model does not have.
* Only Qwen3.6/W=32 was swept; Qwen1.5 and Whittle are one command away
  (`--workload`).
