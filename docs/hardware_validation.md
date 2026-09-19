# Phase 7 — hardware validation: what it needs, and why it cannot run here

Phase 7 is **not executable on this machine**, and this document records exactly
what it would take rather than simulating it and calling it validation.

## Why it cannot run here

| requirement | what we have | consequence |
| --- | --- | --- |
| **≥2 nodes with real NICs** for cross-pod contention | one Mac (MLX / vLLM-Metal) | a single host cannot produce cross-pod traffic; TTFT measured here would be single-node numbers with an injected sleep |
| **OCS hardware** (MEMS / SOA switch) | none | port count and reconfiguration frequency stay *modelled* either way |
| a fabric spanning pods | none | the tier that the entire OCS claim lives on is not physically present |

`src/runtime/*` can run on one host over gloo, but that measures our own kernel,
not a fabric.

## What a real campaign must record

Beyond the usual TTFT / completion time:

* **per-tier achieved bandwidth** (intra-node, intra-pod, cross-pod) — to check the
  450 / 50 / 12.5 GB/s constants that are currently knobs;
* **port occupancy per rank** — to check whether `ports_per_rank = 2` is the real
  constraint;
* **reconfiguration events and their durations** — to place the deployment in one of
  `RECONFIG_CLASSES` (10 ms / 1 ms / 10 µs / 0);
* **which rank owns each token** — the single input this model invents. The routing
  is real (A1, bit-exact); the source side is not.

## What to claim, and what not to

The claim that survives is **ordering, not time**. Ordering is what every result in
this program rests on, and it is the one thing an external instrument can confirm
without reproducing every constant:

* run **12–20 configurations** spanning placement × {EPS, OCS} × σ × ports;
* report **Spearman rank correlation plus an absolute-error band**;
* **a reversal in the top-5 configurations invalidates the cost model** — that is
  the outcome worth paying for, and it is why hardware money should be spent *last*;
* expected absolute error is 10–40 %; that band is not a failure.

## The cheapest substitute, and what it bought

Before hardware, `scripts/astra_ordering.py` performs the same ordering check
against ASTRA-sim on the fabric axis. It returned **Spearman = 1.000 at every
latency accounting** and a clean **0.000 %** at σ=1 (`docs/astra_ordering.md`).

Two things it did **not** cover, which remain genuinely open:

1. **the placement axis** — the grid varies σ/promotion/latency on a uniform
   all-to-all, not our placement-dependent dispatch matrix (needs Phase 4a, the
   custom collective);
2. **per-pair promotion** — the analytical backend is tier-level, so "this one rank
   pair is promoted" needs the NS-3 backend.

## Cost and sequencing

Rented 2–4 node cluster with 400 Gb/s NICs is the realistic route; a collaborator's
cluster is the cheap one. Sequencing, from `docs/ocs_moe_program.md` Part 5: spend
it **last**, after the ownership measurement (done), the plan-stability work (the
real headroom) and the ordering check on the placement axis. The model's largest
remaining error is an unmeasured *input*, not a hardware effect — so hardware
purchased now would validate the wrong thing first.
