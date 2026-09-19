# Phase 4a/4 — the placement ordering, validated in ASTRA-sim

`scripts/astra_placement_order.py`, `scripts/chakra_from_traffic.py`, results in
`outputs/astra_sim/placement_order.qwen36.json` and `outputs/astra_sim/et/`.

This is the check the whole program was missing. `docs/astra_ordering.md` validated
the **fabric** axis (sigma, promotion, latency) on a *uniform* all-to-all, which says
nothing about placement. Our substantive claims are placement orderings on an
**uneven, placement-dependent dispatch matrix**. This runs exactly that.

## The seam now works

Read out of ASTRA-sim's own source (`system/astraccl/custom_collectives/CustomAlgorithm.cc`),
`issue()` executes `COMM_SEND_NODE` / `COMM_RECV_NODE` directly, reading
`comm_dst`, `comm_src`, `comm_size`, `comm_tag` — exactly the attributes our
generator writes — and it takes the byte counts **from the custom ET, ignoring the
workload ET's size**. Selection is by the system-JSON key
`all-to-all-implementation-custom`. So:

* our per-pair byte matrix goes in as a **custom collective**;
* a trivial workload ET only has to *trigger* the all-to-all;
* the ET paths must be resolvable **relative to the working directory** (an absolute
  path is relativised and then fails — that cost one run to discover).

The first successful execution confirms it end to end: ASTRA-sim stepped our uneven
32-rank matrix and returned per-rank wall times.

## Unit of simulation

The aggregate trace is **6.4 GB** and is unsurvivable for a packet-level simulator.
One MoE layer is **165 MB** for `linear` — the right unit, and what is used below.

## Result: model vs ASTRA-sim, one MoE layer, 32 ranks

| placement | bytes on the wire | model bottleneck (µs) | ASTRA max cycles |
| --- | --- | --- | --- |
| `linear` | 164 978 688 | 4040.4 | 159 705 |
| `load_balanced_layer` | 164 298 752 | 3808.7 | 130 713 |
| `affinity_layer` | 72 019 968 | **1155.0** | 97 754 |
| `affinity_coordinated_layer` | 78 913 536 | 1582.9 | **87 683** |

**Spearman = 0.80; 5 of 6 pairwise orderings agree.**

Both models put the two affinity placements far ahead of `linear` and
`load_balanced_layer`, and both agree on those four comparisons. Volume reduction
is visible in both: ~72–79 MB against ~165 MB.

## The one pair that disagrees is the one that decides the affinity story

| comparison | model | ASTRA-sim |
| --- | --- | --- |
| `affinity_layer` vs `affinity_coordinated_layer` | `affinity_layer` better by **27 %** | `affinity_coordinated_layer` better by **10 %** |

`affinity_layer` is the **naive** intra-layer affinity placement — the one the
repo's own multi-layer work flags as actively harmful (`affinity_layer`: −32 %
volume but **+83 % bottleneck vs random**, `docs/ocs_moe_program.md` V4). On this
single layer our model ranks it *best in the field*; ASTRA-sim ranks the
coordinated placement best.

That is a **reversal in the top-2**, which is precisely the outcome
`docs/plan_next_steps.md` §4.6 names as the one that invalidates the ordering claims.

## How much weight this carries — the honest reading

A reversal is the strongest signal we could have got, so it deserves the strongest
scrutiny before it is treated as an invalidation:

**Reasons to take it seriously.** It is not a near-tie in the model (27 % apart), it
lands on the exact comparison the affinity contribution rests on, and it is
consistent with an existing independent measurement (V4 already says
`affinity_layer` is harmful — our model is the one disagreeing).

**Reasons not to over-read it.** This is **one layer, one source model, one
backend**:

* the congestion-**unaware** analytical backend, so queueing is not modelled;
* the model's metric is max-over-ranks of tier-weighted bytes with alpha added once,
  while ASTRA-sim charges latency per rank pair (established in
  `docs/astra_ordering.md`) — the two are not comparable in absolute time, and the
  single-layer slice makes the latency term a larger share of the total than it
  would be across 40 layers;
* `promote_aware_layer` — the fifth placement, and the co-design arm — had not
  finished when this was written (it needs topo/ocs_cfg and the machine was at load
  average 9+).

So the defensible statement is: **a flagged reversal, not yet a confirmed one.** The
follow-up that settles it is cheap and specific — the same script over several
layers and all three workloads, plus the congestion-aware backend. If the reversal
holds, the cost model's placement ordering needs revision before any placement
claim is quoted, and that is a *result*, not a setback.

## Reproducing

```bash
python3.12 scripts/astra_placement_order.py --workload logs/workload/qwen36 \
    --world-size 32 --source-model measured_packed_4
```

Regenerates the per-pair ETs, runs ASTRA-sim per placement, and prints the Spearman
and the pairwise agreement count. It needs the ASTRA-sim build
(`configs/astra_sim/build_astrasim.sh`) and must be run from the repository root so
the custom-ET paths resolve.
