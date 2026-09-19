# ASTRA-sim ordering check — Phase 4

`scripts/astra_ordering.py`, results in `outputs/astra_sim/ordering.json`.
Additive: new script, new outputs, this document.

## What is validated, and what is not

Every claim in this program rests on an **ordering**: "configuration A beats
configuration B". The closed-form model produces that ordering in ~0.5 ms;
ASTRA-sim produces a time in seconds. The claim is *ordering preserved*, with an
absolute-error band — never equality of times.

**Not validated here: the placement axis.** The grid varies the fabric (sigma,
promotion, latency) on a uniform 1 MB all-to-all. Our substantive claims are
placement orderings on a placement-dependent, uneven dispatch matrix, which needs
the custom-collective seam (Phase 4a) — **still not built**. What is validated is
the fabric mechanism and its ordering.

## The structural finding first

Measured on the 16-NPU toy: `cycles = 75620 + 240 x latency` (240 = 16 x 15 rank
pairs). ASTRA-sim charges latency **per rank pair per iteration**; our model adds
alpha **once**. Consequence: the promoted/unpromoted *ratio* is set by which
latency accounting you use, not by bandwidth physics.

| 4x bandwidth promotion (50 -> 12.5 GB/s) at latency | speedup |
| --- | --- |
| 0 ns (bandwidth only) | **3.90x** |
| 500 ns — what the committed `ring16_*.yml` used | **2.12x** |
| 12000 ns — what `FabricConfig` says | **1.07x** |

So the committed "first cross-check" (2.12x / 2.99x in `configs/astra_sim/README.md`)
is **not transferable**: it was measured at a latency 24x below FabricConfig's.
Its qualitative finding survives (contention exists only when oversubscribed);
its number does not.

## The grid and the result

24 configurations: sigma in {1,2,4,8} x latency in {500, 2000, 12000} ns x
{EPS, promoted}, single-tier ring, 16 NPUs, 1 MB all-to-all. Model column is
`Delta = c * (1 - 1/sigma)` with `c = 0.1399`, the covered fraction measured under
**measured ownership** (`docs/ownership_measurement.md`).

| sigma | bw_eps (GB/s) | ASTRA-sim @500 ns | @2000 ns | @12000 ns | model |
| --- | --- | --- | --- | --- | --- |
| 1 | 50.0 | **0.000 %** | **0.000 %** | **0.000 %** | 0.000 % |
| 2 | 25.0 | 27.230 % | 11.641 % | 2.417 % | 6.995 % |
| 4 | 12.5 | 52.888 % | 28.327 % | 6.916 % | 10.492 % |
| 8 | 6.25 | 72.380 % | 47.988 % | 14.781 % | 12.241 % |
| | **Spearman** | **1.000** | **1.000** | **1.000** | |

Three findings:

1. **The ordering is preserved exactly** — Spearman = 1.000 at all three latency
   accountings; sigma=8 > sigma=4 > sigma=2 > sigma=1. This is the property every
   claim rests on, and it does not depend on the latency disagreement.
2. **sigma = 1 gives exactly 0.000 %.** With no oversubscription the promoted
   config is byte-identical to the EPS config, so a circuit buys nothing. That is
   the falsification row for the promotion mechanism, and ASTRA-sim returns it
   cleanly: *a circuit removes contention, it does not add bandwidth.*
3. **Magnitudes disagree by up to ~2x**, and the disagreement grows with latency
   (at 12 us ASTRA-sim is more pessimistic than the model at every sigma; at
   500 ns far more optimistic). The model sits closer to the middle of the band
   than to either end — not obviously wrong-signed, but its absolute gain is not
   trustworthy to better than a factor of ~2 until the latency accounting is fixed.

## What this changes

* The **ordering claim passes its external check** on the fabric axis — the check
  `docs/plan_next_steps.md` section 4.6 requires for a headline.
* The **absolute-gain claim does not**: any "OCS cuts MoE all-to-all by X %" must
  carry the latency accounting, or be reported as a band.
* `configs/astra_sim/README.md`'s speedup table is **superseded** — kept for the
  record, with the latency mismatch named here rather than silently corrected.

## Remaining work on this phase

* **4a — the custom collective**: `examples/system/custom_collectives/` +
  `run_analytical_with_custom_collective.sh`, replaying our per-layer per-pair
  byte matrix instead of a uniform all-to-all. This is what would validate the
  *placement* ordering, the claim we actually make.
* **4c — per-pair promotion**: the analytical backend is tier-level; NS-3 per-link
  JSON is needed to promote one rank pair rather than the whole tier.
* **4d — reconfiguration cost**, modelled by neither config.
* **Latency accounting**: either our model charges alpha per pair, or ASTRA-sim is
  configured to charge it once. Until then the two are not comparable in absolute time.
