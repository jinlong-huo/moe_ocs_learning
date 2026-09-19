# The EPS baseline, named — and what is cited vs what is a knob

Phase 2. Additive: new configs under `configs/astra_sim/`, this document. Nothing
existing was modified.

The baseline must stand on its own, independently of OCS: an EPS fabric that is
credible *before* any circuit is provisioned. That means naming the hardware and
software, and marking honestly which constants are citations and which are knobs.

---

## 1. The fabric, constant by constant

```python
# src/eval/cost_model.py, FabricConfig
intra_node_gbytes_per_s = 450.0   intra_node_latency_us = 2.0
intra_pod_gbytes_per_s  =  50.0   intra_pod_latency_us  = 5.0
nic_gbytes_per_s        =  50.0   cross_pod_latency_us  = 12.0
core_oversubscription   =   4.0   optical_latency_us    = 6.0
pod_oversubscription    =   1.0
```

| constant | value | status |
| --- | --- | --- |
| intra-node bandwidth | 450 GB/s | **vendor figure** — NVLink 4 on H100 SXM, 18 links x 25 GB/s per direction. Matches the class of published H100 numbers; re-verify against the datasheet before quoting. |
| intra-node latency | 2 us | **knob** — order of magnitude (kernel launch + NVLink); not independently measured here. |
| intra-pod bandwidth | 50 GB/s | **derived** — InfiniBand NDR 400 Gb/s / 8 = 50 GB/s per direction, one NIC per GPU. |
| intra-pod latency | 5 us | **knob** — one leaf + one spine crossing. |
| NIC rate | 50 GB/s | same as intra-pod; this is the rate a promoted circuit runs at. |
| cross-pod latency | 12 us | **knob** — three switch crossings. |
| **core_oversubscription** | **4.0** | **KNOB — no citation found.** See section 3. |
| pod_oversubscription | 1.0 | **assumption** — a non-blocking pod. Setting it > 1 is the lever that makes pod-local traffic promotable (plan T2). |
| optical latency | 6 us | **knob** — a circuit is a mirror, not a router; between intra-pod and cross-pod. |
| dispatch semantics | `DEDUP_RANK` / `DEDUP_NODE` | **named**: DeepEP-style node-limited routing with dedup, or Megatron/DeepSpeed all-to-all. |

Software side, named: vLLM backend for the routing captures (`logs/multi_tenant`),
NCCL-over-NVLink inside the node, and ASTRA-sim + Chakra for the external
cross-check (`configs/astra_sim/`).

## 2. ASTRA-sim configs that actually match

```
configs/astra_sim/eps_baseline_2tier.yml            Switch 8 @450 / Ring 4 @12.5, lat 2000/12000
configs/astra_sim/eps_baseline_2tier_promoted.yml   same, cross tier at 50 (whole-tier promotion)
configs/astra_sim/eps_baseline_sigma1.yml           sigma=1 falsifier
configs/astra_sim/eps_baseline_sigma2.yml           sigma=2 envelope point
configs/astra_sim/eps_baseline_sigma8.yml           sigma=8 envelope point
```

Note: the 32-NPU shape has no matching workload in the checkout (the
microbenchmarks are 4/8/16 NPUs), so the runnable configs use 16 NPUs; the
*bands* are FabricConfig's, which is what the cross-check is about.

### Units, measured rather than assumed

Running the 16-NPU all-to-all toy at four latencies:

```
latency(ns)     0      500     2000    12000
cycles      75620   195620   555620  2955620
```

That is exactly `cycles = 75620 + 240 x latency` (240 = 16 x 15 rank pairs), so:

* `latency` is in **ns** (1 cycle = 1 ns at the analytical backend's clock);
* ASTRA-sim charges it **once per rank pair per iteration**, while
  `cost_model.evaluate` adds alpha **once** to the bottleneck.

That is a structural difference, not a unit conversion. **The previously
committed `ring16_*.yml` configs used `latency: 500` (= 0.5 us) where
`FabricConfig` says 12 us** — a 24x mismatch on the tier the entire OCS claim
lives on. The corrected configs use 12000.

## 3. The defensibility gate: sigma = 4 fails it

`docs/plan_next_steps.md` T2 makes this a **hard** gate: *"sigma > 1 at the pod
tier must be backed by cited fabric numbers... If it cannot be defended, this
lever is dropped and the win must come from T1 at EP > pod."*

A literature search for a citable rail-optimized oversubscription ratio returned
vendor material and blog posts only — no reference-architecture number that can be
quoted. **So sigma = 4 remains a knob, and every sigma-dependent number stays an
envelope over sigma in {1, 2, 4, 8}, never a point.** This is the honest reading
of the gate, and it is now recorded rather than assumed.

Consequence: the win must be defended on the **EP/pod** and **DP/EP** axes (both
deployment facts), with sigma swept — the shape of outcome S2.

## 4. What "authentic" now means, precisely

* The **structure** is defensible: three tiers, one NIC per GPU, tier-aware
  charging, a max-over-ranks bottleneck, a named dispatch semantic.
* The **bandwidths** are defensible (vendor NIC/NVLink rates).
* **Two constant families are not**: sigma, and the latencies. Both are swept, and
  every claim depending on them is reported as an envelope.
* The **routing** is real and gated (A1: bit-exact, independent of topology and
  placement) — still the strongest part of the pipeline.
