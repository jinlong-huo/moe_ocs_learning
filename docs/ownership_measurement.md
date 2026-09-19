# Ownership, measured — and what it does to the OCS answer

Status: measurement done, replaces an invented input. **Additive only**: this
document and four new scripts under `scripts/`; no existing module, config or
result was modified. Every number below has a reproducing command.

Companion to `docs/plan_next_steps.md` (P1) and `docs/ocs_moe_program.md` §4.2.

---

## 0. The premise being tested

```docs/ocs_moe_program.md``` §4.2 names the biggest assumption in the program:
the **source side** of the traffic matrix. The destination side comes from the
trace (protected by A1 — routing is a pure function of inputs and weights). The
source side is invented:

```python
# src/eval/cost_model.py:311
src = (run * 1_000_003 + pos) % n_dp          # uniform over DP ranks, by construction
```

That hash decides which rank *owns* a token, hence which **pairs** exist, hence
which pairs a circuit can promote. The audit priced the unknown at **148×**
(4.76 % → 0.088 %). This is the measurement that replaces it.

## 1. What is on disk, and what it actually measures

Two captured multi-tenant serving sessions — real vLLM backend, real Qwen3.6
routing, concurrent co-batched requests:

| run | tenants | steps | tokens | concurrency per step |
| --- | --- | --- | --- | --- |
| `logs/multi_tenant/run_burst_3t` | 3 | 33 | 210 | 3 in 31/33 steps (mean 2.91) |
| `logs/multi_tenant/run_burst_4t` | 4 | 33 | 275 | 4 in 31/33 steps (mean 3.88) |

```bash
python3.12 scripts/measure_ownership.py --sessions logs/multi_tenant --world-size 32
```

**Measured:** which sequences are co-resident per step, how many tokens each
contributes, arrival order, and the per-token per-layer expert routing of every
co-batched sequence.
**Assumed:** how a deployment maps live sequences onto DP replicas. The capture
does not record replicas, so the script reports the **support** — the number of
live sequences, an upper bound on live owning ranks — and the envelope sweeps
the one unknown it leaves (packed vs spread; see §3).

### The number that matters

| | measured | the hash assumes |
| --- | --- | --- |
| live owning ranks per step | **max 4** (3-tenant run: max 3) | **32** |
| tokens per replica, Gini | **0.88** | **0.00** |
| same-replica token share | 0.040 | 0.031 |

The hash spreads tokens uniformly over all 32 ranks. The capture shows **≤4
ranks carrying tokens at any instant** — an 8× smaller source support, and a
distribution that is concentrated (Gini 0.88) rather than flat.

### A second measured fact: co-batched requests overlap heavily in experts

Mean Jaccard of the *union of top-8 expert sets* between two co-batched tenants,
same step:

| layer 0 | layer 20 | layer 39 |
| --- | --- | --- |
| 0.37 – 0.41 | 0.44 – 0.55 | 0.37 – 0.43 |

Concurrent requests are not routing to disjoint expert sets, which is why
concentrating the source side concentrates the *pairs* too.

## 2. What it does to the OCS gain — DP = EP, the deployed case

```bash
python3.12 scripts/ownership_ocs_envelope.py --workload logs/workload/qwen36 --world-size 32
python3.12 scripts/ownership_ocs_envelope.py --workload logs/workload/qwen15 --world-size 30
python3.12 scripts/ownership_ocs_envelope.py --workload logs/workload/whittle --world-size 32
```

Static gain, `linear` placement, DP = EP, 16 circuits / 2 ports per rank:

| workload | hash (the assumption) | per_sequence | **measured (packed)** | **measured (spread)** |
| --- | --- | --- | --- | --- |
| Qwen3.6, W=32 | 4.76 % | 0.088 % | **9.46 %** | **8.40 %** |
| Qwen1.5, W=30 | 1.90 % | 5.11 % | **9.80 %** | **9.41 %** |
| Whittle, W=32 | 8.46 % | 0.028 % | **8.47 %** | **8.62 %** |

Read across the table, not down it:

* The assumption it replaces spreads **1.9 % – 8.5 %** across models.
* `per_sequence` — the model the earlier audit leaned on as the realistic
  pessimistic case — spreads **0.03 % – 5.1 %**.
* The **measured** regime gives **8.4 % – 9.8 %**, and it is the same number in
  all three models.

Mechanism, from the same rows:

| | hash | measured |
| --- | --- | --- |
| candidate promotable pairs | 224 – 256 | **55 – 64** |
| promotable traffic covered by 16 circuits | 0.066 – 0.072 | **0.118 – 0.150** |

Concentrating the source side onto a few ranks concentrates traffic onto fewer
pairs, so the *same* switch budget covers **~2× more** of the traffic.

## 3. The plan formula is wrong, and the stability measurement says why

```docs/plan_next_steps.md``` §0 prices a circuit as
```Delta_oracle = f * (1 - 1/sigma)``` with `f` = the promotable share of
bottleneck bytes. Measuring the attribution directly
(`scripts/bottleneck_attribution.py`, read out of `evaluate`'s own arithmetic,
nothing modified):

| source | binds | promotable share of the binding resource | gain |
| --- | --- | --- | --- |
| hash | ingress NIC | 0.51 | 4.76 % |
| per_sequence | egress NIC | 0.51 | **0.088 %** |
| measured (packed) | egress NIC | 0.52 | 9.46 % |
| measured (spread) | egress NIC | 0.53 | 8.40 % |

`f` is **0.51 in every case** while the gain spans two orders of magnitude, so
`f` is not the operative variable. The circuit is always the NIC (never NVLink),
and always about half the binding rank's bytes are on promotable pairs — what
differs is whether the **plan** puts the circuits on the pairs that bind.

That is decided by window drift. Plan fitted on window A, wanted on window B,
16 circuits (`scripts/plan_window_stability.py`):

| source | Jaccard plan(A) vs plan(B) | shared / 16 |
| --- | --- | --- |
| hash | 0.03 – 0.14 | 1 – 4 |
| **per_sequence** | **0.0000** | **0** |
| measured (packed) | 0.07 – 0.33 | 1 – 4 |
| **measured (spread)** | **1.0000** | **4 / 4** |

`per_sequence` is not a pessimistic *level*, it is a **stale plan**: the two
disjoint windows share not one circuit, so covering 10 % of promotable traffic
moves the bottleneck by 0.03 %. Under the measured regime the pair set is small
enough that the plan is **identical across windows** — the 8–10 % is structural,
not a lucky fit.

Corrected statement: `Delta = c * (1 - 1/sigma) * [plan still valid on the scoring
window]`, where `c` is the covered fraction — not `f`.

## 4. What this changes

* **S2 is the landing, and it is stronger than expected.** The ownership unknown
  does not collapse the OCS story; measured, the story is *more* stable than
  either assumption it replaces (8.4–9.8 % across three models).
* **The "148×" audit item closes** — the swing was mostly *plan staleness*, not
  ownership per se. That reframes the fix: the requirement is not "measure
  ownership" alone, it is "**hold a plan that survives the window**".
* **Inter-layer affinity gets a sharper job** (A2): predicting the next window's
  hot pairs is exactly what converts a stale plan into a valid one, and the
  measured regime is the one where a small, stable pair set makes that tractable.
* **P6/P7 ordering validation is now cheaper to interpret**: any ASTRA-sim
  comparison must use the measured source support, or it validates the hash.

## 5. Honest limits

1. **k = 4 is an inference.** Four concurrent *sequences* is measured; "four
   owning *ranks*" additionally assumes one sequence per replica. Both packing
   extremes are reported, and they agree to ~1 point (8.4 vs 9.5), so the
   conclusion does not rest on that choice.
2. **The capture is one node.** It cannot exhibit cross-pod ownership; the
   workload traces supply the fabric, the capture supplies the source structure.
3. **Static gain only.** The oracle plan is a greedy ½-approximation
   (`value_of_prediction = −0.099 %`), so the headroom above these numbers is
   unquantified until the exact b-matching LP lands (Phase 6).
4. `n_dp` still cannot be set through `ocs_comparison` — the DP < EP sweep
   drives `evaluate` directly, as the existing sensitivity script does.

## Reproducing everything

```bash
bash scripts/check_env.sh                                   # 33 tests, pinned interpreter
python3.12 scripts/measure_ownership.py --sessions logs/multi_tenant --world-size 32
python3.12 scripts/ownership_ocs_envelope.py --workload logs/workload/qwen36 --world-size 32
python3.12 scripts/bottleneck_attribution.py --workload logs/workload/qwen36 --world-size 32
python3.12 scripts/plan_window_stability.py --workload logs/workload/qwen36 --world-size 32
```

Outputs land in `outputs/ownership/`. Scripts are additive; the two that touch
pipeline state (`ownership_ocs_envelope.py`, `plan_window_stability.py`,
`bottleneck_attribution.py`) monkey-patch `cost_model.token_rank` and restore it
in a `finally`.
