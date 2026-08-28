# Assumption Ledger

Every assumption this testbed relies on, what it claims, how it is verified,
where the verdict lives, and what is still open. Each entry is tied to one
verification script (a "phase") with a JSON report; `README.md` has the
condensed version.

| #  | Assumption                                                                                                  | Status                                                        | Gate script                                                                            | Report                                                                      |
| -- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| A1 | Routing is a pure function of (input, weights) — independent of topology, node distribution, and engine | ✅ verified (live + hardware) | `scripts/verify_live_invariance.py` (§topology/placement), `scripts/compare_backend_traces.py` | `logs/live_invariance_report.json`, `logs/phase2/invariance_report.json` |
| A2 | Affinity is model×input specific — OCS presets must be re-derived per model                               | ✅ verified (model level); ⚠️ quantization-level drift open | `scripts/compare_model_affinity.py`                                                  | `logs/phase3/model_diversity_report.json`                                 |
| A3 | Per-cell routing is noisy (quantized-GEMM near-ties); trust distribution-level metrics only                 | ⚠️ empirically established, per-cell decisions avoided      | `scripts/vllm_serve.py affinity` (calibration families)                              | `logs/multi_tenant/run_*/affinity_report.json`                            |
| A4 | Placement is a cost-side variable — recorded affinity can safely configure expert→rank and rank→location | ✅ verified (live payoff) | `scripts/verify_live_invariance.py` (payoff) | `logs/live_invariance_report.json` (payoff) |
| A5 | Multi-tenant co-batching creates measurable expert contention                                               | ✅ measured                                                   | `scripts/vllm_serve.py analyze`                                                      | `logs/multi_tenant/run_*/session_report.json`                             |
| A6 | OCS cost model is the field-standard alpha-beta model: T(n) = α + β·n, with α_ocs = α_eps + T_reconfig | ✅ implemented + comparable | `scripts/compare_ocs_models.py` | `outputs/ocs_model_comparison.json` |

---

## A1 — Routing is a pure function of (input, weights)

**Claim.** In exact arithmetic, gate math depends only on input tokens and
model weights. The engine (MLX / HF / vLLM-metal), the physical topology
(pods × nodes × ranks, latencies, BW), and the node distribution of experts
(which rank/GPU owns which expert) must not change *which* expert a token
hits — only the *cost* of getting there.

**Verification, two levels.**

*Phase 1 (live, exact arithmetic).* The gate runs REAL inference itself
(vLLM-metal, greedy) — no pre-recorded traces — and applies a
one-variable-at-a-time matrix on the captured `token → expert` routing:

```bash
~/.venv-vllm-metal/bin/python scripts/verify_live_invariance.py
```

- vary topology (flat ↔ multi-tier fabrics): `token → expert` bit-identical
  (the same live recording replayed); only pairwise delay/cost moves.
- vary placement (linear ↔ shuffled expert→rank tables): `token → expert`
  identical; `token → rank` relabeled — same experts, different owning
  ranks; the rank-pair plan projection changes.
- vary prompt (same model): `token → expert` CHANGES — divergence metrics
  asserted (top-k overlap < 1, JS > 0, plan hit-rate < 1).
- vary model (same prompt): routing distributions diverge — asserted.
- folded payoff: affinity clustering raises intra-rank affinity and
  centrality-ordered rank locations cut cross-tier exposure.

Report: `logs/live_invariance_report.json`. The baseline capture refreshes
the canonical replay trace (`data/routing_traces/routing.json` + a
model-stamped copy).

*Phase 2 (hardware, up to the noise floor).* Same 4-bit weights, same prompt,
greedy decoding, MLX vs vLLM-metal:

```bash
python3 scripts/compare_backend_traces.py \
    --a logs/phase2/mlx/routing.json \
    --b logs/phase2/run_uniform_1t/traces/tenant-000.json
```

Contract: prompt ids identical; prefill JS ≤ 0.01; affinity corr ≥ 0.99;
plan hit-rate = 1.0; cell overlap bounded by the Metal noise floor.
**Result:** prefill overlap 0.933, JS 1.1e-4, corr 0.998, hit-rate 1.0. ✅

## A2 — Presets are model-specific

**Claim.** Affinity is a property of the *weights*. Same prompt, same
backend, different model ⇒ different affinity graph ⇒ OCS preset plans must
be re-derived per model (and per model version / quantization level).

**Verification (Phase 3).** Qwen1.5-MoE-A2.7B (60e) vs Qwen3.6-35B-A3B (256e):

```bash
python3 scripts/compare_model_affinity.py \
    --small logs/phase2/mlx/routing.json --large logs/phase3/large/routing.json
```

**Result (per-layer statistics only).** The layer-pooled metrics originally
reported here (top-5 expert share 0.101 → 0.043, layer-diversity JS
0.123 → 0.677, off-diagonal affinity strength 16× weaker) are **saturated by
construction** — expert ids are per-layer namespaces, so pooling averages
unrelated distributions (C2/C4 in `docs/research_assessment.md`). They were
discarded, not recalibrated. The per-layer profile separates the models far
more strongly: mean layer Gini 0.316 vs 0.786, mean peak load 3.1× vs 15.4×
uniform. ✅ Authoritative model separation on real workloads: Q2 category
decoding (62.5 % vs 93.75 %) and per-expert category-KL (1.9 % vs 23.7 % of
the log₂(12) bound).

**Open:** affinity drift across quantization levels / fine-tuning checkpoints
(declared in the design principles; not yet measured).

## A3 — Aggregate trust, per-cell noise

**Claim.** Quantized-GEMM near-ties flip marginal expert selections (≈0.84
cell-overlap noise floor), so preset decisions must use distribution-level
metrics (JS divergence, used-expert set, plan hit-rate) — never single cells.

**Evidence.** `scripts/vllm_serve.py affinity` on controlled prompt families
(identical / 1–2 slot edits): identical prompts + greedy → routing identity
1.0; similar prompts → same-token overlap 0.44–0.55 but JS ≈ 0.002–0.005,
affinity corr 0.93–0.98, plan hit-rate 1.0 — the expert *set* to preset is
stable while per-cell marginal selection drifts.

```bash
python3 scripts/vllm_serve.py affinity logs/multi_tenant/run_burst_4t
```

## A4 — Placement follows affinity (co-location)

**Claim.** Because A1 holds, placement (expert→rank and rank→physical
location) is a free, cost-side variable: tokens flow to wherever their
experts live. Recorded affinity can therefore decide *where* experts live to
make communication and computation cheaper, without changing routing.

**Verification (Phase 1 §3) + runtime.**

```bash
python3 scripts/verify_live_invariance.py   # payoff section, folded in
python3 -m src.launcher --config configs/ocs_affinity_placement.yaml
```

- Greedy co-activation clustering (`placement.strategy: affinity`) sets
  expert→rank: intra-rank affinity fraction 0.026 → 0.102 (more co-activated
  experts share a rank, skipping the network).
- Plan-centrality packing (`placement.rank_locations`, exported by the gate)
  sets rank→location: top-16 circuit plan cross-pod pairs 5 → 2, score-weighted
  cross-pod exposure 0.311 → 0.119.
- Default `placement.strategy: linear` reproduces every historical experiment
  bit-for-bit; routing never reads placement (only dispatch and the topology
  delay model do).

## A5 — Multi-tenant contention is measurable

**Claim.** With N co-batched tenants, decode steps carry ≥2 tenants and
tokens route to the same experts in the same forward — expert-collision and
TTFT/ITL growth are measurable and driven by placement/topology (cost side).

**Evidence.** `scripts/vllm_serve.py analyze` on `logs/multi_tenant/run_burst_4t`
(4 tenants): mean expert-collision ratio 0.747, TTFT/ITL growth vs the
sequential baseline in `session_report.json`.

## A6 — OCS cost model: fixed delay over an authentic EPS fabric

> ⚠️ **Superseded for research claims** (C5 in `docs/research_assessment.md`):
> `α_ocs = α_eps + T_reconfig` makes a hot circuit exactly as fast as
> electrical and a cold one strictly slower, so no experiment on this model
> can show an OCS benefit that is not a scheduling artifact. The
> research-grade model is tier promotion (`src/eval/ocs_eval.py` +
> `src/eval/cost_model.py`): a circuit removes oversubscription for the pair
> it serves. This entry describes the legacy data-plane model, retained for
> reproducing prior results and for wall-clock replay.

**Claim.** OCS should be simulated the way the field does it: every transfer
pays the same tier-aware EPS cost as the electrical baseline, plus a *fixed*
reconfiguration delay once per newly established circuit. Authenticity additionally requires the **circuit budget**: a real
switch has finite ports/wavelengths, so each rank holds at most
`max_circuits` simultaneous circuits; when the budget is exhausted the
oldest circuit is reassigned (FIFO port reassignment) and pays T_reconfig:

```
T_ocs(src, dst, bytes) = T_eps(src, dst, bytes) + T_reconfig × N_switches
```

Two canonical fixed-delay parameterizations ("alpha" and "beta" models):
alpha = fast switch class (SOA / ring-resonator, ns–µs reconfig,
T_reconfig ≈ 1 µs) with full fan-out (one circuit per destination, WSS);
beta = 3D-MEMS beam-steering (tens of µs mechanical motion + damping,
T_reconfig ≈ 50 µs) with a single outgoing port — the port-limited switch
serially re-points per destination, the authentic source of OCS
reconfiguration pressure on all-to-all traffic. The EPS fabric itself uses
field-cited numbers (NVLink/NVSwitch ~1 µs / 900 GB/s, InfiniBand NDR
~3 µs / 400 Gb/s = 50 GB/s, core fabric ~10 µs / 200 Gb/s = 25 GB/s —
bandwidths are GB/s end-to-end; the C7 unit error is fixed).

**Evidence.** `scripts/compare_ocs_models.py` runs EPS, alpha, and beta on
the same 2×2×1 fabric with real Qwen weights + captured routing (report:
`outputs/ocs_model_comparison.json`). Reference run: EPS comm 22.1 ms;
alpha +3.4 ms (reconfig 3 µs total, full fan-out); beta +2.3 ms (reconfig
3000 µs over 60 switches, 59 port reassignments — dispatch + gather each
re-point the single MEMS port per target per micro-batch).

---

## Cross-cutting rules

- Affinity is recorded only from the routing *function*: recompute, greedy
  (`--temp 0`), `--prefix-cache off`. The engine must never contaminate the
  trace with what it merely *executed*.
- Every backend validates its trace before saving (`RoutingTrace.validate()`);
  downstream stages refuse mismatched `num_experts` vs `world_size × experts_per_rank` instead of silently dropping experts.
- Each gate exits non-zero on failure and writes a JSON verdict report —
  the ledger above is reproducible by re-running the listed commands.
