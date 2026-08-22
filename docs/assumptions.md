# Assumption Ledger

Every assumption this testbed relies on, what it claims, how it is verified,
where the verdict lives, and what is still open. Each entry is tied to one
verification script (a "phase") with a JSON report; `README.md` has the
condensed version.

| # | Assumption | Status | Gate script | Report |
| - | ---------- | ------ | ----------- | ------ |
| A1 | Routing is a pure function of (input, weights) — independent of topology, node distribution, and engine | ✅ verified (framework + hardware) | `scripts/verify_ocs_invariance.py` (§1–§2), `scripts/compare_backend_traces.py` | `logs/ocs_invariance_report.json`, `logs/phase2/invariance_report.json` |
| A2 | Affinity is model×input specific — OCS presets must be re-derived per model | ✅ verified (model level); ⚠️ quantization-level drift open | `scripts/compare_model_affinity.py` | `logs/phase3/model_diversity_report.json` |
| A3 | Per-cell routing is noisy (quantized-GEMM near-ties); trust distribution-level metrics only | ⚠️ empirically established, per-cell decisions avoided | `scripts/vllm_serve.py affinity` (calibration families) | `logs/multi_tenant/run_*/affinity_report.json` |
| A4 | Placement is a cost-side variable — recorded affinity can safely configure expert→rank and rank→location | ✅ verified (framework) | `scripts/verify_ocs_invariance.py` (§3) | `logs/ocs_invariance_report.json` (affinity_adjustment) |
| A5 | Multi-tenant co-batching creates measurable expert contention | ✅ measured | `scripts/vllm_serve.py analyze` | `logs/multi_tenant/run_*/session_report.json` |

---

## A1 — Routing is a pure function of (input, weights)

**Claim.** In exact arithmetic, gate math depends only on input tokens and
model weights. The engine (MLX / HF / vLLM-metal), the physical topology
(pods × nodes × ranks, latencies, BW), and the node distribution of experts
(which rank/GPU owns which expert) must not change *which* expert a token
hits — only the *cost* of getting there.

**Verification, two levels.**

*Phase 1 (framework, exact arithmetic).* Replays one real Qwen trace
(`data/routing_traces/routing.json`, 256 experts, top-8):

```bash
python3 scripts/verify_ocs_invariance.py \
    --trace data/routing_traces/routing.json --world-size 32 --experts-per-rank 8
```

- §1 topology invariance: affinity matrix + circuit plan bit-identical under
  4 three-tier fabrics (1×1×32 → 4×2×4, slow fabric, delay multiplier);
  only per-circuit dispatch delay moves.
- §2 placement invariance: `Placement.linear` == historical `e//k`/`e%k`
  bit-for-bit; swap/shuffle relabel ranks without touching routing or the
  affinity matrix; only the rank-pair plan projection changes.

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

**Result:** top-5 expert share 0.101 → 0.043, layer-diversity JS 0.123 →
0.677, off-diagonal affinity strength 16× weaker. ✅

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
python3 scripts/verify_ocs_invariance.py   # §3 payoff section
python3 -m src.launcher --config configs/ocs_affinity_placement.yaml
```

- Greedy co-activation clustering (`placement.strategy: affinity`) sets
  expert→rank: intra-rank affinity fraction 0.026 → 0.115 (more co-activated
  experts share a rank, skipping the network).
- Plan-centrality packing (`placement.rank_locations`, exported by the gate)
  sets rank→location: top-16 circuit plan cross-pod pairs 3 → 0, score-weighted
  cross-pod exposure 0.187 → 0.000.
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

---

## Cross-cutting rules

- Affinity is recorded only from the routing *function*: recompute, greedy
  (`--temp 0`), `--prefix-cache off`. The engine must never contaminate the
  trace with what it merely *executed*.
- Every backend validates its trace before saving (`RoutingTrace.validate()`);
  downstream stages refuse mismatched `num_experts` vs `world_size ×
  experts_per_rank` instead of silently dropping experts.
- Each gate exits non-zero on failure and writes a JSON verdict report —
  the ledger above is reproducible by re-running the listed commands.
