# MoE + OCS Research Testbed

A **Mixture of Experts (MoE)** testbed — expert parallelism, all-to-all dispatch, async overlap pipelining, and **Optical Circuit Switching (OCS)** — driven entirely by **real Qwen MoE models**: real weights, real captured routing, real vLLM/MLX serving.

## Quick Start

```bash
pip install torch pyyaml
python3 -m src.launcher --config configs/qwen_ocs_lite.yaml   # real Qwen weights + OCS, fast
```

The pipeline stages:

- **Capture** — real routing traces from Qwen MoE models (vLLM primary, MLX secondary), validated `RoutingTrace` JSON
- **Verify** — invariance gates (topology / placement / hardware / model) before trusting recorded affinity
- **Replay + OCS** — real Qwen SwitchGLU experts on a 3-tier fabric with OCS circuit pool: pipeline, dual-batch overlap, preset, online affinity
- **Configure** — recorded affinity drives `expert → rank` and `rank → location` placement to cut cross-tier traffic

```bash
python3 -m src.launcher --config configs/qwen_ocs_lite.yaml      # OCS pipeline, 8 experts
python3 -m src.launcher --config configs/qwen_ocs_pipeline.yaml  # OCS pipeline, 32 experts
python3 -m src.launcher --config configs/qwen_ocs_dbo.yaml       # dual-batch overlap
python3 -m src.launcher --config configs/ocs_affinity_placement.yaml  # affinity-driven placement
```

## Core Concepts

### Expert Parallelism

```
expert_id → target_rank = expert_id // experts_per_rank
            local_expert  = expert_id %  experts_per_rank
Constraint: num_experts = world_size × experts_per_rank
```

The linear mapping above is the default `Placement.linear`; a `placement` config block can swap in affinity/shuffle/permutation tables (see *Affinity-Driven Placement*).

### Routing (Top-K Gating)

```
tokens [B×S, H] → Linear(hidden, num_experts) → softmax → topk → expert_ids [T, K] + gate_weights [T, K]
```

The gate is the **real Qwen gate** (dequantized weights from disk); in replay
mode it is replaced by the captured routing trace (`ReplayRouter`), so the
testbed replays exactly what the real model did.

### All-to-All Dispatch

`[token | local_expert_id | original_index]` float columns travel through `all_to_all_single` — zero extra communication rounds. Async overlap: scatter/gather fire with `async_op=True`, wait when needed (NCCL CUDA-stream pattern).

### Network Topology (3-tier) — the authentic EPS fabric

Field-cited numbers (NVIDIA DGX/NVSwitch, Mellanox NDR, core fabrics):

| Tier       | Link            | Latency | Bandwidth |
| ---------- | --------------- | ------- | --------- |
| INTRA_NODE | NVLink/NVSwitch | ~1 µs   | 900 GB/s  |
| INTRA_POD  | InfiniBand NDR  | ~3 µs   | 400 Gb/s  |
| CROSS_POD  | core fabric     | ~10 µs  | 200 Gb/s  |

Delay = `latency + tensor_bytes / (bandwidth_gbps × 1000)`. Configurable in YAML (`configs/qwen_eps_baseline.yaml` enables the 2×2×1 fabric used for EPS/OCS comparison).

### OCS Circuit Pool

The **legacy** finite circuit-cache model (optional, `ocs.cost_model: lru`):

|                            | EPS                          | OCS (LRU pool)                                   |
| -------------------------- | ---------------------------- | ------------------------------------------------ |
| **Connection**       | Always-on, per-packet routed | Finite pool of reconfigurable circuits           |
| **Setup cost**       | None                         | `reconfig_time_us` when cold (mirror steering) |
| **Once established** | N/A                          | `circuit_latency_us` + bytes/BW                |
| **Capacity**         | Unlimited                    | `max_circuits` pool, LRU eviction              |

### EPS vs OCS Cost Models (authentic comparison)

The LRU pool above is *not* how most OCS studies model the switch — it makes
OCS hard to compare against EPS. The field-standard formulation used in this
repo (default for the comparison configs):

```
T_ocs(src, dst, bytes) = T_eps(src, dst, bytes) + T_reconfig × N_switches
```

- **Authentic EPS**: the same 3-tier fabric cost every model pays, with
  field-cited numbers (see `src/comm/topology.py`): NVLink/NVSwitch ~1 µs /
  900 GB/s, InfiniBand NDR ~3 µs / 400 Gb/s, core fabric ~10 µs / 200 Gb/s.
- **OCS fixed-delay** (`ocs.cost_model: fixed_delay`): every transfer pays
  the identical tier-aware EPS cost, plus a *fixed* reconfiguration delay
  once per newly established circuit. No LRU cache — instead a **per-rank
  circuit budget** (`max_circuits` = ports/wavelengths); when the budget is
  exhausted the oldest circuit is reassigned (FIFO port reassignment) and
  pays `T_reconfig`. This is what makes OCS honestly expensive on
  all-to-all traffic: a port-limited switch must serially re-point per
  destination.
- **alpha / beta models** — the two canonical fixed-delay parameterizations
  (reconfig times follow the switch-technology literature: SOA/ring
  ns–µs cf. [Sirius](https://dlnext.acm.org/doi/epdf/10.1145/3387514.3406221);
  3D-MEMS tens of µs mechanical motion plus damping, deployed
  Apollo/Palomar-class systems settle in the ms range):

  | model | switch class | `T_reconfig` | circuit budget |
  | ----- | ------------ | ------------ | -------------- |
  | alpha | SOA / ring-resonator (fast, WSS fan-out) | 1 µs | world_size−1 |
  | beta  | 3D-MEMS beam-steering (port-limited) | 50 µs | 1 |

```bash
python3 -m src.launcher --config configs/qwen_eps_baseline.yaml   # authentic EPS
python3 -m src.launcher --config configs/ocs_alpha_model.yaml     # OCS alpha
python3 -m src.launcher --config configs/ocs_beta_model.yaml      # OCS beta
python3 scripts/compare_ocs_models.py    # runs all three on the same fabric + table
```

The reconfig delay is *actually paid* (slept) on the cold path, so the
comparison is honest end-to-end; run traces record reconfig totals,
reuse/establish counts, and port reassignments in the trace metadata.

### OCS Preset: Captured Affinity → Inference Pre-Configuration

Core question: can **captured routing patterns** pre-configure OCS circuits before inference begins, for zero runtime reconfig?

| Mode             | Reconfig                        | Circuits established               | Use case                                  |
| ---------------- | ------------------------------- | ---------------------------------- | ----------------------------------------- |
| `ocs_pipeline` | Per-microbatch, inline          | Before each scatter                | Runtime adaptability                      |
| `ocs_dbo`      | Hidden behind compute           | Batch K+1 during batch K           | Mask reconfig latency                     |
| `ocs_preset`   | **None during inference** | **Before first token**       | Captured-affinity → inference pre-config |
| `ocs_online`   | Adaptive, amortized             | Per-step from accumulated affinity | Inference-self-learning                   |

```bash
bash scripts/run_preset_pipeline.sh data/routing_traces/routing.json   # trace → plan → EPS/OCS/preset → compare
```

Preset sources: `trace` (recorded affinity → plan) or `plan` (pre-computed plan JSON). Key files: [src/ocs/preconfig.py](src/ocs/preconfig.py), [src/eval/affinity_consistency.py](src/eval/affinity_consistency.py).

### OCS Online Affinity

No separate capture phase — track expert co-activation *during* inference and periodically recompute the circuit plan, with exponential decay (`decay_factor` 0.99/step) for responsiveness to pattern shifts.

```bash
python3 -m src.launcher --config configs/ocs_affinity_placement.yaml
```

### Affinity-Driven Placement (record affinity → adjust topology)

One story, two verified halves. `scripts/verify_ocs_invariance.py` replays a real Qwen routing trace and gates the whole chain in one run:

1. **Affinity is independent of node distribution** — the co-activation matrix is built from expert ids only, so it is bit-identical no matter which rank/node owns each expert (and identical under any 3-tier topology config). Placement relabels ranks (cost), never which expert a token hits (routing).
2. **So the recorded affinity can safely configure the topology** — greedy co-activation clustering sets `expert → rank` (intra-rank affinity fraction 0.026 → 0.115 in the reference vLLM trace), and plan-centrality ordering sets `rank → physical location` (cross-pod exposure of the top-16 circuit plan drops 3 → 0 pairs). The derived `rank → location` table is exported in the report for `placement.rank_locations`.

```bash
python3 scripts/verify_ocs_invariance.py                           # one gate: invariance + payoff
python3 -m src.launcher --config configs/ocs_affinity_placement.yaml
```

`placement.strategy` (`linear` = historical `e//k` default, `shuffle`, `affinity`, `permutation`) builds the expert→rank table; `placement.rank_locations` pins ranks to physical spots, consumed by the topology delay model. With no `placement` block, every existing experiment is bit-identical to before.

## Real Qwen MoE Routing Capture

Capture real routing traces from Qwen MoE models into the canonical `RoutingTrace` JSON (same schema across all backends). **vLLM is the primary capture backend** — the canonical traces in `data/routing_traces/` are vLLM-captured (`meta.backend: vllm`); use `--temp 0` for deterministic greedy traces:

```bash
# vLLM backend (CUDA box)
python scripts/run_vllm.py run --model Qwen/Qwen3.6-35B-A3B \
    --prompt "Explain MoE routing." --max-tokens 128 --temp 0

# vLLM + Metal backend (Apple Silicon GPU; runs MLX-format models)
# 1. Install once: curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
# 2. Run — the script pins VLLM_HOST_IP/MASTER_ADDR to 127.0.0.1 itself
#    (the macOS hostname may resolve to an unreachable LAN IP and crash
#    PyTorch's TCPStore otherwise):
source ~/.venv-vllm-metal/bin/activate
python scripts/run_vllm.py run \
    --model ./models/Qwen3.6-35B-A3B-4bit --max-tokens 256 --temp 0

# MLX backend (Apple Silicon, native) — secondary capture path
.venv/bin/python moe_run.py --model models/Qwen3.6-35B-A3B-4bit --max-tokens 128 --temp 0

# Multi-tenant serving capture (vLLM + vllm-metal, see dedicated section)
PY=~/.venv-vllm-metal/bin/python
$PY scripts/vllm_serve.py run --tenants 4 --schedule burst --greedy --max-tokens 32

# Steering / ablation (all backends)
python scripts/run_vllm.py intervene --force-expert 0 100 --max-tokens 32
python scripts/run_vllm.py ablate --ablate-expert 3 12 --max-tokens 32
```

**Capture guarantees:** captured experts are the *executed* experts (same `inds` fed to the experts); every backend validates the trace before saving (`RoutingTrace.validate()`); downstream preconfig refuses mismatched `num_experts` vs `world_size × experts_per_rank` instead of silently dropping experts.

```bash
# Export Qwen expert weights for OCS simulation (one-time)
python3 scripts/export_qwen_experts.py \
    --model models/Qwen3.6-35B-A3B-4bit \
    --output exported_qwen_weights --max-layers 1

# OCS experiments with real weights
python3 -m src.launcher --config configs/qwen_ocs_lite.yaml      # 8 experts, fast
python3 -m src.launcher --config configs/qwen_ocs_pipeline.yaml  # 32 experts, full
python3 -m src.launcher --config configs/qwen_ocs_dbo.yaml       # dual-batch overlap
```

## Routing Independence: Main Assumption & Verification

**The assumption (informal, now verified):** routing is a pure function of
(input tokens, weights) — hardware and physical topology do not enter the
gate math. So the affinity graph is a model×input property: wherever the
experts are physically placed, tokens flow to them anyway, and
topology/placement can be treated as independent, cost-side variables
(decided *after* affinity). Corollary: presets are model-specific.

Verified in phases; each is a script with a JSON verdict report:

```bash
# Phase 1 — framework level: one gate for topology invariance, placement
# invariance, and the affinity→placement payoff (see Affinity-Driven Placement).
python3 scripts/verify_ocs_invariance.py \
    --trace data/routing_traces/routing.json --world-size 32 --experts-per-rank 8

# Phase 2 — hardware level: same 4-bit weights, same prompt, greedy decoding.
# Capture with both engines first, then compare MLX vs vLLM-metal:
python scripts/compare_backend_traces.py \
    --a logs/phase2/mlx/routing.json \
    --b logs/phase2/run_uniform_1t/traces/tenant-000.json

# Phase 3 — model dependence: same prompt, same backend, different models.
python scripts/compare_model_affinity.py \
    --small logs/phase2/mlx/routing.json --large logs/phase3/large/routing.json
```

**Results (Qwen1.5-MoE-A2.7B 60e vs Qwen3.6-35B-A3B 256e):**

| Check   | Holding fixed           | Varied                                                        | Result                                                                | Verdict                                     |
| ------- | ----------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------- |
| Phase 1 | weights, trace          | topology config (1×1×32 → 4×2×4) + expert/rank placement | affinity + plan bit-identical; only dispatch cost moves               | topology/placement-independent ✓           |
| Phase 2 | weights, prompt, greedy | engine (MLX vs vLLM-metal)                                    | prefill overlap 0.933, JS 1.1e-4, corr 0.998, plan hit-rate 1.0       | hardware-independent (up to noise floor) ✓ |
| Phase 3 | prompt, backend         | model (60e vs 256e)                                           | top-5 share 0.101→0.043, layer-JS 0.123→0.677, affinity 16× weaker | model-specific ✓                           |

Conclusion: **affinity = f(input, weights)** — input-driven divergence is
meaningful, hardware/topology/placement is not, and OCS presets must be
derived per model. Reports: `logs/ocs_invariance_report.json`,
`logs/phase2/invariance_report.json`, `logs/phase3/model_diversity_report.json`.
Full per-assumption ledger (claim → gate → report → open items):
[docs/assumptions.md](docs/assumptions.md).

## Multi-Tenant Serving (vLLM + vllm-metal)

Runs the real vLLM V1 engine on Apple Silicon via
[vllm-metal](https://github.com/vllm-project/vllm-metal) as a serving
dispatcher: tenants arrive on a traffic schedule, share the engine, and every
engine step is recorded with its tenant composition. Prefix caching stays at
the engine default (**on**, as in real deployments); `--prefix-cache off` is
the calibration variant that forces every tenant to log full prefill routing.

```bash
PY=~/.venv-vllm-metal/bin/python

# 1. Concurrent serving: 6 tenants, burst arrivals (real case: prefix cache on)
$PY scripts/vllm_serve.py run --tenants 6 --schedule burst --rate 2.0 --max-tokens 64

# 2. Zero-contention baseline: identical workload, one tenant at a time
$PY scripts/vllm_serve.py run --tenants 6 --schedule burst --rate 2.0 --mode sequential

# 3. Analyze: TTFT/ITL, contention windows, expert-collision rate, slowdown vs baseline
$PY scripts/vllm_serve.py analyze logs/multi_tenant/run_burst_6t \
    --baseline logs/multi_tenant/run_burst_6t_sequential --plot

# 4. Cross-tenant routing affinity + edit-distance curve (prompt families)
$PY scripts/vllm_serve.py run --tenants 4 --schedule burst --family similar \
    --greedy --max-tokens 32
$PY scripts/vllm_serve.py affinity logs/multi_tenant/run_burst_4t --plot
```

Sessions land in `logs/multi_tenant/run_<schedule>_<N>t[_sequential]/`
(`session.json`, `traces/<tenant>.json` in canonical `RoutingTrace` format,
`session_report.json`, `timeline.png`, `affinity_report.json`).

**Calibration findings (Qwen3.6-35B-A3B, 256 experts, top-8):** identical
prompts + greedy → co-batched tenants have identical routing (1.0); the
marginal expert flips on near-ties from Metal quantized-GEMM noise
(noise floor ≈ 0.84 cell-overlap). Similar prompts (1–2 slot edits):
same-token routing overlap 0.44–0.55, aggregate distribution nearly
unchanged (JS ≈ 0.002–0.005, affinity corr 0.93–0.98), used-expert
universe fully shared (plan hit-rate 1.0) — the expert set to preset is
stable; only the per-cell marginal selection drifts.

## Additional MoE Models (Hy3 + Whittle)

```bash
# Download (optional --mirror uses https://hf-mirror.com via HF_ENDPOINT)
python scripts/download_external_models.py --target hy3 --mirror
python scripts/download_external_models.py --target whittle --mirror

# Whittle only: convert BF16 → MLX 4-bit (one-time)
mlx_lm.convert --hf-path models/Qwen3.8-Whittle-MoE-27B-A17.8B \
    --mlx-path models/Qwen3.8-Whittle-MoE-27B-A17.8B-4bit -q --q-bits 4

# Capture routing traces — same moe_run.py, only --model changes:
python moe_run.py --model ./models/Hy3-oQ2 \
    --max-tokens 32 --temp 0 --log-dir logs/phase4/hy3
python moe_run.py --model ./models/Qwen3.8-Whittle-MoE-27B-A17.8B-4bit \
    --max-tokens 32 --temp 0 --log-dir logs/phase4/whittle

# Compare against the Qwen baselines:
python scripts/compare_model_affinity.py \
    --small logs/phase2/mlx/routing.json --large logs/phase4/hy3/routing.json
```

## Research Design Principles

Agreed design principles for the OCS/topology work:

- [X] **Formalize the routing-independence assumption.** Routing is a pure
  function of (input tokens, weights) — independent of hardware and
  topology. Affinity must be recorded only from the routing *function*
  (recompute, greedy, `--prefix-cache off`). Verified Phase 1+2.
- [X] **Keep topology/placement as independent variables.** Affinity is a
  model×input property (decides *what* to co-locate); topology and
  expert/rank placement live only on the cost side. Verified by
  `scripts/verify_ocs_invariance.py` (§1 topology, §2 placement).
- [ ] **Trust affinity at aggregate level, not per-cell.** The quantized-GEMM
  noise floor (≈0.84 cell-overlap near-tie flips) makes per-cell routing
  noisy; preset decisions must use distribution-level metrics (JS divergence,
  used-expert set / plan hit-rate).
- [X] **Routing-stability check.** Affinity holds only while weights are
  fixed — quantization changes or fine-tuning drift the affinity graph.
  Model dependence verified (Phase 3); quantization-level drift still to
  measure.
- [X] **Co-location rule.** Place frequently co-demanded experts together per
  the affinity graph — implemented as `placement.strategy: affinity`
  (greedy co-activation clustering) plus `placement.rank_locations`
  (plan-centrality packing); see *Affinity-Driven Placement*.

## Project Structure

```
src/
├── model/        Qwen expert loader + gate, captured-routing replay router
├── comm/         all-to-all, transport, topology, timeline export
├── ocs/          Circuit pool (LRU), affinity placement, preconfig, online controller
├── runtime/      Per-rank worker, scheduler, process groups, placement tables
├── data/         Routing schema, MLX/vLLM capture, interventions, model utils
├── serving/      Multi-tenant vLLM serving capture (engine, workload, affinity, analyze)
├── eval/         Affinity consistency metrics
└── utils/        Timer, logging, seed
configs/          Qwen replay/OCS configs + affinity-driven placement
scripts/          Viz, comparison, export, validation, presets, serving CLI,
                  verification gates (Phase 1–3)
docs/             Research discussions, assumption ledger, architecture alignment
data/             Reference routing traces + fine-tuning dataset (moe_train)
logs/             Phase 1–4 verification reports + captures (phase2/3/4)
models/           MLX model weights (capture backends)
adapters/         LoRA adapters (qwen3.6-moe-lora)
moe_mlx_learning.history.bundle   Pre-merge git history archive (undo path)
```

## Configuration Reference

| Parameter                            | Description                                                                            | Default                              |
| ------------------------------------ | -------------------------------------------------------------------------------------- | ------------------------------------ |
| `world_size`                       | Number of ranks                                                                        | 4                                    |
| `model.num_experts`                | Total experts (= world_size × experts_per_rank)                                       | 32                                   |
| `model.experts_per_rank`           | Experts per GPU                                                                        | 8                                    |
| `model.hidden_dim`                 | Qwen dequantized hidden dim                                                            | 2048                                 |
| `model.top_k`                      | Top-K gating                                                                           | 2                                    |
| `routing.strategy`                 | `replay` (captured routing)                                                          | `fixed`                            |
| `placement.strategy`               | Expert→rank table:`linear`, `shuffle`, `affinity`, `permutation`              | `linear`                           |
| `placement.trace_path`             | Routing trace for the`affinity` strategy                                             | `data/routing_traces/routing.json` |
| `placement.rank_locations`         | Explicit rank→(pod, node, local) table for the topology model                         | none (linear)                        |
| `runtime.mode`                     | `serial`, `overlap`, `ocs_pipeline`, `ocs_dbo`, `ocs_preset`, `ocs_online` | `serial`                           |
| `ocs.enabled`                      | Enable OCS circuit pool                                                                | false                                |
| `ocs.cost_model`                   | `fixed_delay` (EPS + fixed reconfig per switch, field-standard) or `lru` (legacy cache) | `lru`                              |
| `ocs.max_circuits`                 | Max simultaneous optical circuits (LRU model only)                                     | 32                                   |
| `ocs.reconfig_time_us`             | Fixed circuit reconfig time — alpha 1 µs (fast switch), beta 50 µs (MEMS)              | 50.0                                 |
| `ocs.circuit_latency_us`           | Hot-path latency                                                                       | 1.0                                  |
| `ocs.circuit_bandwidth_gbps`       | Circuit bandwidth                                                                      | 200.0                                |
| `ocs.preset.strategy`              | Plan computation:`coactivation`                                                      | `coactivation`                     |
| `ocs.online.update_interval_steps` | Recompute plan every N steps                                                           | 1                                    |
| `ocs.online.decay_factor`          | Exponential decay per step                                                             | 0.99                                 |

## Viewing Results

```bash
open outputs/traces/trace_viewer.html    # interactive trace ("EP Layout" = expert mapping)
open outputs/traces/ocs_view.html        # OCS circuit analysis
# chrome://tracing or https://ui.perfetto.dev → load outputs/traces/merged_trace.json
```

**Event colors:** Green=Route, Blue=Scatter, Orange=Compute, Purple=Gather, Pink=Combine, Cyan=ScatterWait, Red=AllToAll.

## Architecture Alignment

The 4-phase pipeline (route → dispatch → compute → combine) is aligned with Megatron-LM, Tutel, Megablocks, DeepEP, and DeepSpeed-MoE. See [docs/alignment.md](docs/alignment.md).

## Key Design Decisions

- **Metadata packing:** `local_expert_id` + `original_index` ride as float columns through `all_to_all_single` — zero extra comm rounds
- **Async overlap:** scatter/gather with `async_op=True`, wait when needed — NCCL CUDA-stream pattern
- **OCS gating:** all OCS paths behind `ocs.enabled: true` — zero overhead when disabled
- **Circuit pool:** `OrderedDict` LRU eviction in O(1), mirrors real OCS switch behavior
- **Token positions are absolute** (0-indexed full sequence), not per-step — cross-layer analysis works trivially
- **`RoutingTrace` is the single source of truth** — every backend (HF/MLX/vLLM single- and multi-tenant) produces it, all analysis tools consume it
- **Placement is a cost-side variable** — routing never reads placement; only dispatch and the topology delay model do, so affinity-driven expert/rank and rank/location tables can be swapped without touching routing (defaults reproduce historical behavior bit-for-bit)

## License

MIT
