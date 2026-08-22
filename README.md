# MoE + OCS Research Testbed

An **MoE communication testbed driven entirely by real Qwen models** — real weights, real captured routing, real vLLM/MLX serving — for studying expert dispatch over an **authentic 3-tier electrical fabric** (EPS) and **optical circuit switching** (OCS) with a field-standard cost model. (The earlier synthetic toy-expert testbed was removed; it remains in git history.)

```
capture ──▶ verify ──▶ replay ──▶ compare (EPS vs OCS) ──▶ configure (affinity-driven placement)
  │            │           │
vLLM/MLX   invariance   real Qwen experts on the 3-tier fabric
```

## Quick Start

```bash
# 1. One-time: export Qwen expert weights
python3 scripts/export_qwen_experts.py \
    --model models/Qwen3.6-35B-A3B-4bit --output exported_qwen_weights --max-layers 1

# 2. OCS pipeline on real Qwen weights
python3 -m src.launcher --config configs/qwen_ocs_lite.yaml      # 8 experts, fast
python3 -m src.launcher --config configs/qwen_ocs_pipeline.yaml  # 32 experts

# 3. EPS vs OCS comparison (fixed-delay alpha/beta models)
python3 scripts/compare_ocs_models.py
```

## Capture (vLLM primary, MLX secondary)

Canonical `RoutingTrace` JSON, same schema across backends; every backend validates before saving. `--temp 0` = deterministic greedy. The canonical traces in `data/routing_traces/` are vLLM-captured.

```bash
# vLLM (CUDA)
python scripts/run_vllm.py run --model Qwen/Qwen3.6-35B-A3B \
    --prompt "Explain MoE routing." --max-tokens 128 --temp 0

# vLLM + Metal (Apple Silicon; the script pins VLLM_HOST_IP/MASTER_ADDR
# to loopback itself, fixing the macOS TCPStore crash)
source ~/.venv-vllm-metal/bin/activate
python scripts/run_vllm.py run \
    --model ./models/Qwen3.6-35B-A3B-4bit --max-tokens 256 --temp 0

# MLX (secondary)
.venv/bin/python moe_run.py --model models/Qwen3.6-35B-A3B-4bit --max-tokens 128 --temp 0

# Multi-tenant vLLM serving capture
PY=~/.venv-vllm-metal/bin/python
$PY scripts/vllm_serve.py run --tenants 4 --schedule burst --greedy --max-tokens 32
$PY scripts/vllm_serve.py analyze logs/multi_tenant/run_burst_4t --plot      # contention/TTFT
$PY scripts/vllm_serve.py affinity logs/multi_tenant/run_burst_4t --plot     # prompt families

# Steering / ablation
python scripts/run_vllm.py intervene --force-expert 0 100 --max-tokens 32
```

## Verification — routing independence

**Assumption:** routing = f(input, weights). Hardware, topology, and expert
placement never change *which* expert a token hits — only the cost. Each
phase is a script with a JSON verdict report.

```bash
python3 scripts/verify_ocs_invariance.py        # Phase 1: topology + placement invariance,
                                                #   + affinity→placement payoff (one gate)
python3 scripts/compare_backend_traces.py \
    --a logs/phase2/mlx/routing.json \
    --b logs/phase2/run_uniform_1t/traces/tenant-000.json   # Phase 2: MLX vs vLLM-metal
python3 scripts/compare_model_affinity.py \
    --small logs/phase2/mlx/routing.json \
    --large logs/phase3/large/routing.json                  # Phase 3: 60e vs 256e models
```

| Phase | Varied | Result | Verdict |
| ----- | ------ | ------ | ------- |
| 1 | topology config + expert/rank placement | affinity + plan bit-identical; only cost moves | topology/placement-independent ✓ |
| 2 | engine (MLX vs vLLM-metal) | prefill overlap 0.933, JS 1.1e-4, corr 0.998, hit-rate 1.0 | hardware-independent (up to noise floor) ✓ |
| 3 | model (60e vs 256e) | top-5 share 0.101→0.043, layer-JS 0.123→0.677 | presets are model-specific ✓ |

Conclusion: **affinity = f(input, weights)** — so recorded affinity can
safely drive OCS configuration. Full assumption ledger (claim → gate →
report → open items): [docs/assumptions.md](docs/assumptions.md).

## Cost models — authentic EPS vs field-standard OCS

**EPS** is the 3-tier fabric with field-cited numbers:

| Tier | Link | Latency | Bandwidth |
| ---- | ---- | ------- | --------- |
| INTRA_NODE | NVLink/NVSwitch | ~1 µs | 900 GB/s |
| INTRA_POD | InfiniBand NDR | ~3 µs | 400 Gb/s |
| CROSS_POD | core fabric | ~10 µs | 200 Gb/s |

**OCS** uses the field-standard fixed-delay model (no LRU cache):

```
T_ocs = T_eps + T_reconfig × N_switches
```

Each transfer pays the same tier-aware EPS cost plus a **fixed reconfig
delay** per newly established circuit, under a **per-rank circuit budget**
(`max_circuits` = ports/wavelengths, FIFO port reassignment when exhausted).
Reconfig is really *paid* (slept) on the cold path. Two canonical
parameterizations (reconfig times per switch-technology literature —
SOA/ring ns–µs, [Sirius](https://dlnext.acm.org/doi/epdf/10.1145/3387514.3406221);
3D-MEMS tens of µs + damping):

| model | switch | `T_reconfig` | budget |
| ----- | ------ | ------------ | ------ |
| alpha | SOA/ring (fast, WSS fan-out) | 1 µs | world_size−1 |
| beta | 3D-MEMS (port-limited) | 50 µs | 1 |

```bash
python3 -m src.launcher --config configs/qwen_eps_baseline.yaml   # EPS
python3 -m src.launcher --config configs/ocs_alpha_model.yaml     # OCS alpha
python3 -m src.launcher --config configs/ocs_beta_model.yaml      # OCS beta
python3 scripts/compare_ocs_models.py    # all three on the same fabric + table
```

Reference run (real Qwen, 2×2×1 fabric): EPS 22.1 ms comm; alpha +3.4 ms
(3 µs reconfig total); beta +2.3 ms (**3000 µs reconfig**, 60 switches, 59
port reassignments — the single MEMS port re-points per target for both
dispatch and gather). `ocs.cost_model: lru` remains available for backward
compatibility only.

## OCS scheduling modes

| Mode | Reconfig | Use case |
| ---- | -------- | -------- |
| `ocs_pipeline` | inline, before each scatter | runtime adaptability |
| `ocs_dbo` | hidden behind previous batch's compute | mask reconfig latency |
| `ocs_preset` | **none during inference** (plan pre-loaded) | affinity → pre-config |
| `ocs_online` | adaptive, from live co-activation (decay 0.99) | inference self-learning |

```bash
bash scripts/run_preset_pipeline.sh data/routing_traces/routing.json   # trace → plan → EPS/OCS/preset → compare
```

## Affinity-driven placement

Because affinity is placement-independent (Phase 1), the recorded affinity
can reconfigure *where experts live* without touching routing:

1. `placement.strategy: affinity` — greedy co-activation clustering sets
   `expert → rank` (intra-rank affinity 0.026 → 0.115)
2. `placement.rank_locations` — plan-centrality packing sets
   `rank → physical location` (top-16 plan cross-pod pairs 3 → 0)

```bash
python3 -m src.launcher --config configs/ocs_affinity_placement.yaml   # 256 experts, 32 ranks
```

`placement.strategy`: `linear` (default, historical `e//k`), `shuffle`,
`affinity`, `permutation`. Routing never reads placement.

## Additional models (Hy3 / Whittle)

```bash
python3 scripts/download_external_models.py --target hy3 --mirror     # or whittle
python3 moe_run.py --model ./models/Hy3-oQ2 --max-tokens 32 --temp 0 --log-dir logs/phase4/hy3
python3 scripts/compare_model_affinity.py --small logs/phase2/mlx/routing.json \
    --large logs/phase4/hy3/routing.json
```

## Project structure

```
src/
├── model/      Qwen expert loader + gate, captured-routing replay router
├── comm/       all-to-all, transport, 3-tier topology, timeline export
├── ocs/        fixed-delay + LRU cost models, affinity placement, preconfig, online controller
├── runtime/    worker, schedulers, placement tables, process groups
├── data/       routing schema, MLX/vLLM capture, interventions
├── serving/    multi-tenant vLLM serving capture (engine, workload, affinity, analyze)
├── eval/       affinity consistency metrics
└── utils/      timer, logging, seed
configs/        qwen replay/OCS configs, EPS baseline, alpha/beta, affinity placement
scripts/        capture, verification gates (Phase 1–3), EPS-vs-OCS comparison, presets
docs/           assumption ledger, research discussions, architecture alignment
data/           reference routing traces (vLLM-captured) + fine-tuning dataset
logs/           Phase 1–4 verification reports + captures
models/         MLX model weights
adapters/       LoRA adapters
```

## Key design decisions

- **`RoutingTrace` is the single source of truth** — every backend produces it, every tool consumes it; absolute token positions throughout
- **Placement is a cost-side variable** — routing never reads placement; only dispatch and the topology delay model do
- **OCS = EPS + fixed reconfig** — field-standard cost model, circuit-budget constrained, honestly paid on the cold path
- **Metadata packing** — `local_expert_id` + `original_index` ride as float columns through `all_to_all_single` (zero extra comm rounds); async scatter/gather in NCCL CUDA-stream pattern

## License

MIT
