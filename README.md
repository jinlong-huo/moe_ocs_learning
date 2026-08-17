# MoE + OCS Research Testbed

A **Mixture of Experts (MoE)** testbed — expert parallelism, all-to-all dispatch, async overlap pipelining, and **Optical Circuit Switching (OCS)** — verified with real Qwen MoE routing before GPU-cluster deployment.

## Quick Start

```bash
pip install torch pyyaml
python3 -m src.launcher --config configs/synthetic_moe.yaml
```

Three learning paths:

- **Path 1 — MoE Basics:** expert parallelism (`expert_id → (rank, local_expert)`), top-K gate routing, the 4-phase pipeline route → scatter → compute → gather → combine.
- **Path 2 — Overlap & Pipelining:** micro-batch pipelining, async fire-and-forget all-to-all, double buffering.
- **Path 3 — OCS:** circuit pool (LRU `OrderedDict`), reconfig cost, OCS pipeline, dual-batch overlap (DBO), preset mode, online affinity mode.

```bash
python3 -m src.launcher --config configs/ocs_demo.yaml        # OCS pipeline
python3 -m src.launcher --config configs/ocs_dbo_demo.yaml    # dual-batch overlap
python scripts/compare_ocs.py                                  # EPS vs OCS
```

## Core Concepts

### Expert Parallelism

```
expert_id → target_rank = expert_id // experts_per_rank
            local_expert  = expert_id %  experts_per_rank
Constraint: num_experts = world_size × experts_per_rank
```

### Routing (Top-K Gating)

```
tokens [B×S, H] → Linear(hidden, num_experts) → softmax → topk → expert_ids [T, K] + gate_weights [T, K]
```

Router strategies: `fixed`, `top1`, `top2`, `uniform_random`.

### All-to-All Dispatch

`[token | local_expert_id | original_index]` float columns travel through `all_to_all_single` — zero extra communication rounds. Async overlap: scatter/gather fire with `async_op=True`, wait when needed (NCCL CUDA-stream pattern).

### Network Topology (3-tier)

| Tier | Link | Latency | Bandwidth |
| ---- | ---- | ------- | --------- |
| INTRA_NODE | NVLink/NVSwitch | ~2 µs | 600 GB/s |
| INTRA_POD | InfiniBand | ~5 µs | 200 GB/s |
| CROSS_POD | IB fabric | ~15 µs | 100 GB/s |

Delay = `latency + tensor_bytes / (bandwidth_gbps × 1000)`. Configurable in YAML (`configs/realistic_16gpu.yaml`).

### OCS Circuit Pool

| | EPS | OCS |
| - | --- | --- |
| **Connection** | Always-on, per-packet routed | Finite pool of reconfigurable circuits |
| **Setup cost** | None | `reconfig_time_us` when cold (mirror steering) |
| **Once established** | N/A | `circuit_latency_us` + bytes/BW |
| **Capacity** | Unlimited | `max_circuits` pool, LRU eviction |

### OCS Preset: Training → Inference Pre-Configuration

Core question: can **training-time routing patterns** pre-configure OCS circuits before inference begins, for zero runtime reconfig?

| Mode | Reconfig | Circuits established | Use case |
| ---- | -------- | -------------------- | -------- |
| `ocs_pipeline` | Per-microbatch, inline | Before each scatter | Runtime adaptability |
| `ocs_dbo` | Hidden behind compute | Batch K+1 during batch K | Mask reconfig latency |
| `ocs_preset` | **None during inference** | **Before first token** | Training→inference pre-config |
| `ocs_online` | Adaptive, amortized | Per-step from accumulated affinity | Inference-self-learning |

```bash
bash scripts/run_preset_pipeline.sh data/routing_traces/routing.json   # train → plan → preset → compare
python scripts/compare_ocs.py --mode all                                # EPS vs OCS runtime vs OCS preset
```

Preset strategies: `oracle`, `affinity`, `volume`, `random`, `none`. Key files: [src/ocs/preconfig.py](src/ocs/preconfig.py), [src/eval/affinity_consistency.py](src/eval/affinity_consistency.py).

### OCS Online Affinity

No separate training phase — track expert co-activation *during* inference and periodically recompute the circuit plan, with exponential decay (`decay_factor` 0.99/step) for responsiveness to pattern shifts.

```bash
python3 -m src.launcher --config configs/ocs_online.yaml
```

## Real Qwen MoE Routing Capture

Capture real routing traces from Qwen MoE models into the canonical `RoutingTrace` JSON (same schema across all backends):

```bash
# MLX backend (Apple Silicon, native)
python moe_run.py --model models/Qwen3.6-35B-A3B-4bit --max-tokens 128

# vLLM backend (CUDA box)
python scripts/run_vllm.py run --model Qwen/Qwen3.6-35B-A3B \
    --prompt "Explain MoE routing." --max-tokens 128

# vLLM + Metal backend (Apple Silicon GPU; runs MLX-format models)
# 1. Install once: curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
# 2. Run (MASTER_ADDR fixes macOS loopback resolution):
source ~/.venv-vllm-metal/bin/activate
MASTER_ADDR=127.0.0.1 python scripts/run_vllm.py run \
    --model ./models/Qwen3.6-35B-A3B-4bit --max-tokens 256

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

## Project Structure

```
src/
├── model/        MoELayer, Router, Experts, Qwen expert loader, router replay
├── comm/         all-to-all, transport, topology, timeline export
├── ocs/          Circuit pool (LRU), placement, preconfig, online controller
├── runtime/      Per-rank worker, scheduler, process groups
├── train/        Trainer, load-balance loss, microbatching
├── data/         Synthetic dataset, routing schema, HF/MLX/vLLM capture, interventions
├── eval/         Overlap/OCS metrics, affinity consistency, profiler
└── utils/        Timer, logging, seed
configs/          YAML configs (synthetic, realistic, OCS, Qwen, preset)
scripts/          Visualization, comparison, export, validation, presets
docs/             Research discussions, architecture alignment
data/             Reference routing traces for replay
```

## Configuration Reference

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `world_size` | Number of ranks | 4 |
| `model.num_experts` | Total experts (= world_size × experts_per_rank) | 4 |
| `model.experts_per_rank` | Experts per GPU | 1 |
| `model.hidden_dim` | Token embedding dim | 256 |
| `model.top_k` | Top-K gating | 1 |
| `routing.strategy` | `fixed`, `top1`, `top2`, `uniform_random` | `fixed` |
| `runtime.mode` | `serial`, `overlap`, `ocs_pipeline`, `ocs_dbo`, `ocs_preset`, `ocs_online` | `serial` |
| `ocs.enabled` | Enable OCS circuit pool | false |
| `ocs.max_circuits` | Max simultaneous optical circuits | 32 |
| `ocs.reconfig_time_us` | Circuit reconfig time (mirror steering) | 50.0 |
| `ocs.circuit_latency_us` | Hot-path latency | 1.0 |
| `ocs.circuit_bandwidth_gbps` | Circuit bandwidth | 200.0 |
| `ocs.preset.strategy` | Plan computation: `coactivation` | `coactivation` |
| `ocs.online.update_interval_steps` | Recompute plan every N steps | 1 |
| `ocs.online.decay_factor` | Exponential decay per step | 0.99 |

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

## License

MIT
