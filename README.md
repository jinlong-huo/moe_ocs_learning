# MoE + OCS Research Testbed

A  **Mixture of Experts (MoE)** testbed for learning and verifying communication algorithms — expert parallelism, all-to-all dispatch, async overlap pipelining, and **Optical Circuit Switching (OCS)** — before deploying to GPU clusters.

Built with PyTorch + Gloo (real TCP all-to-all).

## Quick Start

```bash
pip install torch pyyaml

# 5-minute verification (4 ranks, 8 experts)
python3 -m src.launcher --config configs/synthetic_moe.yaml

# View the interactive trace
open outputs/traces/trace_viewer.html    # click "EP Layout" for expert mapping
```

## What You'll Learn

This testbed is organized into three learning paths.

### Path 1 — MoE Basics

Start here if you're new to Mixture of Experts. You'll learn:

1. **Expert Parallelism** — how experts are partitioned across GPUs and how `expert_id → (rank, local_expert)` mapping works
2. **Top-K Routing** — how a learned gate selects which experts process each token
3. **All-to-All Dispatch** — the 4-phase pipeline: route → scatter → compute → gather → combine

```bash
# Fixed routing (deterministic, good for understanding the flow)
python3 -m src.launcher --config configs/synthetic_moe.yaml 

# Top-2 routing (more realistic, uses softmax + topk)
# Edit configs/synthetic_moe.yaml: set routing.strategy=top2, model.top_k=2
```

**Key files:** [src/model/moe_layer.py](src/model/moe_layer.py), [src/model/router.py](src/model/router.py), [src/comm/all_to_all.py](src/comm/all_to_all.py)

July29: Those part can be removed then.

### Path 2 — Overlap & Pipelining

Once you understand the basic dispatch, learn how to **hide communication behind computation**:

1. **Micro-batch pipelining** — split tokens into chunks, overlap scatter→compute→gather across chunks
2. **Async operations** — fire-and-forget all-to-all, wait only when data is needed
3. **Double buffering** — use two buffers to avoid stalls between pipeline stages

```bash
# Overlap mode: see comm/compute interleaving
python3 -m src.launcher --config configs/mac_cpu.yaml

# Compare serial vs overlap in the trace viewer
open outputs/traces/trace_viewer.html
```

**Key files:** [src/runtime/scheduler.py](src/runtime/scheduler.py) (`run_overlap`), [src/comm/buffers.py](src/comm/buffers.py)

### Path 3 — Optical Circuit Switching (OCS)

The advanced path. Learn how optical circuit switching changes the communication model:

1. **Circuit pool** — finite pool of reconfigurable optical paths (LRU eviction, O(1) with OrderedDict)
2. **Reconfig cost** — cold start penalty vs hot-path speed (modeled after real MEMS mirror steering)
3. **OCS Pipeline** — pre-establish circuits before scatter, overlap reconfig with prior compute
4. **Dual-Batch Overlap (DBO)** — establish circuits for batch K+1 during batch K's compute (reconfig fully hidden)
5. **Preset mode** — pre-configure circuits from training-time routing affinity, zero runtime reconfig
6. **Online affinity mode (NEW)** — track expert co-activation *during* inference, continuously adjust circuits; no training phase needed

```bash
# OCS pipeline mode (see circuit establishment + reuse)
python3 -m src.launcher --config configs/ocs_demo.yaml

# OCS dual-batch overlap (reconfig hidden behind compute)
python3 -m src.launcher --config configs/ocs_dbo_demo.yaml

# Compare EPS vs OCS (same workload, different transport)
python scripts/compare_ocs.py
open outputs/traces/ocs_comparison/ocs_comparison.html

# OCS circuit analysis
open outputs/traces/ocs_view.html
```

**Key files:** [src/ocs/circuit.py](src/ocs/circuit.py), [src/ocs/placement.py](src/ocs/placement.py), [src/ocs/preconfig.py](src/ocs/preconfig.py)

## Core Concepts

### Expert Parallelism (EP)

```
expert_id → target_rank = expert_id // experts_per_rank
            local_expert  = expert_id %  experts_per_rank

Constraint: num_experts = world_size × experts_per_rank
```

**4-rank example** (8 experts, 2 per rank):

| Rank | Experts |
| ---- | ------- |
| 0    | 0, 1    |
| 1    | 2, 3    |
| 2    | 4, 5    |
| 3    | 6, 7    |

### Routing (Top-K Gating)

```
tokens [B×S, H] → Linear(hidden, num_experts) → softmax → topk → expert_ids [T, K] + gate_weights [T, K]
```

The router is a single linear layer. In `top_k=2` mode, each token is sent to 2 experts and their outputs are weighted by softmax-normalized gate scores.

**Router strategies:** `fixed` (deterministic, good for debugging), `top1`, `top2`, `uniform_random`

### All-to-All Dispatch Pipeline

```
tokens → [ROUTE] → [SCATTER all-to-all] → [EXPERT COMPUTE] → [GATHER all-to-all] → [COMBINE] → output
```

Each token is packed with `[token | local_expert_id | original_index]` as float columns, traveling through `all_to_all_single` with zero extra communication rounds.

### Overlap Pipeline

```
mb_0: [scatter_0 fire] ───────────────┐
      (in flight)                      │
mb_1: [scatter_1 fire]                 │ overlap zone
      [scatter_0 wait]                 │
      [compute_0]                      │
      [gather_0 fire] ─────────────────┘
tail: [scatter_1 wait] [compute_1] [gather_1]
```

Comm and compute interleave across micro-batches. Identical pattern to NCCL CUDA-stream overlap.

### Network Topology (3-tier)

Models realistic cluster hierarchy with configurable latency + bandwidth per tier:

| Tier       | Link            | Typical Latency | Typical Bandwidth |
| ---------- | --------------- | --------------- | ----------------- |
| INTRA_NODE | NVLink/NVSwitch | ~2 µs          | 600 GB/s          |
| INTRA_POD  | InfiniBand      | ~5 µs          | 200 GB/s          |
| CROSS_POD  | IB fabric       | ~15 µs         | 100 GB/s          |

Delay = `latency + tensor_bytes / (bandwidth_gbps × 1000)`. All configurable in YAML.

```bash
python3 -m src.launcher --config configs/realistic_16gpu.yaml
```

**Key file:** [src/comm/topology.py](src/comm/topology.py)

### OCS Circuit Pool

Models the difference between two physical transport layers:

|                            | EPS (Electrical Packet Switching) | OCS (Optical Circuit Switching)                  |
| -------------------------- | --------------------------------- | ------------------------------------------------ |
| **Connection**       | Always-on, per-packet routed      | Finite pool of reconfigurable circuits           |
| **Setup cost**       | None (statistical multiplexing)   | `reconfig_time_us` when cold (mirror steering) |
| **Once established** | N/A                               | Fast path:`circuit_latency_us` + bytes/BW      |
| **Capacity**         | Unlimited concurrent pairs        | `max_circuits` pool, LRU eviction on overflow  |

**Circuit pool** is an `OrderedDict` keyed by `(src_rank, dst_rank)` — O(1) LRU eviction via `popitem(last=False)`, hot-path promotion via `move_to_end()`.

```
ocs_pipeline:
  mb_0: [pre_establish_0] [scatter_0 fire] ───────┐
        (reconfig exposed)  (in flight)             │
  mb_1: [pre_establish_1] [scatter_1 fire]          │ overlap zone
        (reconfig hidden)  [scatter_0 wait]         │
                           [compute_0]              │
                           [gather_0 fire] ─────────┘

ocs_dbo (lookahead):
  mb_0: [pre_establish_1] [scatter_0 fire] ...  ← circuits for K=1 set up during K=0 compute
  mb_1: [pre_establish_2] [scatter_1 fire] ...  ← reconfig fully hidden
```

### OCS Preset: Training → Inference Pre-Configuration

```
```

The core research question: can **training-time expert routing patterns** predict inference OCS circuit needs well enough to pre-configure the fabric *before* inference begins?

```
Training Phase                    Inference Phase
─────────────                     ────────────────
Router outputs                   ┌──────────────┐
  ↓                              │ Pre-configure │
ExpertAffinityTracker            │ OCS circuits  │
(co-activation counts)           │ from plan     │
  ↓                              └──────┬───────┘
compute_circuit_plan()                  ↓
(expert pairs → rank pairs)       Zero reconfig
  ↓                              during inference
Export plan JSON                 (all circuits hot)
```

| Mode                     | Reconfig                        | When Circuits Established    | Use Case                                 |
| ------------------------ | ------------------------------- | ---------------------------- | ---------------------------------------- |
| `ocs_pipeline`         | Per-microbatch, inline          | Before each scatter          | Runtime adaptability                     |
| `ocs_dbo`              | Hidden behind compute           | Batch K+1 during batch K     | Mask reconfig latency                    |
| **`ocs_preset`** | **None during inference** | **Before first token** | **Training→inference pre-config** |
| **`ocs_online`** | Adaptive, amortized | Per-step from accumulated affinity | **Inference-self-learning** |

```bash
# Full pipeline: train → plan → preset → compare
bash scripts/run_preset_pipeline.sh data/routing_traces/routing.json

# Three-way comparison: EPS baseline vs OCS runtime vs OCS preset
python scripts/compare_ocs.py --mode all
```

**Preset strategies:** `oracle` (upper bound), `affinity` (training co-activation), `volume` (traffic volume), `random` (lower bound), `none` (EPS baseline)

**Key files:** [src/ocs/preconfig.py](src/ocs/preconfig.py), [src/eval/affinity_consistency.py](src/eval/affinity_consistency.py)

### OCS Online Affinity: Inference-Self-Learning

Instead of a separate training phase to pre-compute affinity → export plan → pre-configure circuits, the **online affinity mode** tracks expert co-activation *during* inference and continuously adjusts the OCS topology:

```
Step N:
  1. Pre-route all micro-batches → feed routing to OnlineAffinityController
  2. Controller accumulates co-activation counts (ExpertAffinityTracker)
  3. Every N steps: compute circuit plan from affinity, pre-establish
     high-affinity rank-pair circuits in the pool
  4. Per-microbatch: pre-establish + scatter/compute/gather (same as ocs_pipeline)
  5. Apply exponential decay to affinity counts (adapts to pattern shifts)

Step N+1: same, but with more accumulated data → better circuit placement
```

**Why it works:**
- The `OnlineAffinityController` composes `ExpertAffinityTracker` + `OcsCircuitPool`, accumulating routing decisions online and periodically computing circuit plans
- High-affinity circuits are pre-established; LRU handles natural eviction of unused low-affinity pairs
- Per-rank controller with `rank` filtering ensures each pool only holds circuits originating from its own rank — no wasted capacity
- Exponential decay (`decay_factor`, default 0.99/step) makes the system responsive to shifting routing patterns during long generation runs

**Strengths vs preset:**
- Single-phase — no separate training run needed
- Adaptive — responds to distribution shifts (unlike static preset)
- Progressive — starts like `ocs_pipeline`, approaches `ocs_preset` efficiency as affinity converges

**Limitations:**
- Requires top-K routing (K ≥ 2) for meaningful cross-expert co-activation
- Cold-start: first few steps are equivalent to `ocs_pipeline`
- Decay rate is per-step, not per-token — interpret decay_factor carefully with variable batch sizes

```bash
# Online affinity mode — self-learning from inference routing
python3 -m src.launcher --config configs/ocs_online.yaml
```

**Key files:** [src/ocs/online_controller.py](src/ocs/online_controller.py), [src/runtime/scheduler.py](src/runtime/scheduler.py) (`run_ocs_online`)

## Real Qwen MoE Weights

Beyond synthetic experts, you can run with **real Qwen MoE model weights** for authentic routing patterns and expert computation:

```bash
# 1. Export Qwen expert weights (one-time setup)
python3 scripts/export_qwen_experts.py \
    --model models/Qwen3.6-35B-A3B-4bit \
    --output exported_qwen_weights --max-layers 1

# 2. Capture routing trace from real model inference
python moe_run.py --model models/Qwen3.6-35B-A3B-4bit \
    --prompt "Explain MoE routing." --max-tokens 128

# 3. Run OCS experiments with real weights
python3 -m src.launcher --config configs/qwen_ocs_lite.yaml      # 8 experts, fast
python3 -m src.launcher --config configs/qwen_ocs_pipeline.yaml  # 32 experts, full
python3 -m src.launcher --config configs/qwen_ocs_dbo.yaml       # dual-batch overlap
```

## Project Structure

```
src/
├── model/        MoELayer, Router, Experts (FFN, Tiny), Qwen expert loader, router replay
├── comm/         all-to-all scatter/gather, transport layer, topology, timeline export
├── ocs/          Circuit pool (OrderedDict LRU), topology, affinity placement, preconfig, online controller
├── runtime/      Per-rank worker, scheduler (serial/overlap/ocs/dbo/preset/online), process groups
├── train/        Trainer, training step, load-balance loss, microbatch splitting
├── data/         Synthetic dataset, routing schema, HF+MLX routing capture, interventions
├── eval/         Overlap/OCS metrics, affinity consistency, profiler, Gantt charts
└── utils/        ns-precision timer, logging, seed
configs/          20 YAML configs (synthetic, realistic, OCS, Qwen, preset, ablation)
scripts/          Visualization, comparison, export, validation, presets, system diagrams
docs/             Research discussions, architecture alignment report
data/             Reference routing traces for replay experiments
```

## Configuration Reference

| Parameter                      | Description                                                                              | Default          |
| ------------------------------ | ---------------------------------------------------------------------------------------- | ---------------- |
| `world_size`                 | Number of ranks                                                                          | 4                |
| `model.num_experts`          | Total experts (= world_size × experts_per_rank)                                         | 4                |
| `model.experts_per_rank`     | Experts per GPU                                                                          | 1                |
| `model.hidden_dim`           | Token embedding dimension                                                                | 256              |
| `model.expert_type`          | `tiny` (Linear) or `ffn` (2-layer GELU)                                              | `tiny`         |
| `model.top_k`                | Top-K gating                                                                             | 1                |
| `routing.strategy`           | `fixed`, `top1`, `top2`, `uniform_random`                                        | `fixed`        |
| `runtime.mode`               | `serial`, `overlap`, `ocs_pipeline`, `ocs_dbo`, `ocs_preset`, `ocs_online`, `train_serial` | `serial`       |
| `delay.comm_delay_us`        | Flat delay (ignored if topology or OCS enabled)                                          | 0                |
| `topology.enabled`           | Use hierarchical topology delays                                                         | false            |
| `ocs.enabled`                | Enable OCS circuit pool                                                                  | false            |
| `ocs.max_circuits`           | Max simultaneous optical circuits                                                        | 32               |
| `ocs.reconfig_time_us`       | Circuit reconfiguration time (mirror steering)                                           | 50.0             |
| `ocs.circuit_latency_us`     | Base optical path latency (once established)                                             | 1.0              |
| `ocs.circuit_bandwidth_gbps` | Circuit bandwidth                                                                        | 200.0            |
| `ocs.placement_strategy`     | Expert→rank placement:`round_robin` or `affinity`                                   | `round_robin`  |
| `ocs.preset.strategy`        | Plan computation:`coactivation`                                                        | `coactivation` |
| `ocs.online.update_interval_steps` | Recompute circuit plan every N steps                                             | 1                |
| `ocs.online.decay_factor`   | Exponential decay per step (1.0 = no decay)                                             | 0.99             |

## Viewing Results

```bash
# Interactive HTML trace (recommended) — click "EP Layout" for expert mapping
open outputs/traces/trace_viewer.html

# OCS circuit analysis — pool stats, reuse bars, event timeline
open outputs/traces/ocs_view.html

# Chrome Trace Viewer / Perfetto
open chrome://tracing  →  load outputs/traces/merged_trace.json
open https://ui.perfetto.dev  →  drag in merged_trace.json
```

**Event colors:** Green=Route, Blue=Scatter, Orange=Compute, Purple=Gather, Pink=Combine, Cyan=ScatterWait, Red=AllToAll

## Architecture Alignment

This testbed's 4-phase pipeline (route → dispatch → compute → combine) is structurally aligned with five production MoE frameworks: Megatron-LM, Tutel, Megablocks, DeepEP, and DeepSpeed-MoE. The gaps (softmax routing, capacity factor, load balancing loss) are known extension points for later experimental stages, not architectural mismatches.

See the [full alignment report](docs/alignment.md) for per-framework comparison tables and priority-ordered recommendations.

## Key Design Decisions

- **Metadata packing:** `local_expert_id` and `original_index` travel as float columns alongside tokens through `all_to_all_single` — zero extra communication rounds
- **Async overlap:** Scatter/gather issued with `async_op=True`, `handle.wait()` called when result needed. Identical pattern to NCCL CUDA-stream overlap
- **No GPU:** Runs entirely on CPU with real TCP communication. Algorithm verified here transfers directly to GPU/NCCL
- **OCS gating:** All OCS code paths are gated behind `ocs.enabled: true` — when disabled, zero overhead, identical behavior
- **OCS circuit pool:** `OrderedDict` LRU eviction in O(1). Mirrors real OCS switch behavior: least recently used lightpath torn down when pool exhausted

For the full list of 16 design decisions with rationale, see the source files in [src/ocs/](src/ocs/) and [src/comm/](src/comm/).

## License

MIT
