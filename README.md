# MoE Communication Research Testbed

CPU-based MoE overlap-algorithm testbed. Real all-to-all over TCP via PyTorch Gloo with per-rank expert-parallelism, top-K gating, hierarchical topology delay modeling, and **OCS (Optical Circuit Switching)** circuit-pool simulation. Verifies mechanism correctness *before* GPU cluster deployment.

**New: OCS Preset mode** — pre-configure OCS circuits from training-time routing affinity for zero-reconfig inference. [`docs/ocs_pretrigger_research_discussion.md`](docs/ocs_pretrigger_research_discussion.md)

## Quick Start

```bash
pip install torch pyyaml

# Small verification (4 ranks, 8 experts)
python3 -m src.launcher --config configs/synthetic_moe.yaml

# Realistic cluster (16 ranks, 64 experts, FFN, top-2, multi-pod topology)
python3 -m src.launcher --config configs/realistic_16gpu.yaml

# OCS pipeline (4 ranks, optical circuit switching)
python3 -m src.launcher --config configs/ocs_demo.yaml

# OCS vs EPS comparison (overlap pipeline, different transport)
python scripts/compare_ocs.py

# --- Real Qwen MoE weights + OCS (authentic model weights, requires export first) ---

# Export Qwen expert weights (one-time setup)
python3 scripts/export_qwen_experts.py \
    --model models/Qwen3.6-35B-A3B-4bit \
    --output exported_qwen_weights --max-layers 1

# Capture routing trace from real model
python moe_run.py --model models/Qwen3.6-35B-A3B-4bit \
    --prompt "Explain MoE routing." --max-tokens 128

# Qwen + OCS lite (8 experts, fast verification)
python3 -m src.launcher --config configs/qwen_ocs_lite.yaml

# Qwen + OCS pipeline (32 experts, full experiment)
python3 -m src.launcher --config configs/qwen_ocs_pipeline.yaml

# Qwen + OCS DBO (dual-batch overlap, lookahead)
python3 -m src.launcher --config configs/qwen_ocs_dbo.yaml

# View results
open outputs/traces/trace_viewer.html    # interactive HTML (click "EP Layout")
open outputs/traces/ocs_view.html        # OCS circuit analysis

# --- OCS Preset: training → inference pre-configuration ---

# Compute placement plan from training trace
python3 scripts/compute_preset_plan.py \
    --trace data/routing_traces/routing.json \
    --output outputs/preset_plan.json \
    --max-circuits 16 --experts-per-rank 64 --world-size 4

# Validate training-to-inference affinity consistency
python3 scripts/validate_affinity.py \
    --train-trace data/routing_traces/routing_pretrained.json \
    --infer-trace data/routing_traces/routing_finetuned.json \
    --num-experts 256 --top-k 8

# Run preset inference (circuits pre-configured, zero runtime reconfig)
python3 -m src.launcher --config configs/ocs_preset.yaml

# Full pipeline: train → plan → preset → compare
bash scripts/run_preset_pipeline.sh data/routing_traces/routing.json
```

## OCS Preset: Training → Inference Pre-Configuration

The core research question: can **training-time expert routing patterns** predict inference-time OCS circuit needs well enough to pre-configure the fabric *before* inference begins?

### Pipeline

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

### Modes

| Mode | Reconfig | When Circuits Established | Use Case |
|------|----------|--------------------------|----------|
| `ocs_pipeline` | Per-microbatch, inline | Before each scatter | Runtime adaptability |
| `ocs_dbo` | Hidden behind compute | Batch K+1 during batch K | Mask reconfig latency |
| **`ocs_preset`** | **None during inference** | **Before first token** | **Training→inference pre-config** |

### Preset Strategies

| Strategy | Source | Expected Hit Rate |
|----------|--------|-------------------|
| `oracle` | Inference trace itself | Upper bound (~100%) |
| `affinity` | Training co-activation matrix | Depends on train/inference correlation |
| `volume` | Training traffic volume | Good for bandwidth-bound workloads |
| `random` | Random circuits | Lower bound |
| `none` | EPS baseline (no OCS) | Control |

### Key Metrics (`src/eval/affinity_consistency.py`)

- **JS divergence**: per-layer Jensen-Shannon divergence of expert distributions (0 = identical)
- **Top-K overlap**: fraction of tokens with matching expert assignments
- **Affinity correlation**: Pearson R between training and inference co-activation matrices
- **Estimated hit rate**: theoretical OCS hit rate from training affinity
- **Preset viability**: `high` (>0.7), `medium` (0.4–0.7), `low` (<0.4)

### OCS Preset Metrics (`src/eval/metrics.py`)

- **ocs_hit_rate**: fraction of A2A target pairs already in pool
- **zero_reconfig_rate**: fraction of transfers incurring 0 reconfig
- **preset_utilization**: fraction of pre-established circuits actually used
- **per_step_latency_us**: wall time per inference step

### Comparison Workflow

```bash
# Three-way comparison: EPS baseline vs OCS runtime vs OCS preset
python scripts/compare_ocs.py --mode all
open outputs/traces/ocs_comparison/ocs_comparison.html

# Preset-only comparison
python scripts/compare_ocs.py --mode preset
```

## Architecture

```
tokens [B*S, H] --> Router --> expert_ids [T, K] + gate_weights [T, K]
                        |
                        v
              scatter_tokens()  -- groups by (expert_id // experts_per_rank),
                                   packs [token | local_expert_id | orig_idx],
                                   all_to_all_single (async in overlap mode)
                        |
                        v
              compute_experts() -- dispatches to correct local expert
                                   via per-expert masking
                        |
                        v
              gather_tokens()   -- all_to_all_single back,
                                   unpacks orig_idx to reconstruct order
                        |
                        v
              combine_expert_outputs() -- top-K: softmax gate weights, weighted sum
```

### Expert Parallelism (EP)

```
expert_id --> target_rank = expert_id // experts_per_rank
              local_expert = expert_id %  experts_per_rank

Constraint: num_experts = world_size * experts_per_rank
```

**4-rank example** (8 experts, 2/rank):

| Rank | Experts |
| ---- | ------- |
| 0    | 0, 1    |
| 1    | 2, 3    |
| 2    | 4, 5    |
| 3    | 6, 7    |

**16-GPU cluster** (64 experts, 4/rank, `realistic_16gpu.yaml`):

| Pod | Node | Ranks | Experts |
| --- | ---- | ----- | ------- |
| 0   | 0    | 0-3   | 0-15    |
| 0   | 1    | 4-7   | 16-31   |
| 1   | 0    | 8-11  | 32-47   |
| 1   | 1    | 12-15 | 48-63   |

### Network Topology (3-tier)

| Tier       | Link            | Latency | Bandwidth |
| ---------- | --------------- | ------- | --------- |
| INTRA_NODE | NVLink/NVSwitch | ~2 µs  | 600 GB/s  |
| INTRA_POD  | InfiniBand      | ~5 µs  | 200 GB/s  |
| CROSS_POD  | IB fabric       | ~15 µs | 100 GB/s  |

Delay = `latency + tensor_bytes / (bandwidth_gbps * 1000)`. All tiers configurable in YAML.

### Overlap Pipeline

```
mb_0: [scatter_0 fire] -------+
      (in flight)              |
mb_1: [scatter_1 fire]        | overlap zone
      [scatter_0 wait]        |
      [compute_0]             |
      [gather_0 fire] --------+
tail: [scatter_1 wait] [compute_1] [gather_1]
```

### OCS (Optical Circuit Switching)

Models the difference between two physical transport layers:

|                            | EPS (Electrical Packet Switching) | OCS (Optical Circuit Switching)                  |
| -------------------------- | --------------------------------- | ------------------------------------------------ |
| **Connection**       | Always-on, per-packet routed      | Finite pool of reconfigurable circuits           |
| **Setup cost**       | None (statistical multiplexing)   | `reconfig_time_us` when cold (mirror steering) |
| **Once established** | N/A                               | Fast path:`circuit_latency_us` + bytes/BW      |
| **Capacity**         | Unlimited concurrent pairs        | `max_circuits` pool, LRU eviction on overflow  |

**OCS Pipeline mode** (`ocs_pipeline`): Pre-establishes circuits before each scatter, overlapping reconfiguration with computation from prior micro-batches. After cold start, circuits stay hot → near-zero reconfig overhead.

**Dual-Batch Overlap** (`ocs_dbo`): Pre-establishes circuits for batch K+1 during batch K's compute window (lookahead). Reconfig cost fully hidden behind compute for ≥3 microbatches.

```
ocs_pipeline:
  mb_0: [pre_establish_0] [scatter_0 fire] -------+
        (reconfig exposed)  (in flight)            |
  mb_1: [pre_establish_1] [scatter_1 fire]         | overlap zone
        (reconfig hidden)  [scatter_0 wait]        |
                           [compute_0]             |
                           [gather_0 fire] --------+
  tail: ... (same as overlap pipeline)

ocs_dbo (lookahead):
  mb_0: [pre_establish_1] [scatter_0 fire] ...     ← circuits for K=1 set up during K=0 compute
  mb_1: [pre_establish_2] [scatter_1 fire] ...     ← reconfig fully hidden
```

**Circuit pool**: Implemented as `OrderedDict` keyed by `(src_rank, dst_rank)` — O(1) LRU eviction via `popitem(last=False)`, hot-path via `move_to_end()`. When pool is full and a new circuit is needed, the least recently used circuit is evicted and its optical path is reclaimed.

**Expert affinity tracking** (`src/ocs/placement.py`): Tracks co-activation counts from router outputs. A greedy clustering algorithm suggests expert-to-rank placements that group frequently co-activated experts on the same rank, minimizing cross-rank circuit reconfigurations. **New:** `export_affinity()` saves co-activation data for cross-phase analysis. `compute_circuit_plan()` converts affinity to an ordered rank-pair circuit placement list.

### OCS Pre-Configuration Pipeline (`src/ocs/preconfig.py`)

```
RoutingTrace → ExpertAffinityTracker → compute_circuit_plan() → placement plan JSON
    (train)         (co-activation)        (rank pairs)            (preset input)
```

## Project Structure

```
src/
├── main.py, launcher.py         # CLI + multiprocessing spawner
├── model/
│   ├── moe_layer.py             # MoELayer: ModuleList[experts], set_rank(), expert_id_to_local()
│   ├── router.py                # fixed | top1 | top2 | uniform_random
│   ├── router_replay.py         # ReplayRouter, LayerCyclingReplayRouter (trace-driven routing)
│   └── experts.py               # TinyExpert (Linear) | FFNExpert (Linear->GELU->Linear)
├── comm/
│   ├── all_to_all.py            # scatter_tokens, gather_tokens, combine_expert_outputs, DispatchResult
│   ├── transport.py             # Wraps dist ops, injects topology/OCS/flat delay
│   ├── topology.py              # Topology, TopologyConfig, LinkTier (3-tier hierarchy)
│   ├── timeline.py              # Chrome Trace JSON export (with EP + OCS metadata)
│   └── buffers.py               # Double-buffer dataclass
├── ocs/
│   ├── circuit.py               # OcsCircuit, OcsCircuitPool (OrderedDict LRU), OcsPoolMetrics
│   ├── topology.py              # OcsTopologyConfig, OcsTopology wrapper
│   ├── placement.py             # ExpertAffinityTracker (co-activation → greedy clustering)
│   └── preconfig.py             # NEW: RoutingTrace → affinity → circuit plan pipeline
├── runtime/
│   ├── worker.py                # Per-rank: init PG, build MoE+Transport+Topology+OCS, run scheduler
│   ├── scheduler.py             # run_serial, run_overlap, run_ocs_pipeline, run_ocs_dbo, run_ocs_preset
│   └── process_group.py         # dist.init_process_group / cleanup
├── train/
│   ├── trainer.py               # Trainer: epoch loop, metrics, checkpointing
│   ├── step.py                  # train_step_serial (forward→loss→backward→step)
│   ├── loss.py                  # task_loss (MSE) + load_balance_loss (Switch Transformer)
│   └── microbatch.py            # split_microbatches utility
├── data/
│   ├── synthetic.py             # SyntheticMixtureDataset (K Gaussian modes)
│   ├── routing_schema.py        # RoutingTrace, TokenRoute, LayerRoute, RunMeta dataclasses
│   ├── routing_capture.py       # HF inference hook for routing trace capture
│   ├── routing_interventions.py # RouterSteering: force/bias/ablate experts
│   └── model_utils.py           # ModelLayout introspection
├── eval/
│   ├── metrics.py               # overlap_ratio, ocs_overlap_ratio, ocs_step_metrics, ocs_preset_metrics
│   ├── affinity_consistency.py  # NEW: JS divergence, top-K overlap, affinity correlation, estimated hit rate
│   ├── profiler.py              # Multi-rank trace aggregation + ocs_summary()
│   └── plots.py                 # Gantt charts (matplotlib optional)
└── utils/                       # timer (ns-precision), logging, seed

configs/
├── base.yaml                    # 4 ranks, 4 experts, 1 expert/rank
├── synthetic_moe.yaml           # 4 ranks, 8 experts, 2/rank, tiny, fast iteration
├── realistic_16gpu.yaml         # 16 ranks, 64 experts, 4/rank, FFN, top-2, 2p×2n×4r topology
├── realistic_32gpu.yaml         # 32 ranks, 128 experts, 4/rank, FFN, top-2, 2p×4n×4r topology
├── mac_cpu.yaml                 # Overlap mode + 200µs flat delay
├── ocs_demo.yaml                # OCS pipeline mode, 50µs reconfig, 8-circuit pool
├── ocs_dbo_demo.yaml            # OCS dual-batch overlap, 100µs reconfig, 4 microbatches
├── ocs_preset.yaml              # NEW: OCS preset mode — pre-configured circuits, zero runtime reconfig
├── preset_ablation.yaml         # NEW: Ablation: oracle/affinity/volume/random/none strategies
├── compare_ocs_base.yaml        # Baseline for OCS comparison (overlap + EPS, no OCS)
├── compare_ocs_on.yaml          # OCS counterpart (ocs_pipeline, same workload)
├── routing_replay.yaml          # Replay captured routing traces with synthetic experts
├── ocs_replay.yaml              # OCS + routing replay (synthetic experts)
├── ocs_dbo_replay.yaml          # OCS DBO + routing replay (synthetic experts)
├── qwen_replay.yaml             # Real Qwen experts + routing replay (no OCS)
├── qwen_ocs_base.yaml           # Real Qwen experts + routing replay + OCS (base)
├── qwen_ocs_lite.yaml           # Qwen + OCS lite — 8 experts, fast verification
├── qwen_ocs_pipeline.yaml       # Qwen + OCS pipeline — 32 experts, full experiment
└── qwen_ocs_dbo.yaml            # Qwen + OCS DBO — dual-batch overlap, lookahead

scripts/
├── run_synthetic.sh             # One-command: serial or overlap mode
├── trace_viz.py                 # Standalone HTML viewer (EP panel, topology groups, overlap stat)
├── ocs_viz.py                   # OCS circuit viewer (per-rank stats, reuse bars, event timeline)
├── compare_ocs.py               # A/B/C comparison (EPS/OCS-runtime/OCS-preset → HTML report)
├── compute_preset_plan.py       # NEW: CLI — trace → circuit placement plan JSON
├── validate_affinity.py         # NEW: CLI — train vs inference affinity consistency report
├── run_preset_pipeline.sh       # NEW: End-to-end pipeline: train → plan → preset → compare
├── export_qwen_experts.py       # Extract Qwen MoE weights (MLX or safetensors) → per-expert .npz
├── run_research.py              # HF-based CLI: run/analyze/compare/intervene/ablate
├── analyze_routing_for_ocs.py   # Route trace → OCS placement analysis
├── visualize_routing.py         # 4-panel: routing agreement, expert load, gain/loss
└── merge_traces.py              # Merge per-rank traces for chrome://tracing
```

## Configuration Reference

| Parameter                      | Description                                                                 | Default         |
| ------------------------------ | --------------------------------------------------------------------------- | --------------- |
| `world_size`                 | Number of ranks                                                             | 4               |
| `model.num_experts`          | Total experts (= world_size × experts_per_rank)                            | 4               |
| `model.experts_per_rank`     | Experts per GPU                                                             | 1               |
| `model.hidden_dim`           | Token embedding dimension                                                   | 256             |
| `model.expert_type`          | `tiny` (Linear) or `ffn` (2-layer GELU)                                 | `tiny`        |
| `model.top_k`                | Top-K gating                                                                | 1               |
| `routing.strategy`           | `fixed`, `top1`, `top2`, `uniform_random`                           | `fixed`       |
| `runtime.mode`               | `serial`, `overlap`, `ocs_pipeline`, `ocs_dbo`, `ocs_preset`, or `train_serial` | `serial`      |
| `delay.comm_delay_us`        | Flat delay (ignored if topology or OCS enabled)                             | 0               |
| `delay.comm_delay_jitter_us` | Random jitter ± on flat delay                                              | 0               |
| `topology.enabled`           | Use hierarchical topology delays                                            | false           |
| `ocs.enabled`                | Enable OCS circuit pool (EPS otherwise)                                     | false           |
| `ocs.max_circuits`           | Max simultaneous optical circuits in pool                                   | 32              |
| `ocs.reconfig_time_us`       | Time to reconfigure one circuit (mirror steering)                           | 50.0            |
| `ocs.circuit_latency_us`     | Base optical path latency (once established)                                | 1.0             |
| `ocs.circuit_bandwidth_gbps` | Circuit bandwidth (optical = much higher than EPS)                          | 200.0           |
| `ocs.placement_strategy`     | Expert→rank placement:`round_robin` or `affinity`                      | `round_robin` |
| `ocs.preset.source`          | Preset plan source: `trace` (from routing trace), `plan` (from JSON file) | `trace`       |
| `ocs.preset.plan_path`       | Path to pre-computed circuit placement plan JSON (when source=plan)         | `""`          |
| `ocs.preset.strategy`        | Plan computation strategy: `coactivation`                                  | `coactivation` |

## OCS Comparison (EPS vs OCS)

Run an A/B comparison — same workload, same overlap pipeline, different transport:

```bash
python scripts/compare_ocs.py
open outputs/traces/ocs_comparison/ocs_comparison.html
```

The comparison isolates **only the OCS transport effect** by using `overlap` (EPS) vs `ocs_pipeline` (OCS) — identical pipeline structure, different physical layer.

**Sample results** (4 ranks, 8 experts, 2 microbatch, 500µs EPS delay → 50µs OCS reconfig):

| Metric          | Baseline (overlap + EPS) | OCS (ocs_pipeline)      | Delta            |
| --------------- | ------------------------ | ----------------------- | ---------------- |
| Total wall time | 23,269 µs               | 17,424 µs              | **-25.1%** |
| Comm time       | 23,269 µs               | 17,091 µs              | -26.6%           |
| OCS overhead    | —                       | 333 µs                 | —               |
| Circuit reuse   | —                       | 96.7%                   | —               |
| Reconfig time   | —                       | 150 µs (3 cold starts) | —               |
| LRU evictions   | —                       | 0                       | —               |

**Why OCS wins**: Once circuits are established (3 cold starts), the remaining 87/90 requests hit the fast optical path. OCS base latency (1µs) + transfer time (0.7µs per 128KB) is negligible compared to 500µs EPS delay. The 333µs of OCS overhead is pre-establish timer events, not actual reconfig cost.

**Realistic expectations**: In a real OCS fabric, reconfig time is 10-100µs (MEMS mirror), circuit BW is 200-400 Gbps per λ, and pool size is 8-64 circuits per node. The benefit is larger with higher EPS delay (cross-pod traffic) and stable routing patterns (high circuit reuse).

## Key Design Decisions

- **Metadata packing:** `local_expert_id` and `original_index` are packed as float columns alongside tokens — travels through `all_to_all_single` with zero extra communication rounds.
- **Padded equal-size all-to-all:** Counts exchanged via `all_gather`, max computed, padded to uniform chunks. Required by `all_to_all_single` on Gloo.
- **Top-K dispatch:** Tokens replicated K times, `index_add_` handles duplicate indices during gather. Gate weights applied via softmax-weighted sum.
- **Async overlap:** Scatter/gather issued with `async_op=True`, `handle.wait()` called when result needed. Identical pattern to NCCL CUDA-stream overlap.
- **Topology per-process:** Each spawned process pre-assigns *all* ranks to populate the topology lookup table (required by `spawn` start method).
- **No GPU:** Runs entirely on CPU with real TCP communication. Overlap algorithm verified here transfers directly to GPU/NCCL.
- **OCS gating:** All OCS code paths are gated behind `ocs.enabled: true` and `ocs_circuit_pool is not None` checks. When disabled, the transport takes the exact same paths as before — zero overhead, identical behavior.
- **OCS circuit pool (OrderedDict):** LRU eviction in O(1) via `popitem(last=False)`, hot-path promotion via `move_to_end()`. Mirrors real OCS switch behavior: least recently used lightpath is torn down when pool is exhausted.
- **OCS delay model:** `compute_delay()` = reconfig (0 if hot, `reconfig_time_us` if cold/evicted) + `circuit_latency_us` + `bytes / (bw_gbps * 1000)`. This captures the bimodal nature of OCS: penalty on circuit setup, fast path once established.
- **OCS pre-establish:** Circuits for micro-batch K are established BEFORE the scatter fires. In pipeline mode, this runs during the gap between gathers. In DBO mode (≥3 microbatches), it runs during the PRIOR batch's compute — reconfig is fully hidden.
- **Comparison isolation:** `compare_ocs.py` uses `overlap` vs `ocs_pipeline` (not `serial` vs `ocs_pipeline`). Both use the same pipeline structure — the only difference is the transport layer. This isolates the OCS effect from the overlap benefit.
- **Preset pre-configuration:** Circuits are established from a training-derived plan BEFORE the first micro-batch. The `pre_config()` method batch-loads circuits without incurring reconfig delay — simulating the scenario where the OCS fabric is pre-wired for inference.
- **Affinity-driven placement:** `compute_circuit_plan()` maps expert co-activation counts to rank-pair circuit priorities. Higher affinity pairs get circuits first, maximizing hit rate under pool capacity constraints.
- **Train→inference validation:** `affinity_consistency.py` provides JS divergence, top-K overlap, and Pearson correlation metrics to quantify whether training patterns predict inference routing — the core hypothesis behind OCS preset.

## Viewing Traces

```bash
# Interactive HTML (recommended) — click "EP Layout" for expert mapping
open outputs/traces/trace_viewer.html

# OCS circuit analysis — pool stats, reuse bars, event timeline
open outputs/traces/ocs_view.html

# Chrome Trace Viewer
open chrome://tracing  →  load outputs/traces/merged_trace.json

# Perfetto
open https://ui.perfetto.dev  →  drag in merged_trace.json
```

**Event colors:** Green=Route, Blue=Scatter, Orange=Compute, Purple=Gather, Pink=Combine, Cyan=ScatterWait, Red=AllToAll, Green/Red bars=OCS pre-establish (overlapped/exposed)

## What This IS and IS NOT

**IS:** Mechanism verification testbed. Answers: does the async overlap algorithm work correctly? Does OCS circuit pooling reduce all-to-all latency? Shows EP mapping, topology structure, OCS circuit dynamics, comm/compute interleaving. Fast iteration (seconds, not hours).

**IS NOT:** Performance simulator. Do NOT extrapolate timings to GPU clusters. The comm/compute ratio is inverted here (comm dominates because experts are tiny). On real GPUs, compute dwarfs communication → overlap ratio much higher.

Progression: verify here → profile single-GPU NCCL → scale multi-node.

## License

MIT
