"""Per-rank worker: the main execution loop for one process.

Each worker:
  1. Initializes its process group
  2. Builds the MoE layer (owning one expert)
  3. Generates synthetic input data
  4. Runs the scheduler (serial or overlap mode)
  5. Records timeline events
  6. Exports trace and metrics
"""
from __future__ import annotations

import os
from typing import Dict

import torch

from src.runtime.process_group import init_process_group, cleanup_process_group, get_rank
from src.runtime.scheduler import (
    run_serial, run_overlap, run_train_serial,
    run_ocs_pipeline, run_ocs_dbo,
    broadcast_model_params,
)
from src.model.moe_layer import MoELayer
from src.model.router_replay import ReplayRouter, LayerCyclingReplayRouter
from src.model.qwen_experts import create_qwen_moe_layer, QwenMoELayerWrapper
from src.data.routing_schema import RoutingTrace
from src.comm.transport import Transport
from src.comm.topology import Topology, TopologyConfig
from src.ocs.topology import OcsTopology, OcsTopologyConfig
from src.ocs.placement import ExpertAffinityTracker
from src.train.trainer import Trainer
from src.utils.timer import Timer
from src.utils.logging import log, log_summary
from src.utils.seed import set_seed
from src.comm.timeline import export_chrome_trace


def worker(
    rank: int,
    world_size: int,
    config: Dict,
    trace_dir: str = "outputs/traces",
) -> None:
    """Entry point for a single spawned process."""
    # ── Init ────────────────────────────────────────────────────
    set_seed(42 + rank)
    init_process_group(
        rank=rank,
        world_size=world_size,
        master_addr=config.get("master_addr", "127.0.0.1"),
        master_port=config.get("master_port", 29500),
        backend=config.get("backend", "gloo"),
    )

    timer = Timer(rank)

    # ── Build model ─────────────────────────────────────────────
    model_cfg = config["model"]
    delay_cfg = config.get("delay", {})
    runtime_cfg = config["runtime"]
    data_cfg = config["data"]

    # -- Build topology (if enabled) ---------------------------------
    topo_cfg = config.get("topology", {})
    topology = None
    if topo_cfg.get("enabled", False):
        topology = Topology(TopologyConfig(
            num_pods=topo_cfg.get("num_pods", 1),
            nodes_per_pod=topo_cfg.get("nodes_per_pod", 1),
            ranks_per_node=topo_cfg.get("ranks_per_node", world_size),
            intra_node_latency_us=topo_cfg.get("intra_node_latency_us", 1.0),
            intra_pod_latency_us=topo_cfg.get("intra_pod_latency_us", 3.0),
            cross_pod_latency_us=topo_cfg.get("cross_pod_latency_us", 10.0),
            intra_node_bandwidth_gbps=topo_cfg.get("intra_node_bandwidth_gbps", 900.0),
            intra_pod_bandwidth_gbps=topo_cfg.get("intra_pod_bandwidth_gbps", 200.0),
            cross_pod_bandwidth_gbps=topo_cfg.get("cross_pod_bandwidth_gbps", 100.0),
            delay_multiplier=topo_cfg.get("delay_multiplier", 1.0),
        ))
        # Pre-assign ALL ranks (each spawned process has its own copy of topology)
        for r in range(world_size):
            topology.assign(r)
        loc = topology.get_location(rank)
        log(rank, f"Topology: pod={loc.pod_id} node={loc.node_id} local={loc.local_rank}")

    # -- Build OCS topology and circuit pool (if enabled) --------------------
    ocs_cfg = config.get("ocs", {})
    ocs_topology = None
    ocs_pool = None
    affinity_tracker = None
    if ocs_cfg.get("enabled", False):
        ocs_topology = OcsTopology(OcsTopologyConfig(
            enabled=True,
            max_circuits=ocs_cfg.get("max_circuits", 32),
            reconfig_time_us=ocs_cfg.get("reconfig_time_us", 50.0),
            circuit_latency_us=ocs_cfg.get("circuit_latency_us", 1.0),
            circuit_bandwidth_gbps=ocs_cfg.get("circuit_bandwidth_gbps", 200.0),
            placement_strategy=ocs_cfg.get("placement_strategy", "round_robin"),
        ))
        ocs_pool = ocs_topology.pool
        log(rank, f"OCS: {ocs_cfg['max_circuits']} max circuits, "
            f"{ocs_cfg['reconfig_time_us']}us reconfig, "
            f"{ocs_cfg['circuit_bandwidth_gbps']}Gbps BW")

        # Build affinity tracker if using affinity placement
        if ocs_cfg.get("placement_strategy") == "affinity":
            affinity_tracker = ExpertAffinityTracker(model_cfg["num_experts"])
            log(rank, "OCS affinity placement: tracking expert co-activation")

    transport = Transport(
        timer=timer,
        comm_delay_us=delay_cfg.get("comm_delay_us", 0.0),
        comm_delay_jitter_us=delay_cfg.get("comm_delay_jitter_us", 0.0),
        topology=topology,
        rank=rank,
        world_size=world_size,
        ocs_circuit_pool=ocs_pool,
    )

    experts_per_rank = model_cfg.get("experts_per_rank", 1)
    expert_type = model_cfg.get("expert_type", "tiny")
    routing_strategy = config.get("routing", {}).get("strategy", "fixed")

    # ── Build model: synthetic MoELayer or real Qwen experts ────────
    if expert_type == "qwen":
        qwen_cfg = config.get("qwen", {})
        weight_dir = qwen_cfg.get("weight_dir", "exported_qwen_weights/layer_0")
        intermediate_dim = qwen_cfg.get("intermediate_dim", 512)
        hidden_dim_override = qwen_cfg.get("hidden_dim", model_cfg["hidden_dim"])
        num_experts_qwen = qwen_cfg.get("num_experts", model_cfg["num_experts"])
        top_k_qwen = qwen_cfg.get("top_k", model_cfg.get("top_k", 8))

        moe = create_qwen_moe_layer(
            weight_dir=weight_dir,
            rank=rank,
            world_size=world_size,
            experts_per_rank=experts_per_rank,
            hidden_dim=hidden_dim_override,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts_qwen,
            top_k=top_k_qwen,
        )
        log(rank, f"Model: Qwen experts from {weight_dir} "
            f"dim={hidden_dim_override} intermediate={intermediate_dim} "
            f"experts={num_experts_qwen} top_k={top_k_qwen}")
    else:
        moe = MoELayer(
            hidden_dim=model_cfg["hidden_dim"],
            num_experts=model_cfg["num_experts"],
            top_k=model_cfg.get("top_k", 1),
            expert_type=expert_type,
            expert_mult=model_cfg.get("expert_hidden_mult", 4),
            routing_strategy=routing_strategy,
            experts_per_rank=experts_per_rank,
        )
        moe.set_rank(rank, world_size)

    # ── Routing-replay: replace router with captured trace ─────────
    replay_cfg = config.get("routing_replay", {})

    if routing_strategy == "replay" or replay_cfg.get("enabled", False):
        trace_path = replay_cfg.get("trace_path", "data/routing_traces/routing.json")
        trace = RoutingTrace.load(trace_path)
        cycle = replay_cfg.get("cycle_layers", False)
        layer_idx = replay_cfg.get("layer_idx", 0)

        if cycle:
            moe.router = LayerCyclingReplayRouter(
                trace,
                sim_num_experts=model_cfg["num_experts"],
                sim_top_k=model_cfg.get("top_k", 1),
            )
            log(rank, f"Router: replay (cycling {trace.meta.num_moe_layers} layers) "
                f"trace={trace_path} trace_experts={trace.meta.num_experts} "
                f"trace_top_k={trace.meta.top_k} sim_experts={model_cfg['num_experts']}")
        else:
            moe.router = ReplayRouter(
                trace,
                layer_idx=layer_idx,
                sim_num_experts=model_cfg["num_experts"],
                sim_top_k=model_cfg.get("top_k", 1),
            )
            log(rank, f"Router: replay layer={layer_idx} trace={trace_path} "
                f"trace_experts={trace.meta.num_experts} "
                f"trace_top_k={trace.meta.top_k} sim_experts={model_cfg['num_experts']}")

        # Update routing strategy string in metadata
        routing_strategy = "replay" if not cycle else "replay_cycling"

    # Broadcast model params to ensure identical gate weights across ranks
    # Skip for Qwen experts which already load identical gate from disk
    if expert_type != "qwen":
        broadcast_model_params(moe, src=0)

    # ── Synthetic data ──────────────────────────────────────────
    batch_size = data_cfg["batch_size"]
    seq_len = data_cfg["seq_len"]
    hidden_dim = model_cfg["hidden_dim"]
    num_microbatches = data_cfg["num_microbatches"]

    tokens_per_mb = batch_size * seq_len
    full_batch = torch.randn(tokens_per_mb, hidden_dim)

    # Split into micro-batches
    microbatches = torch.chunk(full_batch, num_microbatches, dim=0)

    log(rank, f"Worker ready — {num_microbatches} microbatches × {tokens_per_mb // num_microbatches} tokens each")

    # ── Run ─────────────────────────────────────────────────────
    mode = runtime_cfg.get("mode", "serial")
    num_steps = runtime_cfg.get("num_steps", 5)

    if mode.startswith("train_"):
        # ── Training mode ──
        trainer = Trainer(
            moe=moe,
            transport=transport,
            timer=timer,
            config=config,
            rank=rank,
            world_size=world_size,
            trace_dir=trace_dir,
        )
        trainer.train()

        # Save checkpoint (rank 0 only to avoid file conflicts)
        if rank == 0:
            ckpt_dir = config.get("profiling", {}).get("trace_dir", "outputs/traces")
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
            trainer.save_checkpoint(ckpt_path)
            log(rank, f"Checkpoint saved -> {ckpt_path}")

    elif mode == "serial":
        for step in range(num_steps):
            run_serial(
                step=step,
                microbatches=microbatches,
                moe=moe,
                transport=transport,
                timer=timer,
            )
    elif mode == "overlap":
        for step in range(num_steps):
            run_overlap(
                step=step,
                microbatches=microbatches,
                moe=moe,
                transport=transport,
                timer=timer,
            )
    elif mode == "ocs_pipeline":
        for step in range(num_steps):
            # Record affinity data if tracker is active
            if affinity_tracker is not None:
                for tokens in microbatches:
                    with torch.no_grad():
                        eids, gws, _ = moe.router(tokens)
                    affinity_tracker.record_routing(eids, gws)
            run_ocs_pipeline(
                step=step,
                microbatches=microbatches,
                moe=moe,
                transport=transport,
                timer=timer,
            )
    elif mode == "ocs_dbo":
        for step in range(num_steps):
            if affinity_tracker is not None:
                for tokens in microbatches:
                    with torch.no_grad():
                        eids, gws, _ = moe.router(tokens)
                    affinity_tracker.record_routing(eids, gws)
            run_ocs_dbo(
                step=step,
                microbatches=microbatches,
                moe=moe,
                transport=transport,
                timer=timer,
            )
    else:
        raise ValueError(f"Unknown runtime mode: {mode}")

    # ── Barrier and summarize ───────────────────────────────────
    transport.barrier()

    summary = timer.summary()
    total_us = sum(summary.values())
    comm_us = summary.get("comm", 0.0)
    compute_us = summary.get("compute", 0.0)

    log_summary(rank, {
        "total_steps": num_steps,
        "num_events": len(timer.events),
        "total_us": total_us,
        "comm_us": comm_us,
        "compute_us": compute_us,
        "comm_pct": (comm_us / total_us * 100) if total_us > 0 else 0,
        "mode": mode,
    })

    # -- OCS metrics (if enabled) -----------------------------------------
    if ocs_pool is not None:
        m = ocs_pool.metrics
        total_req = max(m.total_requests, 1)
        log(rank, f"OCS: {m.circuit_reuses}/{m.total_requests} reuses "
            f"({m.circuit_reuses/total_req*100:.1f}%), "
            f"{m.circuit_establishes} establishes, "
            f"{m.circuit_evictions} evictions, "
            f"{m.total_reconfig_time_us:.0f}us reconfig total")

    # -- Export trace ----------------------------------------------------
    if config.get("profiling", {}).get("export_trace", True):
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"rank_{rank:02d}_trace.json")

        # Build EP metadata for the viewer
        ep_meta = {
            "world_size": world_size,
            "num_experts": model_cfg["num_experts"],
            "experts_per_rank": experts_per_rank,
            "top_k": model_cfg.get("top_k", 1),
            "routing_strategy": routing_strategy,
            "mode": mode,
            "backend": config.get("backend", "gloo"),
            "expert_type": expert_type,
        }
        if expert_type == "qwen":
            qwen_cfg = config.get("qwen", {})
            ep_meta["qwen"] = {
                "model": "Qwen3.6-35B-A3B",
                "source_dir": qwen_cfg.get("weight_dir", ""),
                "hidden_dim": qwen_cfg.get("hidden_dim", 0),
                "intermediate_dim": qwen_cfg.get("intermediate_dim", 0),
                "experts_exported": qwen_cfg.get("num_experts", 0),
                "top_k": qwen_cfg.get("top_k", 0),
            }
            if config.get("routing_replay", {}).get("enabled"):
                ep_meta["qwen"]["routing_source"] = config.get("routing_replay", {}).get("trace_path", "")
        # Add topology info if available
        if topology is not None:
            topo_cfg = config.get("topology", {})
            ep_meta["topology"] = {
                "num_pods": topo_cfg.get("num_pods", 1),
                "nodes_per_pod": topo_cfg.get("nodes_per_pod", 1),
                "ranks_per_node": topo_cfg.get("ranks_per_node", world_size),
            }
            # Per-rank location for this rank
            loc = topology.get_location(rank)
            ep_meta["rank_location"] = {
                "pod_id": loc.pod_id,
                "node_id": loc.node_id,
                "local_rank": loc.local_rank,
            }

        # Add OCS info if available
        if ocs_topology is not None and ocs_pool is not None:
            ocs_metrics = transport.get_ocs_metrics()
            ep_meta["ocs"] = {
                "enabled": True,
                "max_circuits": ocs_cfg.get("max_circuits", 32),
                "reconfig_time_us": ocs_cfg.get("reconfig_time_us", 50.0),
                "circuit_latency_us": ocs_cfg.get("circuit_latency_us", 1.0),
                "circuit_bandwidth_gbps": ocs_cfg.get("circuit_bandwidth_gbps", 200.0),
                "metrics": ocs_metrics,
            }

        export_chrome_trace(timer.events, trace_path, pid=rank, tid=0, metadata=ep_meta)
        log(rank, f"Trace exported -> {trace_path}")

    cleanup_process_group()
