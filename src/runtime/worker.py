"""Per-rank worker: the main execution loop for one process.

Each worker:
  1. Initializes its process group
  2. Builds the MoE layer (owning one expert)
  3. Generates input activations (routing comes from the real gate / captured trace)
  4. Runs the scheduler (serial or overlap mode)
  5. Records timeline events
  6. Exports trace and metrics
"""
from __future__ import annotations

import os
from typing import Dict

import torch

from src.runtime.process_group import init_process_group, cleanup_process_group
from src.runtime.scheduler import (
    run_serial, run_overlap,
    run_ocs_pipeline, run_ocs_dbo, run_ocs_preset,
    run_ocs_online,
)
from src.ocs.online_controller import OnlineAffinityController
from src.model.router_replay import ReplayRouter, LayerCyclingReplayRouter
from src.model.qwen_experts import create_qwen_moe_layer
from src.data.routing_schema import RoutingTrace
from src.comm.transport import Transport
from src.comm.topology import Topology, TopologyConfig
from src.comm.path_resolver import PathResolver
from src.ocs.topology import OcsTopology, OcsTopologyConfig
from src.ocs.placement import ExpertAffinityTracker
from src.utils.timer import Timer
from src.utils.logging import log, log_summary
from src.utils.seed import set_seed
from src.comm.timeline import export_chrome_trace

from src.runtime.placement import Placement


def _resolve_placement(config, num_experts, experts_per_rank, world_size):
    """Build the independent expert->rank / rank->physical Placement from config.

    Strategies:
      - linear      (default): contiguous e // k mapping (historical behavior)
      - shuffle     : seeded random uniform permutation
      - affinity    : greedy co-activation clustering from a routing trace
      - permutation : explicit per-rank expert lists (placement.rank_experts)
    """
    pc = config.get("placement", {}) or {}
    strategy = pc.get("strategy", "linear")

    rank_locations = None
    if pc.get("rank_locations"):
        rank_locations = [tuple(t) for t in pc["rank_locations"]]

    if strategy == "linear":
        return Placement.linear(
            num_experts, experts_per_rank, world_size,
            rank_to_location=rank_locations,
        )
    if strategy == "shuffle":
        return Placement.shuffled(
            num_experts, experts_per_rank, world_size,
            seed=pc.get("seed", 0),
            rank_to_location=rank_locations,
        )
    if strategy == "affinity":
        from src.ocs.preconfig import _build_affinity_from_trace
        trace_path = pc.get("trace_path", "data/routing_traces/routing.json")
        trace = RoutingTrace.load(trace_path)
        tracker = _build_affinity_from_trace(trace, num_experts)
        rank_experts = tracker.suggest_placement(experts_per_rank, world_size)
        return Placement.from_permutation(
            rank_experts, experts_per_rank, world_size,
            rank_to_location=rank_locations,
        )
    if strategy == "permutation":
        rank_experts = pc.get("rank_experts")
        if rank_experts is None:
            raise ValueError(
                "placement.strategy='permutation' requires placement.rank_experts"
            )
        return Placement.from_permutation(
            rank_experts, experts_per_rank, world_size,
            rank_to_location=rank_locations,
        )
    raise ValueError(f"Unknown placement.strategy: {strategy!r}")


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

    # -- Placement: independent expert->rank / rank->physical table -----
    # Resolved once, up front: it feeds both the dispatch (expert -> rank)
    # and the topology delay model (rank -> physical location).
    experts_per_rank = model_cfg.get("experts_per_rank", 1)
    # Effective expert count (the qwen block may override model.num_experts)
    num_experts = config.get("qwen", {}).get("num_experts", model_cfg["num_experts"])
    placement = _resolve_placement(config, num_experts, experts_per_rank, world_size)

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
            intra_pod_bandwidth_gbps=topo_cfg.get("intra_pod_bandwidth_gbps", 400.0),
            cross_pod_bandwidth_gbps=topo_cfg.get("cross_pod_bandwidth_gbps", 200.0),
            delay_multiplier=topo_cfg.get("delay_multiplier", 1.0),
            # Rank -> physical location comes from the SAME placement object
            # that owns expert -> rank (single source of truth).
            rank_locations=(
                {i: tuple(t) for i, t in enumerate(placement.rank_to_location)}
                if placement.rank_to_location else None
            ),
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
        cost_model = ocs_cfg.get("cost_model", "lru")
        if cost_model == "fixed_delay":
            # None = full fan-out (world_size-1 circuits); explicit value =
            # per-rank circuit budget (ports/wavelengths).
            max_circuits = ocs_cfg.get("max_circuits")
        else:
            max_circuits = ocs_cfg.get("max_circuits", 32)
        ocs_topology = OcsTopology(
            OcsTopologyConfig(
                enabled=True,
                cost_model=cost_model,
                max_circuits=max_circuits,
                reconfig_time_us=ocs_cfg.get("reconfig_time_us", 50.0),
                circuit_latency_us=ocs_cfg.get("circuit_latency_us", 1.0),
                circuit_bandwidth_gbps=ocs_cfg.get("circuit_bandwidth_gbps", 200.0),
                placement_strategy=ocs_cfg.get("placement_strategy", "round_robin"),
            ),
            # The fixed_delay model layers the fixed reconfig delay on top of
            # the SAME tier-aware EPS cost the electrical baseline pays.
            eps_topology=topology,
            flat_delay_us=delay_cfg.get("comm_delay_us", 0.0),
            world_size=world_size,
        )
        ocs_pool = ocs_topology.pool
        if cost_model == "fixed_delay":
            budget = ocs_pool.max_circuits
            log(rank, f"OCS fixed_delay: EPS tier cost + "
                f"{ocs_cfg.get('reconfig_time_us', 50.0)}us per circuit switch, "
                f"circuit budget={budget} (eps_topology={'on' if topology is not None else 'off'})")
        else:
            log(rank, f"OCS LRU: {ocs_cfg['max_circuits']} max circuits, "
                f"{ocs_cfg['reconfig_time_us']}us reconfig, "
                f"{ocs_cfg['circuit_bandwidth_gbps']}Gbps BW")

        # Build affinity tracker if using affinity placement
        if ocs_cfg.get("placement_strategy") == "affinity":
            affinity_tracker = ExpertAffinityTracker(num_experts)
            log(rank, "OCS affinity placement: tracking expert co-activation")

    # -- Build PathResolver for mixed EPS+OCS transport (NEW) ------------------
    path_resolver = None
    mixed_cfg = ocs_cfg.get("mixed_transport", {})
    mixed_enabled = (
        isinstance(mixed_cfg, dict) and mixed_cfg.get("enabled", False)
    ) or (isinstance(mixed_cfg, bool) and mixed_cfg)
    if mixed_enabled and ocs_pool is not None:
        # EPS path must exist: either topology or flat delay
        has_eps = topology is not None or delay_cfg.get("comm_delay_us", 0) > 0
        if has_eps:
            path_resolver = PathResolver(
                circuit_pool=ocs_pool,
                topology=topology,
                plan=None,  # populated later from preset plan or online controller
                flat_delay_us=delay_cfg.get("comm_delay_us", 0.0),
                flat_jitter_us=delay_cfg.get("comm_delay_jitter_us", 0.0),
            )
            log(rank, f"Mixed transport: OCS + "
                f"{'topology' if topology is not None else 'flat'} EPS fallback, "
                f"{path_resolver.plan_size} plan pairs")
        else:
            log(rank, "WARNING: mixed_transport enabled but no EPS path "
                "(topology or flat delay) — falling back to OCS-only")

    transport = Transport(
        timer=timer,
        comm_delay_us=delay_cfg.get("comm_delay_us", 0.0),
        comm_delay_jitter_us=delay_cfg.get("comm_delay_jitter_us", 0.0),
        topology=topology,
        rank=rank,
        world_size=world_size,
        ocs_circuit_pool=ocs_pool,
        path_resolver=path_resolver,
    )

    routing_strategy = config.get("routing", {}).get("strategy", "fixed")

    # ── Build model: real Qwen MoE experts + gate ────────────────
    qwen_cfg = config.get("qwen", {})
    weight_dir = qwen_cfg.get("weight_dir", "exported_qwen_weights/layer_0")
    intermediate_dim = qwen_cfg.get("intermediate_dim", 512)
    hidden_dim_override = qwen_cfg.get("hidden_dim", model_cfg["hidden_dim"])
    num_experts_qwen = num_experts  # already resolved from the qwen override
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
        placement=placement,
    )
    log(rank, f"Model: Qwen experts from {weight_dir} "
        f"dim={hidden_dim_override} intermediate={intermediate_dim} "
        f"experts={num_experts_qwen} top_k={top_k_qwen}")

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
                sim_num_experts=num_experts,
                sim_top_k=top_k_qwen,
            )
            log(rank, f"Router: replay (cycling {trace.meta.num_moe_layers} layers) "
                f"trace={trace_path} trace_experts={trace.meta.num_experts} "
                f"trace_top_k={trace.meta.top_k} sim_experts={num_experts}")
        else:
            moe.router = ReplayRouter(
                trace,
                layer_idx=layer_idx,
                sim_num_experts=num_experts,
                sim_top_k=top_k_qwen,
            )
            log(rank, f"Router: replay layer={layer_idx} trace={trace_path} "
                f"trace_experts={trace.meta.num_experts} "
                f"trace_top_k={trace.meta.top_k} sim_experts={num_experts}")

        # Update routing strategy string in metadata
        routing_strategy = "replay" if not cycle else "replay_cycling"

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

    if mode == "serial":
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
    elif mode == "ocs_preset":
        # Pre-load circuits from the trace-derived placement plan
        preset_cfg = ocs_cfg.get("preset", {})
        preset_source = preset_cfg.get("source", "trace")
        if preset_source == "plan":
            plan_path = preset_cfg.get("plan_path", "")
            if plan_path and ocs_pool is not None:
                from src.ocs.preconfig import load_plan
                plan = load_plan(plan_path)
                if hasattr(ocs_pool, "pre_config"):
                    n = ocs_pool.pre_config(plan)
                    log(rank, f"OCS preset: loaded {n} circuits from plan {plan_path}")
                if path_resolver is not None:
                    path_resolver.set_plan_from_list(plan)
                    log(rank, f"Mixed transport: {len(plan)} plan pairs → PathResolver")
        elif preset_source == "trace" and affinity_tracker is not None:
            # Build plan from the affinity tracker (populated from routing)
            for tokens in microbatches:
                with torch.no_grad():
                    eids, gws, _ = moe.router(tokens)
                affinity_tracker.record_routing(eids, gws)
            plan = affinity_tracker.compute_circuit_plan(
                expert_to_rank=placement.expert_to_rank_dict(),
                experts_per_rank=experts_per_rank,
                world_size=world_size,
                max_circuits=ocs_cfg.get("max_circuits", 16),
            )
            if hasattr(ocs_pool, "pre_config"):
                n = ocs_pool.pre_config(plan)
                log(rank, f"OCS preset: pre-established {n} circuits from affinity")
            if path_resolver is not None:
                path_resolver.set_plan_from_list(plan)
                log(rank, f"Mixed transport: {len(plan)} plan pairs → PathResolver")
        for step in range(num_steps):
            run_ocs_preset(
                step=step,
                microbatches=microbatches,
                moe=moe,
                transport=transport,
                timer=timer,
            )
    elif mode == "ocs_online":
        # Online affinity-driven OCS: track routing during inference,
        # continuously adjust circuits to prioritize high-affinity rank pairs.
        # No separate training phase needed — the system learns from its own
        # inference-time routing patterns.
        online_cfg = ocs_cfg.get("online", {})
        update_interval = online_cfg.get("update_interval_steps", 1)
        decay_factor = online_cfg.get("decay_factor", 1.0)

        if affinity_tracker is None:
            affinity_tracker = ExpertAffinityTracker(num_experts)

        controller = OnlineAffinityController(
            affinity_tracker=affinity_tracker,
            circuit_pool=ocs_pool,
            experts_per_rank=experts_per_rank,
            world_size=world_size,
            max_circuits=ocs_cfg.get("max_circuits", 16),
            update_interval_steps=update_interval,
            decay_factor=decay_factor,
            rank=rank,
            placement=placement,
        )
        log(rank, f"OCS online: adaptive affinity, "
            f"update_interval={update_interval}, decay={decay_factor}")

        for step in range(num_steps):
            run_ocs_online(
                step=step,
                microbatches=microbatches,
                moe=moe,
                transport=transport,
                timer=timer,
                controller=controller,
            )
            # Sync PathResolver plan from controller's dynamically adjusted plan
            if path_resolver is not None and controller.current_plan is not None:
                path_resolver.set_plan_from_list(controller.current_plan)

        # Log online controller summary
        ctrl_summary = controller.summary()
        log(rank, f"OCS online summary: {ctrl_summary}")
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

    # -- Mixed transport metrics (if enabled) -------------------------------
    if path_resolver is not None:
        pm = path_resolver.get_metrics()
        log(rank, f"Mixed transport: {pm['ocs_requests']}/{pm['total_requests']} "
            f"OCS ({pm['ocs_fraction']*100:.1f}%), "
            f"OCS avg {pm['ocs_avg_delay_us']:.1f}us, "
            f"EPS avg {pm['eps_avg_delay_us']:.1f}us, "
            f"plan={pm['plan_size']} pairs")

    # -- Export trace ----------------------------------------------------
    if config.get("profiling", {}).get("export_trace", True):
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"rank_{rank:02d}_trace.json")

        # Build EP metadata for the viewer
        ep_meta = {
            "world_size": world_size,
            "num_experts": num_experts,
            "experts_per_rank": experts_per_rank,
            "top_k": top_k_qwen,
            "routing_strategy": routing_strategy,
            "mode": mode,
            "backend": config.get("backend", "gloo"),
            "expert_type": "qwen",
            "qwen": {
                "model": "Qwen3.6-35B-A3B",
                "source_dir": qwen_cfg.get("weight_dir", ""),
                "hidden_dim": qwen_cfg.get("hidden_dim", 0),
                "intermediate_dim": qwen_cfg.get("intermediate_dim", 0),
                "experts_exported": qwen_cfg.get("num_experts", 0),
                "top_k": qwen_cfg.get("top_k", 0),
            },
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
                "cost_model": ocs_cfg.get("cost_model", "lru"),
                "max_circuits": ocs_cfg.get("max_circuits", 32),
                "reconfig_time_us": ocs_cfg.get("reconfig_time_us", 50.0),
                "circuit_latency_us": ocs_cfg.get("circuit_latency_us", 1.0),
                "circuit_bandwidth_gbps": ocs_cfg.get("circuit_bandwidth_gbps", 200.0),
                "metrics": ocs_metrics,
            }

        # Add mixed transport info if available
        if path_resolver is not None:
            ep_meta["mixed_transport"] = {
                "enabled": True,
                "metrics": path_resolver.get_metrics(),
            }

        export_chrome_trace(timer.events, trace_path, pid=rank, tid=0, metadata=ep_meta)
        log(rank, f"Trace exported -> {trace_path}")

    cleanup_process_group()
