"""Abstract transport interface wrapping torch.distributed.

All communication passes through this module so we can:
  - swap backends (Gloo / NCCL) without touching model code
  - inject synthetic delay for network simulation
  - instrument every collective with timeline events
  - model hierarchical topology (NVLink / IB / cross-pod) delays
"""

from __future__ import annotations

import random
import time
from typing import Optional

import torch
import torch.distributed as dist

from src.utils.timer import Timer


class Transport:
    """Wraps torch.distributed collective ops with optional delay injection.

    Supports two delay modes:
      1. Flat delay: comm_delay_us +- jitter applied uniformly to all collectives
      2. Topology-aware delay: hierarchical network model with per-tier latency
         and bandwidth (requires a Topology instance)

    When topology is set, the flat delay parameters are ignored in favor
    of topology-computed delays.
    """

    def __init__(
        self,
        timer: Optional[Timer] = None,
        comm_delay_us: float = 0.0,
        comm_delay_jitter_us: float = 0.0,
        topology=None,   # Topology instance (optional, avoids circular import)
        rank: int = 0,
        world_size: Optional[int] = None,
        ocs_circuit_pool=None,  # FixedDelayCircuitPool instance (optional, None disables OCS)
        path_resolver=None,  # PathResolver for mixed EPS+OCS (optional)
    ):
        self.timer = timer
        self.comm_delay_us = comm_delay_us
        self.comm_delay_jitter_us = comm_delay_jitter_us
        self.topology = topology
        self.rank = rank
        self._world_size = world_size
        self.ocs_circuit_pool = ocs_circuit_pool
        self.path_resolver = path_resolver

    def set_world_size(self, world_size: int) -> None:
        """Set the world size (call after init_process_group if not passed)."""
        self._world_size = world_size

    # -- delay injection --------------------------------------------------

    def _inject_delay(self, tensor_bytes: int = 0, target_ranks: Optional[list] = None) -> None:
        """Inject synthetic communication delay.

        Four delay modes (checked in order of priority):
          1. Mixed EPS+OCS: uses PathResolver for per-rank-pair path decisions.
             Pairs in the OCS plan use optical circuits; others fallback to EPS
             (topology or flat). Requires path_resolver + target_ranks.
          2. OCS-only: uses the fixed-delay circuit pool for all pairs. Requires ocs_circuit_pool
             + target_ranks. (backward compatible)
          3. Topology-aware: uses hierarchical network model (NVLink/IB/cross-pod).
          4. Flat delay: simple fixed delay + jitter.

        Args:
            tensor_bytes: total bytes in the tensor (for bandwidth modeling)
            target_ranks: list of destination rank IDs (required for OCS and
                          mixed modes, ignored otherwise)
        """
        # --- Mixed EPS+OCS transport (NEW, highest priority) ---
        if self.path_resolver is not None and target_ranks is not None:
            current_ns = time.perf_counter_ns()
            max_delay_us = 0.0
            for dst in target_ranks:
                if dst == self.rank:
                    continue
                delay_us = self.path_resolver.compute_delay(
                    self.rank, dst, tensor_bytes, current_ns,
                )
                if delay_us > max_delay_us:
                    max_delay_us = delay_us
            if max_delay_us > 0:
                time.sleep(max_delay_us / 1_000_000.0)
            return

        # --- OCS circuit-aware delay (backward compatible) ---
        if self.ocs_circuit_pool is not None and target_ranks is not None:
            current_ns = time.perf_counter_ns()
            max_delay_us = 0.0
            for dst in target_ranks:
                if dst == self.rank:
                    continue
                delay_us = self.ocs_circuit_pool.compute_delay(
                    self.rank, dst, tensor_bytes, current_ns,
                )
                if delay_us > max_delay_us:
                    max_delay_us = delay_us
            if max_delay_us > 0:
                time.sleep(max_delay_us / 1_000_000.0)
            return

        # --- Existing topology-aware delay ---
        if self.topology is not None and self._world_size is not None:
            # Topology-aware delay
            total = self.topology.get_delay(self.rank, self._world_size, tensor_bytes)
            if total > 0:
                time.sleep(total / 1_000_000.0)
            return

        # --- Existing flat delay mode (backward compatible) ---
        if self.comm_delay_us <= 0 and self.comm_delay_jitter_us <= 0:
            return
        jitter = random.uniform(-self.comm_delay_jitter_us, self.comm_delay_jitter_us)
        total = max(0.0, self.comm_delay_us + jitter)
        if total > 0:
            time.sleep(total / 1_000_000.0)

    # -- collective ops ----------------------------------------------------

    def all_to_all(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor,
        async_op: bool = False, active_ranks: Optional[list] = None,
    ):
        """All-to-all collective.  Optionally async for overlap mode.

        Uses all_to_all_single which splits a single tensor evenly across
        ranks along dim 0 -- the natural fit for MoE dispatch where
        each rank handles one or more experts.

        Args:
            active_ranks: optional list of ranks that have non-zero traffic.
                When provided, delay is only injected for these ranks (not all).
                When None (default), delay is injected for all ranks (backward compat).
                This enables per-rank-pair metrics that reflect actual traffic patterns.
        """
        if self.timer:
            self.timer.start("comm/all_to_all", async_op=async_op)

        # Compute tensor bytes for bandwidth-aware delay
        tensor_bytes = input_tensor.numel() * input_tensor.element_size()
        # Use active_ranks if provided, otherwise all ranks (backward compatible)
        if active_ranks is not None:
            target_ranks = active_ranks
        else:
            target_ranks = list(range(dist.get_world_size())) if dist.is_initialized() else []
        self._inject_delay(tensor_bytes=tensor_bytes, target_ranks=target_ranks)

        handle = dist.all_to_all_single(output_tensor, input_tensor, async_op=async_op)
        if self.timer and not async_op:
            self.timer.stop("comm/all_to_all")
        return handle

    def all_gather(self, tensor: torch.Tensor, async_op: bool = False):
        """Gather tensors from all ranks into a list."""
        world_size = dist.get_world_size()
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
        if self.timer:
            self.timer.start("comm/all_gather", async_op=async_op)

        tensor_bytes = tensor.numel() * tensor.element_size()
        self._inject_delay(tensor_bytes=tensor_bytes)

        handle = dist.all_gather(gather_list, tensor, async_op=async_op)
        if self.timer and not async_op:
            self.timer.stop("comm/all_gather")
        return gather_list, handle

    def barrier(self) -> None:
        if self.timer:
            self.timer.start("comm/barrier")
        dist.barrier()
        if self.timer:
            self.timer.stop("comm/barrier")

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> None:
        if self.timer:
            self.timer.start("comm/broadcast", src=src)
        dist.broadcast(tensor, src=src)
        if self.timer:
            self.timer.stop("comm/broadcast")

    def wait(self, handle) -> None:
        """Wait on an async handle and record completion."""
        handle.wait()
        # Timer stop happens at the call-site so caller controls the label

    # -- OCS circuit pre-establishment -----------------------------------

    def pre_establish_circuits(self, target_ranks: list) -> float:
        """Proactively establish OCS circuits to target ranks.

        Called by the scheduler before firing scatter for a micro-batch.
        Circuits are established synchronously here: the reconfiguration
        delay is *actually paid* (slept) so the OCS cost model stays honest
        — in the fixed_delay model this is the field-standard
        T_ocs = T_eps + T_reconfig × N_switches accounting. In ocs_dbo mode
        the lookahead effectively hides this cost behind the previous
        batch's compute.

        Returns total reconfiguration time incurred (microseconds).
        Returns 0.0 if OCS is disabled or all circuits were already hot.

        Circuit-budget aware: pre-warming fills the available per-rank
        circuit budget only. When the switch is port-limited (e.g., a
        single-port MEMS), targets beyond the budget reconfigure at
        transfer time instead of being churned here and again at use.
        """
        if self.ocs_circuit_pool is None:
            return 0.0

        total_reconfig = 0.0
        current_ns = time.perf_counter_ns()
        for dst in target_ranks:
            if dst == self.rank:
                continue
            if self.ocs_circuit_pool.is_established(self.rank, dst):
                continue
            # Port-limited switch: keep the remaining budget for the transfer
            # phase rather than churning circuits that cannot all stay up.
            if self.ocs_circuit_pool.active_circuit_count >= self.ocs_circuit_pool.max_circuits:
                break
            reconfig = self.ocs_circuit_pool.establish(self.rank, dst, current_ns)
            total_reconfig += reconfig
        if total_reconfig > 0:
            time.sleep(total_reconfig / 1_000_000.0)
        return total_reconfig

    def get_ocs_metrics(self) -> dict:
        """Return OCS circuit pool metrics for trace export.

        Returns empty dict when OCS is disabled.
        """
        if self.ocs_circuit_pool is None:
            return {}
        m = self.ocs_circuit_pool.metrics
        return {
            "total_requests": m.total_requests,
            "circuit_reuses": m.circuit_reuses,
            "circuit_establishes": m.circuit_establishes,
            "circuit_evictions": m.circuit_evictions,
            "total_reconfig_time_us": m.total_reconfig_time_us,
            "total_transfer_time_us": m.total_transfer_time_us,
            "reuse_ratio": self.ocs_circuit_pool.reuse_ratio,
            "active_circuits": self.ocs_circuit_pool.active_circuit_count,
            "max_circuits": self.ocs_circuit_pool.max_circuits,
        }
