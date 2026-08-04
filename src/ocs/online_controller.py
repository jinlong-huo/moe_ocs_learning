"""Online affinity controller: adaptive OCS circuit management during inference.

Instead of a separate training phase to pre-compute affinity, this controller
tracks expert co-activation *during* inference and continuously adjusts the
OCS circuit pool to prioritize high-affinity rank pairs.

Key properties:
  - Records routing decisions as they happen (same as ExpertAffinityTracker)
  - Periodically recomputes the circuit plan from accumulated affinity
  - Incrementally adjusts circuits: pre-establishes high-affinity pairs,
    lets LRU naturally evict unused low-affinity ones
  - Optional exponential decay so the system adapts to shifting patterns
  - Per-rank: each process maintains its own controller (no cross-rank sync needed
    since the router is replicated across ranks)

Usage in scheduler:
    controller = OnlineAffinityController(
        affinity_tracker, ocs_pool,
        experts_per_rank=2, world_size=4, max_circuits=16,
    )
    for step in range(num_steps):
        for tokens in microbatches:
            eids, gws, _ = moe.router(tokens)
            # Record and get pre-establish targets for THIS batch
            targets = controller.step(eids, gws)
            transport.pre_establish_circuits(targets)
            # ... scatter/compute/gather pipeline ...
"""
from __future__ import annotations

import time
from typing import List, Optional

import torch

from src.ocs.circuit import OcsCircuitPool
from src.ocs.placement import ExpertAffinityTracker


def _target_ranks_from_experts(
    expert_ids: torch.Tensor, experts_per_rank: int,
) -> list:
    """Derive target ranks for communication from expert assignments.

    Each expert_id maps to a target rank via: rank = expert_id // experts_per_rank.
    Returns a deduplicated list of rank IDs this micro-batch communicates with.
    """
    ids = expert_ids.reshape(-1)
    return list(set((ids // experts_per_rank).tolist()))


class OnlineAffinityController:
    """Tracks expert co-activation online and adjusts OCS circuits adaptively.

    Composes an ExpertAffinityTracker (for accumulation) and an OcsCircuitPool
    (for circuit management). Called from the scheduler each step to feed
    routing decisions and periodically recompute circuit plans.

    Parameters
    ----------
    affinity_tracker : ExpertAffinityTracker
        Pre-constructed tracker (shares the same num_experts as the model).
    circuit_pool : OcsCircuitPool
        The per-rank OCS circuit pool to manage.
    experts_per_rank : int
        Experts per GPU rank (for expert_id → rank mapping).
    world_size : int
        Number of GPU ranks.
    max_circuits : int
        Maximum circuits in the plan (top-N affinity pairs).
    update_interval_steps : int
        Recompute the circuit plan every N steps. Default 1 (every step).
        Higher values amortize plan computation cost.
    decay_factor : float
        Exponential decay applied to co-activation counts each step.
        1.0 = no decay (all history equally weighted).
        0.99 = each step, old counts are multiplied by 0.99.
        Lower values make the system more responsive to pattern shifts.
    """

    def __init__(
        self,
        affinity_tracker: ExpertAffinityTracker,
        circuit_pool: OcsCircuitPool,
        experts_per_rank: int,
        world_size: int,
        max_circuits: int = 16,
        update_interval_steps: int = 1,
        decay_factor: float = 1.0,
        rank: int = 0,
    ):
        self.tracker = affinity_tracker
        self.pool = circuit_pool
        self.experts_per_rank = experts_per_rank
        self.world_size = world_size
        self.max_circuits = max_circuits
        self.update_interval_steps = update_interval_steps
        self.decay_factor = decay_factor
        self.rank = rank

        self._step_count = 0
        self._total_circuits_adjusted = 0
        self._plan_cache: Optional[List[tuple]] = None

    # ── Core API (called from scheduler) ────────────────────────────────

    def step(
        self,
        expert_ids: torch.Tensor,
        gate_weights: torch.Tensor,
    ) -> list:
        """Record one routing decision and return pre-establish targets.

        Call this once per micro-batch from the scheduler. The returned
        target_ranks list should be passed to transport.pre_establish_circuits()
        before firing scatter.

        Args:
            expert_ids: [T] or [T, K] expert assignments from router.
            gate_weights: [T, K] gate weights from router.

        Returns:
            List of target rank IDs to pre-establish circuits for.
        """
        # Accumulate affinity
        self.tracker.record_routing(expert_ids, gate_weights)

        # Increment step counter and maybe trigger plan recompute
        self._step_count += 1
        if self._step_count % self.update_interval_steps == 0:
            self.adjust_circuits()
            # Apply decay after adjustment (so the plan sees full counts)
            if self.decay_factor < 1.0:
                self.apply_decay()

        # Return immediate targets for THIS micro-batch (same as ocs_pipeline)
        return _target_ranks_from_experts(expert_ids, self.experts_per_rank)

    def record(self, expert_ids: torch.Tensor, gate_weights: torch.Tensor) -> None:
        """Feed one routing decision into the tracker (without returning targets).

        Use this for bulk pre-recording before the pipeline, e.g., during
        the pre-route phase or when accumulating from external sources.
        """
        self.tracker.record_routing(expert_ids, gate_weights)

    # ── Circuit adjustment ──────────────────────────────────────────────

    def adjust_circuits(self, current_time_ns: Optional[int] = None) -> int:
        """Recompute circuit plan from accumulated affinity and adjust pool.

        Computes the top-N rank pairs by affinity score, then pre-establishes
        any that are not yet in the pool. Low-affinity circuits are NOT
        explicitly evicted — the pool's LRU policy naturally evicts them
        when higher-affinity circuits are established.

        Args:
            current_time_ns: timestamp for circuit establishment tracking.
                Uses time.perf_counter_ns() if None.

        Returns:
            Number of circuits newly established during this adjustment.
        """
        if current_time_ns is None:
            current_time_ns = time.perf_counter_ns()

        plan = self.compute_plan()
        established = 0

        for src, dst, score in plan:
            # Only establish circuits originating from this rank.
            # The plan includes all rank pairs; each rank only needs its own
            # outgoing circuits (transport queries (self.rank, dst) only).
            if src != self.rank:
                continue
            if self.pool.is_established(src, dst):
                continue
            self.pool.establish(src, dst, current_time_ns)
            established += 1

        self._total_circuits_adjusted += established
        return established

    def compute_plan(self) -> List[tuple]:
        """Compute the current circuit plan from accumulated affinity.

        Returns:
            List of (src_rank, dst_rank, score) sorted by score descending,
            capped at max_circuits entries.
        """
        plan = self.tracker.compute_circuit_plan(
            experts_per_rank=self.experts_per_rank,
            world_size=self.world_size,
            max_circuits=self.max_circuits,
        )
        self._plan_cache = plan
        return plan

    # ── Decay ───────────────────────────────────────────────────────────

    def apply_decay(self) -> None:
        """Apply exponential decay to accumulated co-activation counts.

        Multiplies co_activation_counts and expert_usage by decay_factor.
        This allows old patterns to fade, making the system responsive to
        distribution shifts during long generation runs.
        """
        self.tracker.co_activation_counts *= self.decay_factor
        self.tracker.expert_usage *= self.decay_factor

    # ── Query / metrics ─────────────────────────────────────────────────

    def advance_step(self) -> None:
        """Advance the internal step counter by one.

        Used by schedulers that manage the controller's lifecycle externally
        (e.g., pre-route-all then batch-adjust, rather than calling step()
        per micro-batch).
        """
        self._step_count += 1

    @property
    def step_count(self) -> int:
        """Number of steps elapsed since controller creation."""
        return self._step_count

    @property
    def current_plan(self) -> Optional[List[tuple]]:
        """The most recently computed circuit plan (or None if never computed)."""
        return self._plan_cache

    @property
    def total_circuits_adjusted(self) -> int:
        """Cumulative count of circuit adjustments performed."""
        return self._total_circuits_adjusted

    def get_affinity_matrix(self) -> torch.Tensor:
        """Return the current normalized co-activation matrix [E, E]."""
        return self.tracker.get_affinity_scores()

    def summary(self) -> dict:
        """Return a summary dict for logging and trace export."""
        pool_metrics = self.pool.metrics
        total_req = max(pool_metrics.total_requests, 1)
        return {
            "tracker_samples": self.tracker.total_samples,
            "total_circuits_adjusted": self._total_circuits_adjusted,
            "steps_elapsed": self._step_count,
            "pool_active_circuits": self.pool.active_circuit_count,
            "pool_reuse_ratio": self.pool.reuse_ratio,
            "pool_establishes": pool_metrics.circuit_establishes,
            "pool_evictions": pool_metrics.circuit_evictions,
            "pool_reconfig_total_us": pool_metrics.total_reconfig_time_us,
            "pool_transfer_total_us": pool_metrics.total_transfer_time_us,
            "current_plan_size": len(self._plan_cache) if self._plan_cache else 0,
        }
