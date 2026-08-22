"""Per-rank-pair transport path resolution: OCS vs EPS.

In the current codebase, Transport._inject_delay() uses a strict priority:
  OCS pool → topology → flat delay
When OCS is enabled, ALL rank pairs go through OCS — there is no EPS fallback.

This module introduces the PathResolver, which decides per-rank-pair whether
to use OCS (optical fast path) or EPS (electrical packet switching fallback).
It is the architectural linchpin for "mixed transport" — where high-affinity
rank pairs use pre-established OCS circuits and the rest fall back to the
existing electrical fabric.

Design
------
PathResolver holds:
  - circuit_pool (FixedDelayCircuitPool): for OCS delay computation
  - topology (Topology | None): for per-pair EPS delay
  - flat_delay_us / flat_jitter_us: fallback when no topology
  - plan (set of (src, dst)): which rank pairs should use OCS

The plan is the "network input" — it tells the transport which paths
to provision with OCS. Pairs not in the plan use EPS.

Resolution rule:
  if (src, dst) in plan → OCS (circuit_pool.compute_delay)
  else                  → EPS (topology.get_pairwise_delay or flat)

When a pair is OCS, the EPS path for that pair is effectively "dark" —
the EPS fabric bandwidth is not consumed by that traffic. This is modeled
implicitly: each EPS pair computes its own delay independently, and only
non-OCS pairs contribute to EPS congestion.

Integration
-----------
Created in worker.py when both OCS pool and an EPS path (topology or flat)
are active, then passed to Transport. Transport._inject_delay() checks for
path_resolver first, before falling through to the existing three-tier logic.

Backward compatibility
----------------------
When path_resolver is None, Transport uses the existing three-tier priority.
All existing configs and modes continue to work unchanged.
"""
from __future__ import annotations

import random
import time
from typing import Optional, Set, Tuple


class PathResolution:
    """Enum-like constants for path resolution results."""
    OCS: str = "ocs"
    EPS: str = "eps"


class PathResolver:
    """Resolves transport path (OCS or EPS) per rank pair.

    Created once per rank in worker.py and passed to Transport.
    Thread safety is not required — used sequentially within a single process.

    Parameters
    ----------
    circuit_pool : FixedDelayCircuitPool
        The per-rank OCS circuit pool. Must not be None.
    topology : Topology | None
        Hierarchical topology model for per-pair EPS delays.
    plan : set of (src, dst) pairs
        Rank pairs that should use OCS. Pairs not in this set use EPS.
    flat_delay_us : float
        Flat delay for EPS when topology is None.
    flat_jitter_us : float
        Uniform jitter for EPS when topology is None.
    """

    def __init__(
        self,
        circuit_pool,  # FixedDelayCircuitPool
        topology=None,  # Topology | None
        plan: Optional[Set[Tuple[int, int]]] = None,
        flat_delay_us: float = 0.0,
        flat_jitter_us: float = 0.0,
    ):
        if circuit_pool is None:
            raise ValueError("circuit_pool must not be None for PathResolver")
        self.circuit_pool = circuit_pool
        self.topology = topology
        self.plan: Set[Tuple[int, int]] = plan or set()
        self.flat_delay_us = flat_delay_us
        self.flat_jitter_us = flat_jitter_us

        # Per-path metrics
        self.ocs_requests: int = 0
        self.eps_requests: int = 0
        self.ocs_total_delay_us: float = 0.0
        self.eps_total_delay_us: float = 0.0

    # -- Path resolution ---------------------------------------------------

    def resolve(self, src: int, dst: int) -> str:
        """Determine which transport path to use for src → dst.

        Returns PathResolution.OCS or PathResolution.EPS.
        """
        if (src, dst) in self.plan:
            return PathResolution.OCS
        return PathResolution.EPS

    def has_path(self, src: int, dst: int) -> bool:
        """Check whether (src, dst) pair is in the OCS plan."""
        return (src, dst) in self.plan

    # -- Delay computation ------------------------------------------------

    def compute_delay(
        self,
        src: int,
        dst: int,
        tensor_bytes: int,
        current_time_ns: int = 0,
    ) -> float:
        """Compute transport delay for src → dst using the resolved path.

        For OCS pairs, delegates to circuit_pool.compute_delay() which
        auto-establishes the circuit and returns reconfig + transfer delay.

        For EPS pairs, uses topology.get_pairwise_delay() if available,
        otherwise falls back to flat delay + jitter.

        Side effects:
          - OCS: may establish/evict circuits (through circuit_pool)
          - EPS: purely a computation, no side effects

        Returns delay in microseconds.
        """
        path = self.resolve(src, dst)

        if path == PathResolution.OCS:
            delay = self.circuit_pool.compute_delay(
                src, dst, tensor_bytes, current_time_ns,
            )
            self.ocs_requests += 1
            self.ocs_total_delay_us += delay
            return delay
        else:
            delay = self._eps_delay(src, dst, tensor_bytes)
            self.eps_requests += 1
            self.eps_total_delay_us += delay
            return delay

    def _eps_delay(self, src: int, dst: int, tensor_bytes: int) -> float:
        """Compute EPS (electrical) delay for a single rank pair.

        Uses topology.get_pairwise_delay() when topology is available,
        otherwise flat delay + jitter.
        """
        if self.topology is not None:
            return self.topology.get_pairwise_delay(src, dst, tensor_bytes)

        # Flat delay fallback (same as original flat delay logic)
        if self.flat_delay_us <= 0 and self.flat_jitter_us <= 0:
            return 0.0
        jitter = random.uniform(-self.flat_jitter_us, self.flat_jitter_us)
        return max(0.0, self.flat_delay_us + jitter)

    # -- Plan management --------------------------------------------------

    def set_plan(self, plan: Set[Tuple[int, int]]) -> None:
        """Replace the current OCS plan.

        Called when the plan is recomputed (e.g., by OnlineAffinityController
        or after loading a preset plan).
        """
        self.plan = plan

    def set_plan_from_list(
        self, plan_list: list,  # [(src, dst, score), ...]
    ) -> None:
        """Set the plan from a list of (src, dst, score) tuples.

        Convenience wrapper for the format returned by
        ExpertAffinityTracker.compute_circuit_plan().
        """
        self.plan = {(src, dst) for src, dst, _score in plan_list}

    @property
    def plan_size(self) -> int:
        """Number of rank pairs in the current OCS plan."""
        return len(self.plan)

    @property
    def ocs_fraction(self) -> float:
        """Fraction of total requests routed through OCS."""
        total = self.ocs_requests + self.eps_requests
        if total == 0:
            return 0.0
        return self.ocs_requests / total

    # -- Metrics ----------------------------------------------------------

    def get_metrics(self) -> dict:
        """Return path-level metrics for trace export and logging."""
        total = self.ocs_requests + self.eps_requests
        return {
            "ocs_requests": self.ocs_requests,
            "eps_requests": self.eps_requests,
            "ocs_fraction": self.ocs_fraction,
            "ocs_total_delay_us": self.ocs_total_delay_us,
            "eps_total_delay_us": self.eps_total_delay_us,
            "ocs_avg_delay_us": (
                self.ocs_total_delay_us / self.ocs_requests
                if self.ocs_requests > 0 else 0.0
            ),
            "eps_avg_delay_us": (
                self.eps_total_delay_us / self.eps_requests
                if self.eps_requests > 0 else 0.0
            ),
            "plan_size": self.plan_size,
            "total_requests": total,
        }

    def reset_metrics(self) -> None:
        """Reset per-path counters (preserves plan)."""
        self.ocs_requests = 0
        self.eps_requests = 0
        self.ocs_total_delay_us = 0.0
        self.eps_total_delay_us = 0.0
