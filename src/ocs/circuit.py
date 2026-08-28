"""OCS circuit model — the field-standard alpha-beta time model.

Every communication cost in this testbed follows the classic alpha-beta
model used across the networking/parallel-computing literature:

    T(n) = alpha + beta * n

  - alpha : fixed per-transfer latency (us) — tier latency for EPS, plus the
            OCS reconfiguration delay on a cold circuit
  - beta  : inverse bandwidth (us/byte) — 1 / fabric bandwidth

OCS is modeled exactly as the field does it: it pays the SAME alpha-beta
EPS cost as the electrical baseline, and a FIXED reconfiguration delay is
added to alpha once per newly established circuit:

    alpha_ocs = alpha_eps + T_reconfig   (cold circuit)
    beta_ocs  = beta_eps                 (same fabric)

⚠️ SUPERSEDED for research claims (C5 in docs/research_assessment.md):
this alpha-adder formulation makes a hot circuit exactly as fast as
electrical and a cold one strictly slower, so no experiment on it can
show an OCS benefit that is not a scheduling artifact. The physically
grounded mechanism (a circuit removes oversubscription — tier promotion)
lives in `src/eval/ocs_eval.py` + `src/eval/cost_model.py`. This module
is retained for the legacy data plane and for reproducing prior results.

The switch is circuit-budget constrained (``max_circuits`` = ports or
wavelengths per rank): when the budget is exhausted the oldest circuit is
reassigned (FIFO port reassignment) and pays T_reconfig. With
max_circuits = world_size - 1 the switch has full fan-out (one circuit per
destination, WSS-style); with max_circuits = 1 it is a single-port space
switch (MEMS) that serially re-points per destination — the authentic
source of OCS reconfiguration pressure on all-to-all traffic.

Two canonical parameterizations of the T_reconfig adder are provided as
configs (see README "OCS verification"):
  - alpha model: fast switch class (SOA / ring-resonator, ns-us),
                 T_reconfig ≈ 1 us, full fan-out
  - beta  model: 3D-MEMS beam-steering (tens of us mechanical + damping),
                 T_reconfig ≈ 50 us, single port

Each spawned process maintains its own pool (spawn isolation).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class OcsPoolMetrics:
    """Accumulated OCS circuit statistics for one rank.

    These are exported in trace metadata and logged at end of run.
    """
    total_requests: int = 0
    circuit_reuses: int = 0          # how often an existing circuit was reused
    circuit_establishes: int = 0     # how many new circuits were established
    circuit_evictions: int = 0       # port reassignments (FIFO budget overflow)
    total_reconfig_time_us: float = 0.0   # cumulative reconfig delay
    total_transfer_time_us: float = 0.0   # cumulative alpha-beta transfer delay


class FixedDelayCircuitPool:
    """OCS = EPS alpha-beta cost + fixed reconfig per circuit switch.

    ``max_circuits`` is the per-rank circuit budget (ports/wavelengths);
    None means full fan-out (world_size - 1). When the topology is enabled,
    the EPS part is the tier-aware pairwise cost; otherwise the flat optical
    path (circuit_latency_us + bytes / circuit_bw) is used as the EPS base.
    """

    def __init__(
        self,
        reconfig_time_us: float,
        topology=None,           # Topology for the EPS tier cost (may be None)
        world_size: int = 1,
        max_circuits: Optional[int] = None,  # per-rank circuit budget (ports)
        circuit_latency_us: float = 1.0,     # flat-path alpha when no topology
        circuit_bw_gbs: float = 25.0,        # flat-path beta when no topology (GB/s)
        rank: int = 0,                       # owning rank (spawn-isolated pool)
    ):
        self.reconfig_time_us = reconfig_time_us
        self.topology = topology
        self.circuit_latency_us = circuit_latency_us
        self.circuit_bw_gbs = circuit_bw_gbs
        self.rank = rank
        if max_circuits is None:
            max_circuits = max(1, world_size - 1)
        if max_circuits < 1:
            raise ValueError(f"max_circuits must be >= 1, got {max_circuits}")
        self.max_circuits = max_circuits

        # FIFO: oldest established circuit first — port reassignment order.
        self._circuits: "OrderedDict[Tuple[int, int], object]" = OrderedDict()
        self.metrics = OcsPoolMetrics()

    # -- Query -----------------------------------------------------------

    def is_established(self, src: int, dst: int) -> bool:
        """Check if a circuit from src to dst is currently established."""
        return (src, dst) in self._circuits

    @property
    def reuse_ratio(self) -> float:
        """Fraction of requests satisfied by an existing circuit."""
        if self.metrics.total_requests == 0:
            return 0.0
        return self.metrics.circuit_reuses / self.metrics.total_requests

    @property
    def active_circuit_count(self) -> int:
        """Number of currently established circuits."""
        return len(self._circuits)

    # -- Circuit management -----------------------------------------------

    def establish(self, src: int, dst: int, current_time_ns: int = 0) -> float:
        """Ensure a circuit exists from src to dst.

        Returns the fixed reconfiguration time incurred (microseconds):
          0.0 if the circuit was already established (hot path),
          reconfig_time_us otherwise (cold path). When the per-rank circuit
          budget is exhausted, the oldest circuit is reassigned (FIFO port
          reassignment — a scheduling policy, not a cache eviction).
        """
        key = (src, dst)
        self.metrics.total_requests += 1

        if key in self._circuits:
            self.metrics.circuit_reuses += 1
            return 0.0

        self.metrics.circuit_establishes += 1

        if len(self._circuits) >= self.max_circuits:
            # Port reassignment: repoint the oldest circuit.
            evicted_key, _evicted = self._circuits.popitem(last=False)
            self.metrics.circuit_evictions += 1

        self._circuits[key] = current_time_ns
        self.metrics.total_reconfig_time_us += self.reconfig_time_us
        return self.reconfig_time_us

    def pre_config(self, plan: list, current_time_ns: int = 0) -> int:
        """Batch-establish circuits from a pre-computed placement plan.

        The plan is GLOBAL (rank, rank, score) triples but each spawned
        process owns a per-rank pool, so only circuits with
        ``src == self.rank`` are established here — historically (C12 in
        docs/research_assessment.md) this filter was missing and a rank
        could burn its entire port budget on circuits ``(other, dst)``
        that its own transport never queries.

        Reconfiguration cost is accounted through ``establish()`` (it lands
        in ``metrics.total_reconfig_time_us``) but is NOT slept on: preset
        mode pays it before inference begins, off the critical path.

        Establishes circuits up to the per-rank budget (no overflow).
        """
        established = 0
        for src, dst, _score in plan:
            if src != self.rank:
                continue
            if (src, dst) in self._circuits:
                continue
            if len(self._circuits) >= self.max_circuits:
                break
            # Account the cold-circuit reconfig (no sleep: off critical path).
            self.establish(src, dst, current_time_ns)
            established += 1
        return established

    def pre_established_count(self) -> int:
        """Number of circuits that were pre-established (preset mode)."""
        return len(self._circuits)

    def compute_delay(
        self, src: int, dst: int, tensor_bytes: int, current_time_ns: int = 0,
    ) -> float:
        """Total delay for a transfer over OCS from src -> dst.

        alpha-beta: EPS part (alpha_eps + beta_eps * n) — the same tier-aware
        pairwise cost the electrical baseline pays (or the flat optical path
        when topology is disabled) — plus the fixed reconfig delay added to
        alpha if this circuit is cold.

        Returns total delay in microseconds. Side effect: updates circuit state.
        """
        reconfig = self.establish(src, dst, current_time_ns)

        if self.topology is not None:
            eps_delay = self.topology.get_pairwise_delay(src, dst, tensor_bytes)
        else:
            bw_bytes_per_us = self.circuit_bw_gbs * 1000.0
            eps_delay = self.circuit_latency_us
            if bw_bytes_per_us > 0 and tensor_bytes > 0:
                eps_delay += tensor_bytes / bw_bytes_per_us

        self.metrics.total_transfer_time_us += eps_delay
        return reconfig + eps_delay

    # -- Snapshot (for debugging) ----------------------------------------

    def snapshot(self) -> dict:
        """Return a snapshot of current circuit state for debugging."""
        return {
            "reconfig_time_us": self.reconfig_time_us,
            "max_circuits": self.max_circuits,
            "active_count": len(self._circuits),
            "reuse_ratio": self.reuse_ratio,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "circuit_reuses": self.metrics.circuit_reuses,
                "circuit_establishes": self.metrics.circuit_establishes,
                "circuit_evictions": self.metrics.circuit_evictions,
                "total_reconfig_time_us": self.metrics.total_reconfig_time_us,
                "total_transfer_time_us": self.metrics.total_transfer_time_us,
            },
        }
