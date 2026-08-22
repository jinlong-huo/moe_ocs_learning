"""OCS circuit pool with LRU eviction and per-rank metrics.

Models a finite pool of reconfigurable optical circuits. Key properties:
  - Circuit establishment costs reconfig_time_us (MEMS: 1-50us, beam-steering: 10-1000us)
  - Once established, circuits have high bandwidth + low latency (optical fast path)
  - Circuit reuse is free — only cold paths and evictions incur reconfig cost
  - LRU eviction when the pool is exhausted (simulates OCS switch reconvergence)

Each spawned process maintains its own OcsCircuitPool (spawn isolation).
Metrics are per-rank; system-wide analysis merges per-rank data post-hoc.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class OcsCircuitState(Enum):
    """State of an individual optical circuit."""
    IDLE = "idle"
    ACTIVE = "active"


@dataclass
class OcsCircuit:
    """A single optical circuit connecting two ranks.

    In a real OCS fabric, circuits are bidirectional or unidirectional
    depending on the architecture. We model unidirectional circuits
    (src -> dst) for finer-grained control — a bidirectional connection
    requires two circuits.
    """
    src_rank: int
    dst_rank: int
    state: OcsCircuitState = OcsCircuitState.ACTIVE
    bw_gbps: float = 200.0
    established_at_ns: int = 0
    last_used_at_ns: int = 0
    pre_established: bool = False


@dataclass
class OcsPoolMetrics:
    """Accumulated OCS circuit pool statistics for one rank.

    These are exported in trace metadata and logged at end of run.
    """
    total_requests: int = 0
    circuit_reuses: int = 0          # how often an existing circuit was reused
    circuit_establishes: int = 0     # how many new circuits were established
    circuit_evictions: int = 0       # how many circuits were evicted (LRU)
    total_reconfig_time_us: float = 0.0   # cumulative reconfig delay
    total_transfer_time_us: float = 0.0   # cumulative transfer delay (excl. reconfig)


class OcsCircuitPool:
    """Finite pool of reconfigurable optical circuits with LRU eviction.

    Key behaviors:
      - establish(src, dst): establishes a circuit. Returns 0 if already active.
        If cold (not in pool), incurs reconfig_time_us. If pool is full, evicts
        the least-recently-used circuit first (also incurs reconfig_time_us).
      - compute_delay(src, dst, bytes): total delay = any reconfig cost +
        circuit_latency + bytes / bandwidth. Auto-establishes the circuit.
      - LRU is implemented via OrderedDict: popitem(last=False) for eviction,
        move_to_end(key) for marking as most recently used.

    Thread safety: not required — used sequentially within a single process.
    """

    def __init__(
        self,
        max_circuits: int,
        reconfig_time_us: float,
        circuit_latency_us: float = 1.0,
        circuit_bw_gbps: float = 200.0,
    ):
        if max_circuits < 1:
            raise ValueError(f"max_circuits must be >= 1, got {max_circuits}")
        self.max_circuits = max_circuits
        self.reconfig_time_us = reconfig_time_us
        self.circuit_latency_us = circuit_latency_us
        self.circuit_bw_gbps = circuit_bw_gbps

        # OrderedDict keyed by (src_rank, dst_rank) — MRU at end, LRU at front
        self._circuits: OrderedDict[Tuple[int, int], OcsCircuit] = OrderedDict()
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
        """Number of currently active circuits in the pool."""
        return len(self._circuits)

    # -- Circuit management -----------------------------------------------

    def establish(self, src: int, dst: int, current_time_ns: int = 0) -> float:
        """Ensure a circuit from src to dst exists in the pool.

        Returns the reconfiguration time incurred (microseconds):
          - 0.0 if circuit was already active (hot path)
          - reconfig_time_us if a new circuit was established (cold path)
          - reconfig_time_us if an existing circuit was evicted + new one established

        Side effects: updates pool metrics, LRU ordering.
        """
        key = (src, dst)
        self.metrics.total_requests += 1

        # Hot path: circuit already active — just bump LRU position
        if key in self._circuits:
            self.metrics.circuit_reuses += 1
            self._circuits[key].last_used_at_ns = current_time_ns
            self._circuits.move_to_end(key)
            return 0.0

        # Cold path: need to establish. Evict LRU if pool is full.
        self.metrics.circuit_establishes += 1

        if len(self._circuits) >= self.max_circuits:
            # Evict least-recently-used (first item in OrderedDict)
            evicted_key, _evicted = self._circuits.popitem(last=False)
            self.metrics.circuit_evictions += 1

        circuit = OcsCircuit(
            src_rank=src,
            dst_rank=dst,
            state=OcsCircuitState.ACTIVE,
            bw_gbps=self.circuit_bw_gbps,
            established_at_ns=current_time_ns,
            last_used_at_ns=current_time_ns,
        )
        self._circuits[key] = circuit
        self.metrics.total_reconfig_time_us += self.reconfig_time_us
        return self.reconfig_time_us

    def pre_config(
        self,
        plan: list,
        current_time_ns: int = 0,
    ) -> int:
        """Batch-establish circuits from a pre-computed placement plan.

        The plan is a list of (src_rank, dst_rank, score) tuples sorted by
        descending affinity score. Circuits are established in order, filling
        the pool up to max_circuits. Pre-established circuits are tagged so
        that metrics can distinguish training-time reconfig from inference-time
        pre-config.

        Used for inference preset mode: the plan is computed from training
        affinity data, then pre-loaded before the first token is processed.
        No runtime reconfiguration is incurred for pre-established circuits.

        Returns:
            Number of circuits pre-established (may be fewer than plan if pool
            fills up).
        """
        established = 0
        for src, dst, _score in plan:
            if len(self._circuits) >= self.max_circuits:
                break
            key = (src, dst)
            if key in self._circuits:
                continue
            circuit = OcsCircuit(
                src_rank=src,
                dst_rank=dst,
                state=OcsCircuitState.ACTIVE,
                bw_gbps=self.circuit_bw_gbps,
                established_at_ns=current_time_ns,
                last_used_at_ns=current_time_ns,
                pre_established=True,
            )
            self._circuits[key] = circuit
            self.metrics.circuit_establishes += 1
            established += 1
        return established

    def pre_established_count(self) -> int:
        """Number of circuits that were pre-established (preset mode)."""
        return sum(1 for c in self._circuits.values() if c.pre_established)

    def compute_delay(
        self, src: int, dst: int, tensor_bytes: int, current_time_ns: int = 0,
    ) -> float:
        """Compute total delay for a transfer over OCS from src -> dst.

        This is the main entry point for delay injection. It:
          1. Establishes the circuit if needed (may incur reconfig cost)
          2. Computes the per-byte transfer delay over the optical path

        Returns total delay in microseconds. Side effect: updates circuit state.
        """
        reconfig = self.establish(src, dst, current_time_ns)

        # Bandwidth-dependent component:
        #   1 GB/s = 1e9 bytes/s = 1000 bytes/us
        #   transfer_us = bytes / (bw_gbps * 1000)
        bw_bytes_per_us = self.circuit_bw_gbps * 1000.0
        transfer = self.circuit_latency_us
        if bw_bytes_per_us > 0 and tensor_bytes > 0:
            transfer += tensor_bytes / bw_bytes_per_us

        self.metrics.total_transfer_time_us += transfer
        return reconfig + transfer

    # -- Snapshot (for debugging) ----------------------------------------

    def snapshot(self) -> dict:
        """Return a snapshot of current pool state for debugging."""
        circuits = []
        for (src, dst), circ in self._circuits.items():
            circuits.append({
                "src": src,
                "dst": dst,
                "state": circ.state.value,
                "bw_gbps": circ.bw_gbps,
                "established_at_ns": circ.established_at_ns,
                "last_used_at_ns": circ.last_used_at_ns,
            })
        return {
            "max_circuits": self.max_circuits,
            "active_count": len(self._circuits),
            "reuse_ratio": self.reuse_ratio,
            "circuits": circuits,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "circuit_reuses": self.metrics.circuit_reuses,
                "circuit_establishes": self.metrics.circuit_establishes,
                "circuit_evictions": self.metrics.circuit_evictions,
                "total_reconfig_time_us": self.metrics.total_reconfig_time_us,
                "total_transfer_time_us": self.metrics.total_transfer_time_us,
            },
        }


class FixedDelayCircuitPool:
    """Field-standard OCS cost model: EPS tier cost + fixed reconfig per switch.

    Most OCS simulations in the literature do NOT model a finite circuit
    cache with LRU eviction. They model the switch as: every transfer pays
    the same tier-aware cost as the electrical baseline (EPS), and a *fixed*
    reconfiguration delay is added once per newly established circuit:

        T_ocs(src, dst, bytes) = T_eps(src, dst, bytes) + T_reconfig * [cold]

    Authenticity constraint — circuit budget: a real switch has a finite
    number of ports / wavelengths, so each rank can hold at most
    ``max_circuits`` simultaneous outgoing circuits. When the budget is
    exhausted, the OLDEST circuit is reassigned (FIFO port reassignment —
    a scheduling policy, not an LRU cache), paying T_reconfig. With
    max_circuits = world_size - 1 the switch has full fan-out (one
    wavelength per destination, WSS-style); with max_circuits = 1 it is a
    single-port space switch (MEMS) that must serially re-point per
    destination — which is exactly where OCS's reconfiguration pressure
    comes from in real all-to-all workloads.

    Two canonical parameterizations of T_reconfig are provided as configs
    (see README "EPS vs OCS cost models"):
      - alpha model: fast switch class (SOA / ring-resonator, ns-us),
                     T_reconfig ≈ 1 us
      - beta  model: MEMS beam-steering class (tens of us mechanical +
                     damping), T_reconfig ≈ 50 us

    This class exposes the same interface as OcsCircuitPool so Transport,
    the schedulers, and the worker need no changes.
    """

    def __init__(
        self,
        reconfig_time_us: float,
        topology=None,           # Topology for the EPS tier cost (may be None)
        flat_delay_us: float = 0.0,  # EPS fallback when topology is disabled
        world_size: int = 1,
        max_circuits: int | None = None,  # per-rank circuit budget (ports);
                                          # None = full fan-out (world_size-1)
    ):
        self.reconfig_time_us = reconfig_time_us
        self.topology = topology
        self.flat_delay_us = flat_delay_us
        if max_circuits is None:
            max_circuits = max(1, world_size - 1)
        if max_circuits < 1:
            raise ValueError(f"max_circuits must be >= 1, got {max_circuits}")
        self.max_circuits = max_circuits

        # FIFO: oldest established circuit first — port reassignment order.
        self._circuits: "OrderedDict[Tuple[int, int], OcsCircuit]" = OrderedDict()
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
          budget is exhausted, the oldest circuit is reassigned (FIFO) and
          also pays reconfig_time_us.
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

        circuit = OcsCircuit(
            src_rank=src,
            dst_rank=dst,
            state=OcsCircuitState.ACTIVE,
            bw_gbps=0.0,  # data path = EPS fabric; BW comes from the topology
            established_at_ns=current_time_ns,
            last_used_at_ns=current_time_ns,
        )
        self._circuits[key] = circuit
        self.metrics.total_reconfig_time_us += self.reconfig_time_us
        return self.reconfig_time_us

    def pre_config(self, plan: list, current_time_ns: int = 0) -> int:
        """Batch-establish circuits from a pre-computed placement plan.

        Establishes plan circuits up to the per-rank budget (no overflow).
        Used by preset mode: the reconfig cost is paid before inference
        begins, so it does not appear on the inference-time critical path.
        """
        established = 0
        for src, dst, _score in plan:
            if len(self._circuits) >= self.max_circuits:
                break
            if (src, dst) in self._circuits:
                continue
            circuit = OcsCircuit(
                src_rank=src,
                dst_rank=dst,
                state=OcsCircuitState.ACTIVE,
                bw_gbps=0.0,
                established_at_ns=current_time_ns,
                last_used_at_ns=current_time_ns,
                pre_established=True,
            )
            self._circuits[key := (src, dst)] = circuit
            self.metrics.circuit_establishes += 1
            established += 1
        return established

    def pre_established_count(self) -> int:
        """Number of circuits that were pre-established (preset mode)."""
        return sum(1 for c in self._circuits.values() if c.pre_established)

    def compute_delay(
        self, src: int, dst: int, tensor_bytes: int, current_time_ns: int = 0,
    ) -> float:
        """Total delay for a transfer over OCS from src -> dst.

        EPS part: the same tier-aware pairwise cost the electrical baseline
        pays (topology), or the flat delay when topology is disabled.
        OCS part: the fixed reconfig delay if this circuit is cold (or a
        port had to be reassigned).

        Returns total delay in microseconds. Side effect: updates circuit state.
        """
        reconfig = self.establish(src, dst, current_time_ns)

        if self.topology is not None:
            eps_delay = self.topology.get_pairwise_delay(src, dst, tensor_bytes)
        else:
            eps_delay = self.flat_delay_us

        self.metrics.total_transfer_time_us += eps_delay
        return reconfig + eps_delay

    # -- Snapshot (for debugging) ----------------------------------------

    def snapshot(self) -> dict:
        """Return a snapshot of current circuit state for debugging."""
        return {
            "cost_model": "fixed_delay",
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
