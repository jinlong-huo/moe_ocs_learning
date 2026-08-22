"""OCS-aware topology configuration and pool management.

OcsTopology wraps a circuit pool (LRU cache or the field-standard
fixed-delay model) with configuration. It is separate from the fabric
Topology (src/comm/topology.py) — OCS is an overlay that layers a fixed
reconfiguration delay on top of the tier-aware EPS cost.

When both OCS and hierarchical topology are enabled, the Transport's
_inject_delay uses the OCS model:
  - cost_model "fixed_delay" (recommended, field-standard): each transfer
    pays the same tier-aware EPS cost as the electrical baseline, plus a
    fixed reconfig delay once per newly established circuit. Directly
    comparable with the EPS baseline (same fabric, same bytes, + delta).
  - cost_model "lru" (legacy): finite circuit cache with LRU eviction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.ocs.circuit import (
    FixedDelayCircuitPool,
    OcsCircuitPool,
    OcsPoolMetrics,
)


@dataclass
class OcsTopologyConfig:
    """Configuration for OCS circuit topology.

    Attributes:
        enabled: master on/off switch — when False, no OCS code runs
        cost_model: "fixed_delay" (EPS + fixed reconfig per switch, the
                    field-standard comparable model) or "lru" (legacy
                    finite circuit cache with eviction)
        max_circuits: maximum simultaneous optical circuits per rank
                      (LRU: cache size; fixed_delay: per-rank circuit
                      budget in ports/wavelengths, None = full fan-out)
        reconfig_time_us: circuit establishment time. alpha model ≈ 1 us
                          (fast switch: SOA / ring-resonator class);
                          beta model ≈ 50 us (MEMS beam-steering class)
        circuit_latency_us: base optical path latency (LRU model only)
        circuit_bandwidth_gbps: per-circuit bandwidth (LRU model only)
        placement_strategy: "round_robin" (default) or "affinity"
                            (expert co-activation aware)
    """
    enabled: bool = False
    cost_model: str = "lru"
    max_circuits: int = 32
    reconfig_time_us: float = 50.0
    circuit_latency_us: float = 1.0
    circuit_bandwidth_gbps: float = 200.0
    placement_strategy: str = "round_robin"


class OcsTopology:
    """Holds OCS configuration and the circuit pool.

    Created once per rank in worker.py. When enabled, the pool is passed
    to Transport for delay injection and to the scheduler for circuit
    pre-establishment.

    Args:
        config: OCS configuration.
        eps_topology: the fabric Topology used for the EPS tier cost
            (required for the authentic fixed_delay model; may be None,
            in which case the flat delay is used as the EPS base).
        flat_delay_us: flat EPS delay fallback (topology disabled).
        world_size: number of ranks (fixed_delay capacity bookkeeping).
    """

    def __init__(
        self,
        config: OcsTopologyConfig,
        eps_topology=None,
        flat_delay_us: float = 0.0,
        world_size: int = 1,
    ):
        self.config = config
        self.pool: Optional[OcsCircuitPool] = None
        if config.enabled:
            if config.cost_model == "fixed_delay":
                self.pool = FixedDelayCircuitPool(
                    reconfig_time_us=config.reconfig_time_us,
                    topology=eps_topology,
                    flat_delay_us=flat_delay_us,
                    world_size=world_size,
                    # None = full fan-out (world_size-1); explicit value =
                    # per-rank circuit budget (ports/wavelengths).
                    max_circuits=config.max_circuits,
                )
            elif config.cost_model == "lru":
                self.pool = OcsCircuitPool(
                    max_circuits=config.max_circuits,
                    reconfig_time_us=config.reconfig_time_us,
                    circuit_latency_us=config.circuit_latency_us,
                    circuit_bw_gbps=config.circuit_bandwidth_gbps,
                )
            else:
                raise ValueError(
                    f"Unknown ocs.cost_model: {config.cost_model!r} "
                    f"(expected 'fixed_delay' or 'lru')"
                )

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.pool is not None

    def get_pool_metrics(self) -> Optional[OcsPoolMetrics]:
        """Return accumulated circuit pool metrics, or None if disabled."""
        if self.pool is None:
            return None
        return self.pool.metrics
