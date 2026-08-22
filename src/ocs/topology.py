"""OCS-aware topology configuration and pool management.

OcsTopology wraps the fixed-delay circuit pool (the field-standard
alpha-beta model) with configuration. It is separate from the fabric
Topology (src/comm/topology.py) — OCS is an overlay that layers a fixed
reconfiguration delay on top of the tier-aware EPS alpha-beta cost.

When both OCS and hierarchical topology are enabled, the Transport's
_inject_delay uses the OCS model:

    T_ocs(n) = alpha_ocs + beta_ocs * n
    alpha_ocs = alpha_eps + T_reconfig   (cold circuit)
    beta_ocs  = beta_eps                 (same fabric)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.ocs.circuit import FixedDelayCircuitPool, OcsPoolMetrics


@dataclass
class OcsTopologyConfig:
    """Configuration for OCS circuit topology.

    Attributes:
        enabled: master on/off switch — when False, no OCS code runs
        max_circuits: per-rank circuit budget (ports/wavelengths);
                      None = full fan-out (world_size - 1)
        reconfig_time_us: fixed reconfig delay added to alpha on a cold
                          circuit. alpha model ≈ 1 us (fast switch: SOA /
                          ring-resonator class); beta model ≈ 50 us (MEMS
                          beam-steering class)
        circuit_latency_us: flat optical-path alpha (used when the 3-tier
                            topology is disabled)
        circuit_bandwidth_gbps: flat optical-path beta (1/BW; same)
        placement_strategy: "round_robin" (default) or "affinity"
                            (expert co-activation aware)
    """
    enabled: bool = False
    max_circuits: Optional[int] = None
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
            (None = flat optical-path alpha-beta fallback).
        world_size: number of ranks (full-fan-out bookkeeping).
    """

    def __init__(
        self,
        config: OcsTopologyConfig,
        eps_topology=None,
        world_size: int = 1,
    ):
        self.config = config
        self.pool: Optional[FixedDelayCircuitPool] = None
        if config.enabled:
            self.pool = FixedDelayCircuitPool(
                reconfig_time_us=config.reconfig_time_us,
                topology=eps_topology,
                world_size=world_size,
                max_circuits=config.max_circuits,
                circuit_latency_us=config.circuit_latency_us,
                circuit_bw_gbps=config.circuit_bandwidth_gbps,
            )

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self.pool is not None

    def get_pool_metrics(self) -> Optional[OcsPoolMetrics]:
        """Return accumulated circuit pool metrics, or None if disabled."""
        if self.pool is None:
            return None
        return self.pool.metrics
