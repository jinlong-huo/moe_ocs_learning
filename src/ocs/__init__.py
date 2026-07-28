"""OCS (Optical Circuit Switching) module for MoE communication modeling.

Provides:
  - OcsCircuitPool: finite pool of reconfigurable optical circuits with LRU eviction
  - OcsTopology: OCS-aware topology configuration
  - ExpertAffinityTracker: expert co-activation tracking for OCS-aware placement
  - preconfig: training trace → affinity → circuit placement plan pipeline

All OCS functionality is opt-in — gated behind ocs.enabled: true in config.
When disabled, zero code paths are affected in the rest of the codebase.
"""

from src.ocs.circuit import OcsCircuit, OcsCircuitPool, OcsCircuitState, OcsPoolMetrics
from src.ocs.topology import OcsTopology, OcsTopologyConfig
from src.ocs.placement import ExpertAffinityTracker

__all__ = [
    "OcsCircuit",
    "OcsCircuitPool",
    "OcsCircuitState",
    "OcsPoolMetrics",
    "OcsTopology",
    "OcsTopologyConfig",
    "ExpertAffinityTracker",
]
