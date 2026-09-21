"""src/net -- an inference-execution-driven EPS/OCS network instrument.

New in this revision; nothing under src/eval, src/ocs or src/comm is touched.

The distinction this package exists to make
───────────────────────────────────────────
`src/eval/cost_model.py` answers "how much traffic does this workload generate, and
what is its bottleneck".  It takes a static per-pair byte matrix and returns a
scalar.  That is the right object for comparing placements, and the wrong object for
watching a fabric: it has no time, no queue, no link state, and no notion of a
circuit existing or not existing at an instant.

This package answers the other question: *when this particular collective happens,
what does the network actually do?*  It drives the fabric from the inference
execution loop (layer -> compute -> collective -> network -> next layer), and it
records the state of every participant while that happens:

    GPU state      COMPUTE / SEND / RECV / IDLE, per rank, over time
    EPS state      active flows, pooled-core utilisation, per-port queue bytes,
                   per-flow instantaneous rate and delay
    OCS state      configuration G(t), circuit ACTIVE / IDLE / DARK, bytes carried,
                   reconfiguration epochs and their cost
    collective     participants, start, finish, network time, EPS/OCS byte split

Model semantics (all fluid, exact under piecewise-constant rates)
────────────────────────────────────────────────────────────────
EPS is statistical multiplexing, so its share is dynamic: a flow's rate is the
minimum of

    egress NIC   port_rate / (# active flows leaving the source rank)
    ingress NIC  port_rate / (# active flows entering the destination rank)
    core pool    core_rate / (# active flows on the core)      core_rate = R*W/sigma

with R = 50 GB/s per rank.  At full saturation that reproduces this repo's
`FabricConfig`: R*W/sigma spread over W ranks = R/sigma = 12.5 GB/s per pair.  When
only a few ranks are active it is *not* congested, which is the behaviour the static
model cannot express.

OCS is circuit switching, so its state is binary and its timescale is the
reconfiguration: a circuit is ACTIVE or it does not exist.  A configured circuit
carries its pair's traffic at `ocs_rate` (full NIC rate) and, crucially, that traffic
**does not draw on the EPS core pool** -- which is what a tier promotion means.  A
reconfiguration is a DARK interval in which no circuit exists at all and every
cross-pod flow falls back to EPS.
"""

from src.net.fabric import EpsFabric, OcsFabric, OcsConfig  # noqa: F401
from src.net.collective import Flow, flows_for_matrix  # noqa: F401
from src.net.engine import EngineConfig, InferenceNetEngine, RunResult  # noqa: F401
from src.net.ocs_controller import (  # noqa: F401
    ControllerConfig, OcsController, make_controller,
)
