"""fabric.py -- the two fabrics this instrument contrasts, and their state.

EPS  a pooled, dynamically shared core.  Every flow on it is one of k active flows
     and receives core_rate/k, capped by its endpoints' NIC ports.  As flows arrive
     and depart the share changes -- that is the "EPS reacts continuously" half of
     the contrast.

OCS  a set of established circuits, each a dedicated path for one rank pair.  The
     state is a configuration G, held until a reconfiguration replaces it; during a
     reconfiguration no circuit exists (DARK).  Traffic on a circuit is served at
     the full NIC rate and does **not** consume the EPS core pool -- which is
     exactly what "a circuit is a tier promotion" means in this repo.

Nothing here knows about the workload or about time stepping; the engine owns both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.eval.cost_model import FabricConfig, Tier


@dataclass
class EpsFabric:
    """Pooled electrical core + per-rank NIC ports.

    `port_gbytes_per_s` is the per-rank NIC rate (one port per rank, shared by every
    flow that rank is currently sending or receiving).

    `core_gbytes_per_s` is the *pooled* cross-pod capacity.  The default is derived,
    not asserted: `port_rate * world_size / sigma`, so that when every rank is
    saturating its NIC at once each flow receives exactly `port_rate / sigma` --
    which is the constant this repo's FabricConfig calls the cross-pod bandwidth.
    Under lighter load the pool is not saturated and flows run faster, which is the
    behaviour a single literal cannot express.
    """

    port_gbytes_per_s: float = 50.0
    sigma: float = 4.0
    world_size: int = 32
    pod_gbytes_per_s: float = 50.0
    nvlink_gbytes_per_s: float = 450.0

    @property
    def core_gbytes_per_s(self) -> float:
        return self.port_gbytes_per_s * self.world_size / max(self.sigma, 1.0)

    def rate_for_tier(self, tier: Tier) -> float:
        """Nominal path rate before dynamic sharing."""
        if tier == Tier.INTRA_NODE:
            return self.nvlink_gbytes_per_s
        if tier == Tier.INTRA_POD:
            return self.pod_gbytes_per_s
        return self.core_gbytes_per_s          # pooled; per-flow share is computed

    def as_fabric_config(self) -> FabricConfig:
        return FabricConfig(intra_node_gbytes_per_s=self.nvlink_gbytes_per_s,
                            intra_pod_gbytes_per_s=self.pod_gbytes_per_s,
                            nic_gbytes_per_s=self.port_gbytes_per_s,
                            core_oversubscription=self.sigma)


@dataclass
class OcsConfig:
    """What the optical switch is and what it costs to change it."""

    n_circuits: int = 16                 # radix / port budget of the OCS
    ports_per_rank: int = 2              # a circuit consumes one port at each end
    rate_gbytes_per_s: float = 50.0      # a circuit runs at the full NIC rate
    reconfig_us: float = 10_000.0        # 10 ms class by default; 0 / 10 / 1000
    latency_us: float = 6.0              # mirror, not a router
    epoch_collectives: int = 0           # 0 = configure once and hold for the run


@dataclass
class CircuitState:
    """One established circuit and what it has carried inside its epoch."""

    pair: frozenset
    epoch: int
    established_ns: float
    bytes_carried: float = 0.0
    busy_ns: float = 0.0
    last_active_ns: float = 0.0

    @property
    def src(self) -> int:
        return min(self.pair)

    @property
    def dst(self) -> int:
        return max(self.pair)


@dataclass
class OcsFabric:
    """The OCS as a configuration that persists, not as a switch that routes."""

    cfg: OcsConfig = field(default_factory=OcsConfig)
    generation: int = 0
    circuits: dict = field(default_factory=dict)        # frozenset -> CircuitState
    dark_until_ns: float = 0.0
    epochs: list = field(default_factory=list)          # {gen, pairs, start_ns, end_ns}
    dark_intervals: list = field(default_factory=list)  # [start_ns, end_ns] per switchover
    _epoch_start_ns: float = 0.0

    # ── state questions the engine asks ──────────────────────────────
    def is_dark(self, t_ns: float) -> bool:
        return t_ns < self.dark_until_ns

    def has_circuit(self, a: int, b: int, t_ns: float) -> bool:
        if self.is_dark(t_ns):
            return False
        return frozenset((a, b)) in self.circuits

    def active_pairs(self) -> set:
        return set(self.circuits)

    # ── the reconfiguration itself ───────────────────────────────────
    def reconfigure(self, pairs, t_ns: float, free: bool = False) -> dict:
        """Tear down G, hold DARK for reconfig_us, then establish the new G.

        `free` is the setup case: the first configuration can be established before
        the serving window starts, so it is not charged to the window.  Every later
        switchover pays the full class cost.
        """
        old = set(self.circuits)
        new = {frozenset(p) for p in pairs}
        cost_ns = 0.0 if free else self.cfg.reconfig_us * 1e3
        if old:
            self.epochs.append({"generation": self.generation, "start_ns": self._epoch_start_ns,
                                "end_ns": t_ns, "pairs": sorted(tuple(sorted(p)) for p in old)})
        if old == new and old:
            return {"changed": False, "dark_ns": 0.0, "torn_down": 0, "established": 0}
        self.circuits = {}
        self.dark_until_ns = t_ns + cost_ns
        if cost_ns > 0:
            self.dark_intervals.append([t_ns, self.dark_until_ns])
        self.generation += 1
        self._epoch_start_ns = self.dark_until_ns
        self.circuits = {p: CircuitState(pair=p, epoch=self.generation,
                                         established_ns=self.dark_until_ns) for p in new}
        return {"changed": True, "dark_ns": cost_ns, "torn_down": len(old),
                "established": len(new)}
