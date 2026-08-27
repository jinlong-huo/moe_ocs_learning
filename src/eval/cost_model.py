"""
cost_model.py — communication cost of a routing trace under a placement and a
physical topology.

Scope discipline
────────────────
This module NEVER touches routing.  Its inputs are (a) a ``CellTable`` (the
immutable logical token->expert map), (b) an expert->rank placement, (c) a
topology.  That separation is what makes the invariance claim structural: the
cost model literally cannot change a routing decision.

What was wrong before, and what this fixes
──────────────────────────────────────────
1. **Every peer was charged the whole send buffer.**  The old
   ``Transport._inject_delay`` passed the full ``input_tensor`` byte count to
   *each* destination, then took the max.  That overstates the beta term by
   roughly ``world_size``.  Here traffic is a genuine per-pair byte matrix
   derived from the routing cells.

2. **No congestion, no port capacity.**  Cost was a stateless per-pair
   ``alpha + beta*n`` maxed over peers, i.e. an infinitely wide NIC.  An
   all-to-all cannot finish before the busiest egress and ingress port drains,
   so the headline metric here is a **bottleneck** time over per-rank port
   capacity, not a per-pair delay.

3. **OCS was modelled as strictly worse than EPS.**  The old model set
   ``beta_ocs = beta_eps`` and ``alpha_ocs = alpha_eps + T_reconfig``, so a
   circuit could never win.  Real OCS wins by *bypassing spine
   oversubscription*: a cross-pod pair carried on a direct optical circuit runs
   at full NIC rate instead of its oversubscribed share.  That is modelled here
   as a **tier promotion**, which is the only mechanism by which OCS can
   plausibly pay for its reconfiguration cost.

4. **Dispatch semantics were unmodelled.**  Whether a token is replicated
   per-expert or sent once per destination rank changes the conclusion
   qualitatively (see ``DispatchMode``), so both are first-class and reported
   side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from src.eval.trace_ir import CellTable


# ═══════════════════════════════════════════════════════════════════════
# Physical model
# ═══════════════════════════════════════════════════════════════════════

class Tier(IntEnum):
    """Distance class between two ranks.  ``OPTICAL`` is not a distance — it is
    a cross-pod pair that has been promoted onto a direct circuit."""
    INTRA_NODE = 0     # same server: NVLink / NVSwitch
    INTRA_POD = 1      # different server, same pod: one leaf-spine crossing
    CROSS_POD = 2      # different pod: through the oversubscribed core
    OPTICAL = 3        # cross-pod pair carried on a dedicated OCS circuit


@dataclass
class FabricConfig:
    """Per-tier capability.

    Defaults are chosen to be defensible against current hardware rather than
    round numbers:

      intra-node   NVLink 4 (H100 SXM): 18 links x 25 GB/s = 450 GB/s per GPU
                   per direction.  Latency ~2 us including kernel launch.
      intra-pod    InfiniBand NDR 400 Gb/s = 50 GB/s per direction per NIC.
                   One leaf + one spine crossing, ~5 us.
      cross-pod    Same NIC, but the core layer is oversubscribed, so the
                   sustained per-GPU share is 50 / oversubscription.  Three
                   switch crossings, ~12 us.
      optical      A dedicated circuit removes the core oversubscription: the
                   pair runs at full NIC rate again.  Switching latency of the
                   optical cross-connect itself is negligible once the circuit
                   is up (it is a mirror, not a router), so alpha sits between
                   intra-pod and cross-pod.

    ``*_gbytes_per_s`` are GB/s (bytes), not Gb/s.  The previous code named the
    field ``bandwidth_gbps``, documented it in Gb/s, and then divided as if it
    were GB/s — an 8x error on the two inter-node tiers.
    """

    intra_node_gbytes_per_s: float = 450.0
    intra_pod_gbytes_per_s: float = 50.0
    nic_gbytes_per_s: float = 50.0
    core_oversubscription: float = 4.0
    pod_oversubscription: float = 1.0

    intra_node_latency_us: float = 2.0
    intra_pod_latency_us: float = 5.0
    cross_pod_latency_us: float = 12.0
    optical_latency_us: float = 6.0

    def bandwidth(self, tier: Tier) -> float:
        if tier == Tier.INTRA_NODE:
            return self.intra_node_gbytes_per_s
        if tier == Tier.INTRA_POD:
            return self.intra_pod_gbytes_per_s / max(self.pod_oversubscription, 1.0)
        if tier == Tier.CROSS_POD:
            return self.nic_gbytes_per_s / max(self.core_oversubscription, 1.0)
        return self.nic_gbytes_per_s          # OPTICAL: full NIC rate

    def oversubscription(self, tier: Tier) -> float:
        """How much a tier is contended.  A circuit is only worth provisioning
        for a tier whose oversubscription exceeds 1: OCS removes contention, it
        does not create bandwidth."""
        if tier == Tier.CROSS_POD:
            return max(self.core_oversubscription, 1.0)
        if tier == Tier.INTRA_POD:
            return max(self.pod_oversubscription, 1.0)
        return 1.0

    def latency_us(self, tier: Tier) -> float:
        return {
            Tier.INTRA_NODE: self.intra_node_latency_us,
            Tier.INTRA_POD: self.intra_pod_latency_us,
            Tier.CROSS_POD: self.cross_pod_latency_us,
            Tier.OPTICAL: self.optical_latency_us,
        }[tier]


@dataclass
class Topology:
    """GPU -> node -> pod hierarchy plus an optional set of optical circuits.

    ``circuits`` is a set of unordered rank pairs that have been given a direct
    optical path.  A circuit only ever *promotes* a CROSS_POD pair; provisioning
    one for an already-local pair is legal but pointless, and the tier function
    reflects that (no promotion below CROSS_POD).
    """

    world_size: int
    gpus_per_node: int = 8
    nodes_per_pod: int = 32
    fabric: FabricConfig = field(default_factory=FabricConfig)
    circuits: set[frozenset] = field(default_factory=set)
    rank_to_slot: np.ndarray | None = None    # optional rank -> physical slot
    promote_from: tuple = (Tier.CROSS_POD,)   # which tiers a circuit may replace

    def __post_init__(self):
        if self.rank_to_slot is None:
            self.rank_to_slot = np.arange(self.world_size)
        self.rank_to_slot = np.asarray(self.rank_to_slot)

    # ── geography ────────────────────────────────────────────────────
    def node_of(self, rank: np.ndarray | int):
        return self.rank_to_slot[rank] // self.gpus_per_node

    def pod_of(self, rank: np.ndarray | int):
        return self.rank_to_slot[rank] // (self.gpus_per_node * self.nodes_per_pod)

    @property
    def n_nodes(self) -> int:
        return int(np.ceil(self.world_size / self.gpus_per_node))

    @property
    def n_pods(self) -> int:
        return int(np.ceil(self.world_size / (self.gpus_per_node * self.nodes_per_pod)))

    def tier_matrix(self) -> np.ndarray:
        """[W, W] tier of every ordered rank pair (diagonal = INTRA_NODE)."""
        W = self.world_size
        r = np.arange(W)
        nd, pd = self.node_of(r), self.pod_of(r)
        T = np.full((W, W), int(Tier.INTRA_NODE), dtype=np.int8)
        T[nd[:, None] != nd[None, :]] = int(Tier.INTRA_POD)
        T[pd[:, None] != pd[None, :]] = int(Tier.CROSS_POD)
        # A circuit only replaces a tier that is actually contended, and only
        # if doing so raises the bandwidth.  Provisioning a circuit for an
        # uncontended pair is legal and yields exactly zero benefit.
        promotable = {int(x) for x in self.promote_from
                      if self.fabric.oversubscription(Tier(int(x))) > 1.0
                      and self.fabric.bandwidth(Tier.OPTICAL)
                      > self.fabric.bandwidth(Tier(int(x)))}
        for c in self.circuits:
            it = tuple(c)
            if len(it) != 2:
                continue
            a, b = it
            if int(T[a, b]) in promotable:
                T[a, b] = T[b, a] = int(Tier.OPTICAL)
        return T

    def describe(self) -> dict:
        T = self.tier_matrix()
        off = ~np.eye(self.world_size, dtype=bool)
        vals, cnts = np.unique(T[off], return_counts=True)
        return {
            "world_size": self.world_size, "gpus_per_node": self.gpus_per_node,
            "nodes_per_pod": self.nodes_per_pod, "n_nodes": self.n_nodes,
            "n_pods": self.n_pods, "n_circuits": len(self.circuits),
            "pair_tier_counts": {Tier(int(v)).name: int(c)
                                 for v, c in zip(vals, cnts)},
        }


def hierarchy_for(world_size: int, style: str = "realistic") -> Topology:
    """Named topology classes, each motivated by a real deployment shape.

    ``single_node``   all ranks on one server — the NVLink-only regime.  This
                      is what an 8-GPU EP=8 deployment actually looks like.
    ``single_pod``    ranks spread over nodes inside one pod.  This is the
                      regime nearly all published MoE inference deployments
                      occupy (EP <= 64 fits in ~8 nodes).
    ``multi_pod``     forced across pods (small pods) — the only regime where
                      cross-pod traffic, and therefore OCS, exists at all.
    ``realistic``     8 GPUs/node, 32 nodes/pod: a pod holds 256 GPUs, so a
                      deployment only crosses pods above EP=256.  Using this
                      honestly means small-EP experiments report *zero*
                      cross-pod traffic, which is the correct answer.
    """
    if style == "single_node":
        return Topology(world_size, gpus_per_node=max(world_size, 1), nodes_per_pod=1)
    if style == "single_pod":
        return Topology(world_size, gpus_per_node=8,
                        nodes_per_pod=max(1, int(np.ceil(world_size / 8))))
    if style == "multi_pod":
        # 8 GPUs/node, 2 nodes/pod -> a pod is 16 GPUs, so anything above
        # EP=16 genuinely spans pods.  Used to expose the OCS regime on
        # models whose expert count cannot reach 256-GPU EP.
        return Topology(world_size, gpus_per_node=8, nodes_per_pod=2)
    if style == "realistic":
        return Topology(world_size, gpus_per_node=8, nodes_per_pod=32)
    raise ValueError(f"unknown topology style {style!r}")


TOPOLOGY_STYLES = ("single_node", "single_pod", "multi_pod", "realistic")


# ═══════════════════════════════════════════════════════════════════════
# Dispatch semantics
# ═══════════════════════════════════════════════════════════════════════

class DispatchMode(IntEnum):
    """How a token's activation reaches its experts.

    ``REPLICATED``  one copy per selected expert (DeepSpeed-MoE / classic
                    Megatron all-to-all).  Total dispatch volume is then
                    N*K*H*dtype REGARDLESS of placement — placement can only
                    move bytes between tiers and ranks, never remove them.
    ``DEDUP_RANK``  one copy per destination rank, unpacked locally (the
                    modern kernel; DeepEP does this at node granularity).
                    Total volume now depends on placement through fan-out,
                    which is the only channel through which expert affinity can
                    reduce *volume* rather than just relocate it.
    ``DEDUP_NODE``  one copy per destination NODE, forwarded on NVLink inside
                    the node.  This is what DeepSeek's node-limited routing
                    plus DeepEP actually implements, and it is the tightest
                    (least optimistic) assumption for inter-node volume.
    """
    REPLICATED = 0
    DEDUP_RANK = 1
    DEDUP_NODE = 2


# ═══════════════════════════════════════════════════════════════════════
# Placement
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Placement:
    """Expert -> rank assignment.

    ``expert_to_rank`` is either [E] (one map shared by every layer) or
    [n_layers, E] (a per-layer map).  Per-layer maps are legal in expert
    parallelism — each layer's experts are independent weight tensors — and are
    where essentially all of the exploitable structure lives.
    """

    expert_to_rank: np.ndarray
    world_size: int
    scope: str = "global"                    # "global" | "per_layer"
    layers: np.ndarray | None = None         # layer ids for the per-layer axis
    name: str = "unnamed"

    def __post_init__(self):
        self.expert_to_rank = np.asarray(self.expert_to_rank, dtype=np.int32)
        if self.expert_to_rank.ndim == 2 and self.scope != "per_layer":
            self.scope = "per_layer"

    def ranks_for(self, experts: np.ndarray, layer_idx: np.ndarray | None = None
                  ) -> np.ndarray:
        """[N, K] expert ids -> [N, K] rank ids."""
        if self.scope == "global":
            return self.expert_to_rank[experts]
        return self.expert_to_rank[layer_idx[:, None], experts]

    def experts_per_rank_counts(self) -> np.ndarray:
        m = self.expert_to_rank
        if m.ndim == 1:
            return np.bincount(m, minlength=self.world_size)
        return np.stack([np.bincount(row, minlength=self.world_size) for row in m])


# ═══════════════════════════════════════════════════════════════════════
# The traffic matrix
# ═══════════════════════════════════════════════════════════════════════

def _layer_axis(t: CellTable) -> np.ndarray:
    """Map absolute layer ids to a compact 0..n_layers-1 axis."""
    lut = np.full(int(t.layers.max()) + 1, -1, dtype=np.int32)
    lut[t.layers] = np.arange(t.n_layers, dtype=np.int32)
    return lut[t.layer]


def token_rank(t: CellTable, n_dp: int, seed: int = 0) -> np.ndarray:
    """Which DP rank holds each cell's token.

    Tokens are sharded across DP ranks independently of their content — this is
    what every real batch scheduler does, and it is also the assumption that
    makes the source side of the traffic matrix independent of routing.  It is
    stated explicitly here because it is load-bearing: it is *why* the traffic
    matrix turns out to be near rank-1.
    """
    rng = np.random.default_rng(seed)
    # stable per (run, pos) so the same token keeps the same owner across
    # placements/topologies being compared
    key = t.run.astype(np.int64) * 1_000_003 + t.pos.astype(np.int64)
    return (key % n_dp).astype(np.int32)


@dataclass
class TrafficMatrix:
    """[D, W] dispatch message counts, plus the per-cell fan-out it came from."""
    counts: np.ndarray                 # messages (one message = one H-vector)
    fanout: np.ndarray                 # [n_cells] distinct destinations per cell
    n_dp: int
    world_size: int
    mode: DispatchMode
    n_cells: int

    def rank1_energy(self) -> float:
        """Fraction of the matrix's spectral energy in its first singular value.

        ~1.0 means the matrix is an outer product (token load) x (expert load
        per rank), so no *pairwise* structure exists for a topology optimiser to
        exploit — only marginal load and total volume matter.
        """
        if self.counts.size == 0 or self.counts.sum() == 0:
            return 0.0
        s = np.linalg.svd(self.counts, compute_uv=False)
        return float(s[0] ** 2 / (s ** 2).sum())


def traffic_matrix(t: CellTable, placement: Placement, topo: Topology,
                   mode: DispatchMode = DispatchMode.DEDUP_RANK,
                   n_dp: int | None = None, seed: int = 0) -> TrafficMatrix:
    """Build the dispatch message-count matrix for one workload+placement."""
    W = placement.world_size
    n_dp = n_dp or W
    lay = _layer_axis(t) if placement.scope == "per_layer" else None
    dst = placement.ranks_for(t.experts, lay)          # [N, K]
    src = token_rank(t, n_dp, seed)                    # [N]

    if mode == DispatchMode.REPLICATED:
        fan = np.full(t.n_cells, t.top_k, dtype=np.int32)
        flat_src = np.repeat(src, t.top_k)
        flat_dst = dst.ravel()
    else:
        if mode == DispatchMode.DEDUP_NODE:
            grp = topo.node_of(dst)                    # collapse to node ids
        else:
            grp = dst
        srt = np.sort(grp, axis=1)
        newv = np.ones_like(srt, dtype=bool)
        newv[:, 1:] = srt[:, 1:] != srt[:, :-1]
        fan = newv.sum(1).astype(np.int32)
        rows = np.repeat(np.arange(t.n_cells), t.top_k)[newv.ravel()]
        # representative destination RANK for each distinct group
        rep = np.take_along_axis(dst, np.argsort(grp, axis=1), axis=1).ravel()[newv.ravel()]
        flat_src = src[rows]
        flat_dst = rep

    counts = np.zeros((n_dp, W), dtype=np.float64)
    np.add.at(counts, (flat_src, flat_dst), 1.0)
    return TrafficMatrix(counts, fan, n_dp, W, mode, t.n_cells)


# ═══════════════════════════════════════════════════════════════════════
# Cost
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CostConfig:
    hidden_size: int = 2048
    dtype_bytes: int = 2                    # bf16 activations
    n_microbatches: int = 1
    include_combine: bool = True            # combine mirrors dispatch


def evaluate(t: CellTable, placement: Placement, topo: Topology,
             cost: CostConfig | None = None,
             mode: DispatchMode = DispatchMode.DEDUP_RANK,
             n_dp: int | None = None, seed: int = 0) -> dict:
    """Full communication report for one (routing, placement, topology).

    Metrics, and why each is here:

    ``total_bytes``            volume moved.  Under REPLICATED this is
                              placement-invariant; the fact that it moves under
                              DEDUP is the entire mechanism by which affinity
                              can reduce cost.
    ``inter_node_bytes`` /
    ``cross_pod_bytes``       volume that leaves the cheap domain.  When
                              intra-node is 9x the bandwidth of inter-node,
                              this is the number that sets the wall clock.
    ``bottleneck_us``         max over ranks of egress and ingress drain time
                              plus tier latency.  An all-to-all cannot complete
                              before its busiest port, so this is the honest
                              time metric; a per-pair max is not.
    ``sum_pair_us``           the old-style metric (sum of independent per-pair
                              alpha+beta), kept only for comparison against the
                              previous cost model.
    ``mean_fanout``           mean distinct destinations per routing cell.
    ``active_pairs``          communicating rank pairs -> OCS port demand.
    ``rank1_energy``          spectral test for exploitable pairwise structure.
    ``load_imbalance``        max/mean ingress -> how far the bottleneck is
                              above the ideal balanced collective.
    """
    cost = cost or CostConfig()
    tm = traffic_matrix(t, placement, topo, mode, n_dp, seed)
    B = cost.hidden_size * cost.dtype_bytes
    bytes_mat = tm.counts * B

    T = topo.tier_matrix()
    W, D = tm.world_size, tm.n_dp
    # tier lookup restricted to the DP x EP block actually used
    Tb = T[:D, :W]
    bw = np.array([topo.fabric.bandwidth(Tier(i)) for i in range(4)])
    lat = np.array([topo.fabric.latency_us(Tier(i)) for i in range(4)])

    off = ~np.eye(min(D, W), dtype=bool)
    loc = np.zeros_like(Tb, dtype=bool)
    loc[:min(D, W), :min(D, W)] = ~off      # self pairs are local, no network
    net = bytes_mat.copy()
    net[loc] = 0.0

    def tier_bytes(tier: Tier) -> float:
        m = (Tb == int(tier)) & ~loc
        return float(bytes_mat[m].sum())

    inter_node = float(net[(Tb != int(Tier.INTRA_NODE))].sum())
    cross_pod = tier_bytes(Tier.CROSS_POD)
    optical = tier_bytes(Tier.OPTICAL)

    # ── bottleneck: per-rank egress / ingress drain over per-tier capacity ──
    # Bytes on a rank's inter-node port share the NIC; intra-node bytes use
    # NVLink.  Model them as two separate resources per rank and take the max.
    nvl = (Tb == int(Tier.INTRA_NODE)) & ~loc
    egress_nvl = (net * nvl).sum(1) / topo.fabric.intra_node_gbytes_per_s
    # inter-node bytes drain through the NIC; a cross-pod byte occupies the NIC
    # for longer because the core layer gives it a smaller share
    eff = np.where(nvl, np.inf, bw[Tb])
    with np.errstate(divide="ignore", invalid="ignore"):
        drain = np.where(np.isfinite(eff), net / eff, 0.0)
    egress_nic = drain.sum(1)
    ingress_nic = drain.sum(0)
    ingress_nvl = (net * nvl).sum(0) / topo.fabric.intra_node_gbytes_per_s

    # bytes are in bytes, bandwidth in GB/s -> seconds*1e-9 ; convert to us
    def us(x):
        return x / 1e3          # bytes / (GB/s) = ns  ->  /1e3 = us

    max_tier_present = int(Tb[~loc].max()) if (~loc).any() else 0
    alpha = float(lat[max_tier_present])
    bottleneck = alpha + us(float(max(
        egress_nic.max(initial=0.0), ingress_nic.max(initial=0.0),
        egress_nvl.max(initial=0.0), ingress_nvl.max(initial=0.0))))

    pair_us = float((np.where(net > 0, lat[Tb], 0.0) + us(drain)).sum())

    ing = bytes_mat.sum(0)
    active = int(((net > 0)).sum())
    mult = (2 if cost.include_combine else 1) * cost.n_microbatches

    return {
        "placement": placement.name,
        "placement_scope": placement.scope,
        "topology": topo.describe(),
        "dispatch_mode": mode.name,
        "n_cells": tm.n_cells,
        "world_size": W, "n_dp": D,
        "total_bytes": float(bytes_mat.sum()) * mult,
        "network_bytes": float(net.sum()) * mult,
        "intra_node_bytes": tier_bytes(Tier.INTRA_NODE) * mult,
        "inter_node_bytes": inter_node * mult,
        "intra_pod_bytes": tier_bytes(Tier.INTRA_POD) * mult,
        "cross_pod_bytes": cross_pod * mult,
        "optical_bytes": optical * mult,
        "bottleneck_us": bottleneck * mult,
        "sum_pair_us": pair_us * mult,
        "mean_fanout": float(tm.fanout.mean()),
        "max_fanout": int(tm.fanout.max()),
        "fanout_of_max_possible": float(tm.fanout.mean() / min(t.top_k, W)),
        "active_pairs": active,
        "active_pair_fraction": active / max(1, D * W - min(D, W)),
        "rank1_energy": tm.rank1_energy(),
        "ingress_imbalance": float(ing.max() / max(ing.mean(), 1e-12)),
        "experts_per_rank_imbalance": float(
            placement.experts_per_rank_counts().max()
            / max(placement.experts_per_rank_counts().mean(), 1e-12)),
    }


def compare(base: dict, variant: dict, keys: tuple[str, ...] = (
        "total_bytes", "network_bytes", "inter_node_bytes", "cross_pod_bytes",
        "bottleneck_us", "mean_fanout", "active_pairs", "ingress_imbalance")
        ) -> dict:
    """Relative reduction of ``variant`` against ``base`` (positive = better)."""
    out = {}
    for k in keys:
        b, v = base.get(k), variant.get(k)
        if b is None or v is None:
            continue
        out[k] = {"base": b, "variant": v,
                  "reduction_pct": round(100.0 * (1 - v / b), 3) if b else None}
    return out
