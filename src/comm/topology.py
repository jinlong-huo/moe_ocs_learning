"""Hierarchical network topology for MoE cluster simulation.

Models the three-tier network fabric found in real GPU clusters:

  Tier 0: Intra-node (NVLink / NVSwitch)
    - GPUs within the same node communicate via NVLink
    - ~900 GB/s bidirectional, ~1-5 us latency
    - All-to-all within a node is very fast

  Tier 1: Intra-pod (InfiniBand / RoCE)
    - Nodes within the same pod/rack connected via IB switch
    - ~200-400 Gb/s per port (25-50 GB/s), ~1-5 us latency
    - All-to-all across nodes within a pod adds switch hop latency

  Tier 2: Cross-pod (IB fabric / Ethernet)
    - Pods connected via core switches or IB routers
    - Lower bandwidth, higher latency (10-50+ us)
    - Cross-pod all-to-all is the most expensive

The topology injects realistic per-tier latency and models bandwidth
by adding a byte-size-dependent delay: delay = latency + bytes / bandwidth.

Usage:
    topo = Topology(TopologyConfig(num_pods=2, nodes_per_pod=2, ranks_per_node=4))
    topo.assign(rank)  # call once per rank
    delay_us = topo.get_delay(my_rank, world_size, tensor_bytes)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional, Tuple


class LinkTier(IntEnum):
    """Network tier between two ranks."""
    INTRA_NODE = 0   # same node, NVLink
    INTRA_POD = 1    # different node, same pod, InfiniBand
    CROSS_POD = 2    # different pod, IB fabric / Ethernet


@dataclass
class TopologyConfig:
    """Configuration for hierarchical network topology.

    Authentic field numbers (the EPS baseline must be believable to the
    networking community). Bandwidths are in GB/s — the unit the delay
    math actually divides in (1 GB/s = 1000 bytes/us):
      - Intra-node: NVLink 4 / NVSwitch — ~1 us, 900 GB/s switch bandwidth
        (NVIDIA DGX H100: 900 GB/s aggregated NVSwitch bisection)
      - Intra-pod: InfiniBand NDR — ~3 us switch hop, 400 Gb/s = 50 GB/s
        per port (Mellanox NDR 400Gb/s, ~1-3 us port-to-port)
      - Cross-pod: spine/core fabric — ~10 us, 200 Gb/s = 25 GB/s per port
        (aggregated core-switch hop; 100-400 Gb/s in deployed fabrics)

    Delay = latency + tensor_bytes / (bandwidth_gbs × 1000 bytes/us).
    All values are overridable in YAML.

    Historical note (C7 in docs/research_assessment.md): these fields were
    once named ``*_bandwidth_gbps``, documented as Gb/s, and divided as
    GB/s — the inter-node tiers were modelled 8× faster than the cited
    hardware. The rename makes the unit un-lie-able.
    """
    num_pods: int = 1
    nodes_per_pod: int = 1
    ranks_per_node: int = 8

    # Per-tier latency in microseconds
    intra_node_latency_us: float = 1.0
    intra_pod_latency_us: float = 3.0
    cross_pod_latency_us: float = 10.0

    # Per-tier bandwidth in GB/s (used for byte-dependent delay)
    intra_node_bandwidth_gbs: float = 900.0
    intra_pod_bandwidth_gbs: float = 50.0
    cross_pod_bandwidth_gbs: float = 25.0

    # Multiplier applied to all delays (for scaling experiments)
    delay_multiplier: float = 1.0

    # Optional explicit rank -> physical location override (placement variable).
    # Maps rank -> (pod_id, node_id, local_rank). When present, ``assign`` uses
    # this table instead of the flat linear formula. Ranks not present in the
    # dict fall back to the linear formula.
    rank_locations: Optional[Dict[int, Tuple[int, int, int]]] = None


@dataclass
class RankLocation:
    """Physical location of a rank in the cluster topology."""
    pod_id: int
    node_id: int
    local_rank: int   # index within the node


class Topology:
    """Resolves link tiers and computes topology-aware delays.

    Each rank is placed in the topology based on a flat rank index:
      total_ranks_per_pod = nodes_per_pod * ranks_per_node
      pod_id = rank // total_ranks_per_pod
      node_id = (rank % total_ranks_per_pod) // ranks_per_node
      local_rank = rank % ranks_per_node

    Example: 2 pods × 2 nodes × 4 ranks = 16 ranks
      Ranks 0-3:  pod 0, node 0
      Ranks 4-7:  pod 0, node 1
      Ranks 8-11: pod 1, node 0
      Ranks 12-15: pod 1, node 1
    """

    def __init__(self, config: TopologyConfig):
        self.config = config
        self._locations: dict[int, RankLocation] = {}
        self._total_per_pod = config.nodes_per_pod * config.ranks_per_node

    # -- Rank placement -------------------------------------------------

    def assign(self, rank: int) -> RankLocation:
        """Assign a rank to its physical location and return it.

        Honors an explicit ``rank_locations`` override (the placement variable)
        before falling back to the flat linear formula.
        """
        override = self.config.rank_locations
        if override is not None and rank in override:
            pod_id, node_id, local_rank = override[rank]
            loc = RankLocation(pod_id=pod_id, node_id=node_id, local_rank=local_rank)
            self._locations[rank] = loc
            return loc

        pod_id = rank // self._total_per_pod
        remainder = rank % self._total_per_pod
        node_id = remainder // self.config.ranks_per_node
        local_rank = remainder % self.config.ranks_per_node

        loc = RankLocation(pod_id=pod_id, node_id=node_id, local_rank=local_rank)
        self._locations[rank] = loc
        return loc

    def get_location(self, rank: int) -> RankLocation:
        """Get the location for a previously assigned rank."""
        if rank not in self._locations:
            raise KeyError(f"Rank {rank} has not been assigned. Call assign() first.")
        return self._locations[rank]

    # -- Link tier resolution -------------------------------------------

    def get_link_tier(self, rank_a: int, rank_b: int) -> LinkTier:
        """Determine the network tier between two ranks."""
        a = self.get_location(rank_a)
        b = self.get_location(rank_b)

        if a.pod_id != b.pod_id:
            return LinkTier.CROSS_POD
        if a.node_id != b.node_id:
            return LinkTier.INTRA_POD
        return LinkTier.INTRA_NODE

    def get_max_tier(
        self,
        participating_ranks: list[int],
        viewpoint_rank: Optional[int] = None,
    ) -> LinkTier:
        """Find the highest (slowest) tier among a set of ranks.

        With ``viewpoint_rank`` set, the max is over (viewpoint, other) pairs —
        the correct quantity for one rank's egress bottleneck. Without it, the
        max is over every unordered pair in the set.

        Historical note (C9 in docs/research_assessment.md): this used to
        silently pin the viewpoint to ``participating_ranks[0]``, which made
        the ``my_rank`` argument of ``get_delay`` dead and charged every rank
        the worst tier from rank 0's viewpoint.
        """
        if viewpoint_rank is not None:
            max_tier = LinkTier.INTRA_NODE
            for other in participating_ranks:
                if other == viewpoint_rank:
                    continue
                tier = self.get_link_tier(viewpoint_rank, other)
                if tier > max_tier:
                    max_tier = tier
            return max_tier

        max_tier = LinkTier.INTRA_NODE
        for i, a in enumerate(participating_ranks):
            for b in participating_ranks[i + 1:]:
                tier = self.get_link_tier(a, b)
                if tier > max_tier:
                    max_tier = tier
        return max_tier

    # -- Delay computation -----------------------------------------------

    def get_delay(
        self,
        my_rank: int,
        world_size: int,
        tensor_bytes: int = 0,
    ) -> float:
        """Compute delay in microseconds for a collective involving all ranks.

        The bottleneck is the worst-tier link from *this* rank's viewpoint
        (its egress). This finds the max tier between my_rank and every
        other rank, then computes:
          delay = (latency_us + tensor_bytes / (bandwidth_gbs * 1000)) * multiplier

        Unit conversion: 1 GB/s = 1e9 bytes/s = 1000 bytes/us (1e9 / 1e6).

        Args:
            my_rank: this rank's ID
            world_size: total number of ranks
            tensor_bytes: total bytes in the tensor being communicated

        Returns:
            Delay in microseconds
        """
        # Find the worst link tier from this rank's viewpoint
        participating = list(range(world_size))
        max_tier = self.get_max_tier(participating, viewpoint_rank=my_rank)

        # Get tier-specific parameters
        if max_tier == LinkTier.INTRA_NODE:
            latency = self.config.intra_node_latency_us
            bw = self.config.intra_node_bandwidth_gbs
        elif max_tier == LinkTier.INTRA_POD:
            latency = self.config.intra_pod_latency_us
            bw = self.config.intra_pod_bandwidth_gbs
        else:
            latency = self.config.cross_pod_latency_us
            bw = self.config.cross_pod_bandwidth_gbs

        # 1 GB/s = 10^9 bytes / 10^6 us = 1000 bytes/us
        bw_bytes_per_us = bw * 1000.0
        bw_delay = tensor_bytes / bw_bytes_per_us if bw_bytes_per_us > 0 else 0.0

        total = (latency + bw_delay) * self.config.delay_multiplier
        return total

    def get_pairwise_delay(
        self,
        rank_a: int,
        rank_b: int,
        tensor_bytes: int = 0,
    ) -> float:
        """Compute delay for a pairwise link between two specific ranks."""
        tier = self.get_link_tier(rank_a, rank_b)

        if tier == LinkTier.INTRA_NODE:
            latency = self.config.intra_node_latency_us
            bw = self.config.intra_node_bandwidth_gbs
        elif tier == LinkTier.INTRA_POD:
            latency = self.config.intra_pod_latency_us
            bw = self.config.intra_pod_bandwidth_gbs
        else:
            latency = self.config.cross_pod_latency_us
            bw = self.config.cross_pod_bandwidth_gbs

        bw_bytes_per_us = bw * 1000.0
        bw_delay = tensor_bytes / bw_bytes_per_us if bw_bytes_per_us > 0 else 0.0

        return (latency + bw_delay) * self.config.delay_multiplier
