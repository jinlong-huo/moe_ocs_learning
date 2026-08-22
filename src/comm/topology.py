"""Hierarchical network topology for MoE cluster simulation.

Models the three-tier network fabric found in real GPU clusters:

  Tier 0: Intra-node (NVLink / NVSwitch)
    - GPUs within the same node communicate via NVLink
    - ~900 GB/s bidirectional, ~1-5 us latency
    - All-to-all within a node is very fast

  Tier 1: Intra-pod (InfiniBand / RoCE)
    - Nodes within the same pod/rack connected via IB switch
    - ~200-400 GB/s, ~1-5 us latency
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

    Realistic defaults (can be overridden via YAML):
      - NVLink:    1 us latency, 900 GB/s
      - IB:        3 us latency, 200 GB/s
      - Cross-pod: 10 us latency, 100 GB/s
    """
    num_pods: int = 1
    nodes_per_pod: int = 1
    ranks_per_node: int = 8

    # Per-tier latency in microseconds
    intra_node_latency_us: float = 1.0
    intra_pod_latency_us: float = 3.0
    cross_pod_latency_us: float = 10.0

    # Per-tier bandwidth in GB/s (used for byte-dependent delay)
    intra_node_bandwidth_gbps: float = 900.0
    intra_pod_bandwidth_gbps: float = 200.0
    cross_pod_bandwidth_gbps: float = 100.0

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

    def get_max_tier(self, participating_ranks: list[int]) -> LinkTier:
        """Find the highest (slowest) tier among a set of ranks."""
        max_tier = LinkTier.INTRA_NODE
        my_rank = participating_ranks[0] if participating_ranks else 0
        for other in participating_ranks:
            tier = self.get_link_tier(my_rank, other)
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

        For a global all-to-all, the bottleneck is the worst-tier link
        among all participating rank pairs. This function finds the max
        tier between my_rank and every other rank, then computes:
          delay = (latency_us + tensor_bytes / (bandwidth_gbps * 125)) * multiplier

        The 125 factor converts GB/s to bytes/us: 1 GB/s = 1e9 bytes/s = 1000 bytes/us.
        So bytes / (bandwidth_gbps * 1e9 / 1e6) = bytes / (bandwidth_gbps * 1000).

        Args:
            my_rank: this rank's ID
            world_size: total number of ranks
            tensor_bytes: total bytes in the tensor being communicated

        Returns:
            Delay in microseconds
        """
        # Find the worst link tier
        participating = list(range(world_size))
        max_tier = self.get_max_tier(participating)

        # Get tier-specific parameters
        if max_tier == LinkTier.INTRA_NODE:
            latency = self.config.intra_node_latency_us
            bw = self.config.intra_node_bandwidth_gbps
        elif max_tier == LinkTier.INTRA_POD:
            latency = self.config.intra_pod_latency_us
            bw = self.config.intra_pod_bandwidth_gbps
        else:
            latency = self.config.cross_pod_latency_us
            bw = self.config.cross_pod_bandwidth_gbps

        # Bandwidth-dependent component: bytes / (GB/s * 125 bytes/us per GB/s)
        # Actually 1 GB/s = 1e9 bytes/s = 1000 bytes/us (since 1e9/1e6 = 1000)
        # Wait, 1 GB = 10^9 bytes. 1 second = 10^6 microseconds.
        # So 1 GB/s = 10^9 bytes / 10^6 us = 1000 bytes/us.
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
            bw = self.config.intra_node_bandwidth_gbps
        elif tier == LinkTier.INTRA_POD:
            latency = self.config.intra_pod_latency_us
            bw = self.config.intra_pod_bandwidth_gbps
        else:
            latency = self.config.cross_pod_latency_us
            bw = self.config.cross_pod_bandwidth_gbps

        bw_bytes_per_us = bw * 1000.0
        bw_delay = tensor_bytes / bw_bytes_per_us if bw_bytes_per_us > 0 else 0.0

        return (latency + bw_delay) * self.config.delay_multiplier
