"""Placement tables: expert -> rank (EP) and rank -> physical location.

Placement is the *independent, cost-side* variable in the MoE testbed. It maps
each global expert id to the rank that owns it (and its local slot on that
rank), and each rank to its physical home in the cluster.

The decoupling invariant is directional and enforced by construction:

    routing never reads placement — only dispatch (scatter_tokens) and the
    network topology delay model do.

So changing placement never changes *which* expert a token hits (that is fixed
by model x input, per Phase 2/3); it only changes *where* that expert's weights
live and how long the token has to travel. Affinity (expert co-activation) is a
property of expert ids, not ranks, so it is unaffected by placement.

The default reproduces the historical ``e // experts_per_rank`` / ``e % k``
mapping exactly — all existing experiments stay bit-identical until a non-linear
placement is explicitly requested.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch


class Placement:
    """Global expert -> (rank, local index) table, plus rank -> physical home.

    Attributes:
        num_experts: total number of experts globally.
        experts_per_rank: number of experts owned by each rank (uniform).
        world_size: number of ranks.
        expert_to_rank: [num_experts] int64 tensor, rank owning each expert.
        expert_to_local: [num_experts] int64 tensor, local index (0..k-1) of
                         each expert on its owning rank.
        rank_to_location: optional [world_size] list of (pod_id, node_id,
                          local_rank). ``None`` means "use Topology's linear
                          default". Owned here so the whole placement is one
                          object; the network model (Topology) may consume it.
    """

    def __init__(
        self,
        num_experts: int,
        experts_per_rank: int,
        world_size: Optional[int] = None,
        expert_to_rank: Optional[Sequence[int]] = None,
        rank_to_location: Optional[Sequence[Tuple[int, int, int]]] = None,
    ):
        if world_size is None:
            world_size = num_experts // experts_per_rank
        if experts_per_rank <= 0:
            raise ValueError(f"experts_per_rank must be positive, got {experts_per_rank}")
        if num_experts % experts_per_rank != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by "
                f"experts_per_rank ({experts_per_rank})"
            )
        if world_size * experts_per_rank != num_experts:
            raise ValueError(
                f"world_size ({world_size}) x experts_per_rank ({experts_per_rank}) "
                f"!= num_experts ({num_experts})"
            )

        self.num_experts = num_experts
        self.experts_per_rank = experts_per_rank
        self.world_size = world_size

        if expert_to_rank is None:
            expert_to_rank = [e // experts_per_rank for e in range(num_experts)]
        expert_to_rank = [int(r) for r in expert_to_rank]
        if len(expert_to_rank) != num_experts:
            raise ValueError(
                f"expert_to_rank length {len(expert_to_rank)} != num_experts {num_experts}"
            )
        if any(not (0 <= r < world_size) for r in expert_to_rank):
            raise ValueError("expert_to_rank contains a rank outside [0, world_size)")

        # Uniform layout: every rank owns exactly experts_per_rank experts.
        counts = [0] * world_size
        for r in expert_to_rank:
            counts[r] += 1
        if any(c != experts_per_rank for c in counts):
            raise ValueError(
                "expert_to_rank is not a uniform permutation: each rank must own "
                f"exactly {experts_per_rank} experts, got counts {counts}"
            )

        self.expert_to_rank = torch.tensor(expert_to_rank, dtype=torch.int64)
        self.expert_to_local = self._build_local_map()

        if rank_to_location is not None:
            rank_to_location = list(rank_to_location)
            if len(rank_to_location) != world_size:
                raise ValueError(
                    f"rank_to_location length {len(rank_to_location)} != world_size {world_size}"
                )
        self.rank_to_location = rank_to_location

    # ── Construction helpers ──────────────────────────────────────────────

    def _build_local_map(self) -> torch.Tensor:
        """Assign local index 0..k-1 to each expert, in ascending global id."""
        local = torch.empty(self.num_experts, dtype=torch.int64)
        seen = [0] * self.world_size
        for e in range(self.num_experts):
            r = int(self.expert_to_rank[e].item())
            local[e] = seen[r]
            seen[r] += 1
        return local

    @classmethod
    def linear(
        cls,
        num_experts: int,
        experts_per_rank: int,
        world_size: Optional[int] = None,
        rank_to_location: Optional[Sequence[Tuple[int, int, int]]] = None,
    ) -> "Placement":
        """The historical contiguous mapping: expert e -> rank e // k, local e % k."""
        return cls(
            num_experts=num_experts,
            experts_per_rank=experts_per_rank,
            world_size=world_size,
            expert_to_rank=None,
            rank_to_location=rank_to_location,
        )

    @classmethod
    def from_permutation(
        cls,
        rank_experts: Sequence[Sequence[int]],
        experts_per_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        rank_to_location: Optional[Sequence[Tuple[int, int, int]]] = None,
    ) -> "Placement":
        """Build from a per-rank list of expert ids (the format
        ``ExpertAffinityTracker.suggest_placement`` returns).

        Args:
            rank_experts: rank_experts[r] is the list of global expert ids owned
                          by rank r. Must be a uniform partition of all experts.
        """
        rank_experts = [list(g) for g in rank_experts]
        world_size = world_size if world_size is not None else len(rank_experts)
        if len(rank_experts) != world_size:
            raise ValueError(
                f"rank_experts has {len(rank_experts)} ranks, expected {world_size}"
            )
        lengths = {len(g) for g in rank_experts}
        if len(lengths) != 1:
            raise ValueError(
                f"non-uniform expert counts across ranks: {sorted(lengths)}"
            )
        k = lengths.pop() if lengths else 0
        experts_per_rank = experts_per_rank if experts_per_rank is not None else k
        if experts_per_rank != k:
            raise ValueError(
                f"experts_per_rank ({experts_per_rank}) != actual per-rank count ({k})"
            )

        num_experts = world_size * experts_per_rank
        expert_to_rank = [0] * num_experts
        for r, ids in enumerate(rank_experts):
            for e in ids:
                if not (0 <= e < num_experts):
                    raise ValueError(f"expert id {e} outside [0, {num_experts})")
                expert_to_rank[e] = r
        return cls(
            num_experts=num_experts,
            experts_per_rank=experts_per_rank,
            world_size=world_size,
            expert_to_rank=expert_to_rank,
            rank_to_location=rank_to_location,
        )

    @classmethod
    def shuffled(
        cls,
        num_experts: int,
        experts_per_rank: int,
        world_size: Optional[int] = None,
        seed: int = 0,
        rank_to_location: Optional[Sequence[Tuple[int, int, int]]] = None,
    ) -> "Placement":
        """A random (but seeded) uniform permutation of experts onto ranks."""
        world_size = world_size if world_size is not None else num_experts // experts_per_rank
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(num_experts, generator=g).tolist()
        expert_to_rank = [0] * num_experts
        for rank, chunk in enumerate(
            [perm[i * experts_per_rank:(i + 1) * experts_per_rank] for i in range(world_size)]
        ):
            for e in chunk:
                expert_to_rank[e] = rank
        return cls(
            num_experts=num_experts,
            experts_per_rank=experts_per_rank,
            world_size=world_size,
            expert_to_rank=expert_to_rank,
            rank_to_location=rank_to_location,
        )

    # ── Query API ─────────────────────────────────────────────────────────

    def resolve(self, expert_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map global expert ids -> (target_rank, local_expert) via the table.

        ``expert_ids`` may be 1D [T] or already-flattened [T*K]; the returned
        tensors have the same shape.
        """
        flat = expert_ids.reshape(-1).long()
        return self.expert_to_rank[flat], self.expert_to_local[flat]

    def experts_on_rank(self, rank: int) -> List[int]:
        """Global expert ids owned by ``rank`` (ascending)."""
        mask = self.expert_to_rank == rank
        return torch.nonzero(mask, as_tuple=False).flatten().tolist()

    def expert_to_rank_dict(self) -> Dict[int, int]:
        """Plain {expert_id: rank} dict (for compute_circuit_plan, exports)."""
        return {e: int(self.expert_to_rank[e].item()) for e in range(self.num_experts)}

    def move_expert(self, expert_id: int, new_rank: int) -> "Placement":
        """Return a new Placement with ``expert_id`` moved to ``new_rank``.

        Uniformity is preserved by swapping ``expert_id`` with the lowest-id
        expert currently owned by ``new_rank``. Returns a *new* object; this
        Placement is unchanged.
        """
        if not (0 <= expert_id < self.num_experts):
            raise ValueError(f"expert_id {expert_id} out of range")
        if not (0 <= new_rank < self.world_size):
            raise ValueError(f"new_rank {new_rank} out of range")
        cur_rank = int(self.expert_to_rank[expert_id].item())
        if cur_rank == new_rank:
            return self

        new_rank_experts = self.experts_on_rank(new_rank)
        swap_with = new_rank_experts[0]
        mapping = self.expert_to_rank.tolist()
        mapping[expert_id], mapping[swap_with] = new_rank, cur_rank
        return Placement(
            num_experts=self.num_experts,
            experts_per_rank=self.experts_per_rank,
            world_size=self.world_size,
            expert_to_rank=mapping,
            rank_to_location=self.rank_to_location,
        )

    # ── Export ────────────────────────────────────────────────────────────

    def to_rank_experts(self) -> List[List[int]]:
        """Per-rank lists of owned expert ids (inverse of ``from_permutation``)."""
        return [self.experts_on_rank(r) for r in range(self.world_size)]

    def to_dict(self) -> Dict:
        d = {
            "num_experts": self.num_experts,
            "experts_per_rank": self.experts_per_rank,
            "world_size": self.world_size,
            "expert_to_rank": self.expert_to_rank.tolist(),
            "rank_experts": self.to_rank_experts(),
        }
        if self.rank_to_location is not None:
            d["rank_to_location"] = [list(t) for t in self.rank_to_location]
        return d

    def __repr__(self) -> str:
        return (
            f"Placement(num_experts={self.num_experts}, "
            f"experts_per_rank={self.experts_per_rank}, world_size={self.world_size})"
        )


def build_placement_manifest(
    placement: "Placement",
    topology=None,
    *,
    strategy: str = "linear",
    seed: Optional[int] = None,
    topology_name: Optional[str] = None,
) -> dict:
    """Serialize the cost-side projection into the trace's placement manifest.

    Records three things, each a *derived* function of (routing, placement):

      1. ``expert_to_rank``  — which rank owns each global expert id
         (the compute-side half: where the expert's weights live).
      2. ``rank_to_location`` — each rank's physical (pod, node, local_rank)
         home (the comm-side half: how far a token must travel).
      3. ``topology``        — the fabric shape + per-tier latency/bandwidth
         that produced the locations.

    ``topology`` is a duck-typed ``Topology`` (has ``.config``, ``.get_location``,
    ``.get_link_tier``). It is optional: when omitted the locations fall back to
    ``placement.rank_to_location`` and the topology section records a flat
    single-node fabric. This helper never touches routing — it only reads
    ``expert_to_rank`` and resolves physical homes.
    """
    manifest: dict = {
        "strategy": strategy,
        "seed": seed,
        "experts_per_rank": placement.experts_per_rank,
        "world_size": placement.world_size,
        "expert_to_rank": placement.expert_to_rank.tolist(),
    }

    if topology is not None:
        cfg = topology.config
        world = placement.world_size
        rank_to_location = [
            [loc.pod_id, loc.node_id, loc.local_rank]
            for loc in (topology.get_location(r) for r in range(world))
        ]
        # Which tiers are actually reachable between any two ranks (exact, not
        # assumed from the shape) — this is the "3-tier rack" evidence.
        tiers_present = set()
        for r in range(world):
            for s in range(r + 1, world):
                tiers_present.add(int(topology.get_link_tier(r, s)))
        manifest["rank_to_location"] = rank_to_location
        manifest["topology"] = {
            "name": topology_name,
            "num_pods": cfg.num_pods,
            "nodes_per_pod": cfg.nodes_per_pod,
            "ranks_per_node": cfg.ranks_per_node,
            "tiers_present": sorted(tiers_present),
            "latency_us": {
                "intra_node": cfg.intra_node_latency_us,
                "intra_pod": cfg.intra_pod_latency_us,
                "cross_pod": cfg.cross_pod_latency_us,
            },
            "bandwidth_gbps": {
                "intra_node": cfg.intra_node_bandwidth_gbps,
                "intra_pod": cfg.intra_pod_bandwidth_gbps,
                "cross_pod": cfg.cross_pod_bandwidth_gbps,
            },
        }
    else:
        # No Topology object: record the flat single-node home explicitly so
        # the trace stays self-documenting (every rank in pod 0 / node 0,
        # local_rank == rank).
        manifest["rank_to_location"] = (
            [list(t) for t in placement.rank_to_location]
            if placement.rank_to_location is not None
            else [[0, 0, r] for r in range(placement.world_size)]
        )
        manifest["topology"] = {
            "name": topology_name,
            "num_pods": 1,
            "nodes_per_pod": 1,
            "ranks_per_node": placement.world_size,
            "tiers_present": [0],  # INTRA_NODE only
            "latency_us": {"intra_node": 1.0, "intra_pod": 3.0, "cross_pod": 10.0},
            "bandwidth_gbps": {"intra_node": 900.0, "intra_pod": 400.0, "cross_pod": 200.0},
        }
    return manifest
