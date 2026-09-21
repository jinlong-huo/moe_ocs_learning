"""collective.py -- turning one inference event into flows the fabric can carry.

The inference execution loop produces, per MoE layer, exactly the collective this
repo has been modelling all along: a dispatch all-to-all derived from the routing
trace, and (optionally) its mirror, the combine.  What is new here is that a
collective is an *object with participants and a lifetime*, and each of its flows is
routed onto EPS or onto a circuit **at the moment the collective starts**, using
whatever configuration the OCS is holding at that instant.

Path assignment is the whole physics:

    a cross-pod pair with an established circuit  -> OCS, full NIC rate, no core share
    a cross-pod pair without one                  -> EPS, shares the pooled core
    intra-pod / intra-node                        -> their own tier, never the core

so the same collective is carried differently depending on G(t) -- which is the
effect the instrument exists to show.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.eval.cost_model import Tier, Topology


@dataclass
class Flow:
    """One rank-to-rank message inside one collective."""

    fid: int
    collective: int
    phase: str                      # "dispatch" | "combine"
    src: int
    dst: int
    bytes: float
    tier: str
    path: str                       # "EPS" | "OCS" | "POD" | "NVLINK"
    remaining: float = 0.0
    start_ns: float = 0.0
    finish_ns: float = 0.0
    rate_history: list = field(default_factory=list)

    @property
    def pair(self) -> frozenset:
        return frozenset((self.src, self.dst))


def tier_of(T: np.ndarray, a: int, b: int) -> Tier:
    return Tier(int(T[a, b]))


def path_for(tier: Tier, pair: frozenset, ocs, t_ns: float) -> str:
    if tier == Tier.INTRA_NODE:
        return "NVLINK"
    if tier == Tier.INTRA_POD:
        return "POD"
    if ocs is not None and ocs.has_circuit(min(pair), max(pair), t_ns):
        return "OCS"
    return "EPS"


def flows_for_matrix(mat: np.ndarray, T: np.ndarray, collective: int, t_ns: float,
                     ocs=None, include_combine: bool = True,
                     fid_start: int = 0) -> list:
    """One collective's flows, routed on the configuration held at `t_ns`."""
    W = mat.shape[0]
    out, fid = [], fid_start
    order = [(a, b) for a in range(W) for b in range(W)
             if a != b and mat[a, b] > 0]
    order.sort(key=lambda ab: -mat[ab[0], ab[1]])          # largest first, as the ET does
    for phase, flip in (("dispatch", False), ("combine", True)):
        if phase == "combine" and not include_combine:
            continue
        for (a, b) in order:
            src, dst = (b, a) if flip else (a, b)           # combine mirrors dispatch
            tier = tier_of(T, src, dst)
            out.append(Flow(fid=fid, collective=collective, phase=phase, src=src, dst=dst,
                            bytes=float(mat[a, b]), tier=tier.name,
                            path=path_for(tier, frozenset((src, dst)), ocs, t_ns)))
            fid += 1
    return out
