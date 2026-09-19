#!/usr/bin/env python3
"""contention.py — the two idealisations the OCS claim rests on, as an overlay.

WHY THIS IS A SEPARATE MODULE
─────────────────────────────
```src/eval/cost_model.py``` charges every cross-pod byte a FIXED share `1/sigma`
of the NIC rate, and credits a promoted circuit with the full NIC rate,
unconditionally.  Both are mean-field idealisations of contention, and they are
exactly what decides whether a circuit helps:

1. **Contention is not a deployment constant.**  A fixed `sigma` cannot know how
   many flows are actually competing on a port at that instant.
2. **A circuit is not automatically line rate.**  A single flow is
   window/BDP-limited: filling 400 Gb/s at ~24 us RTT needs a large amount of
   outstanding data, and the fixed model assumes that is free.

The ASTRA-sim cross-check (docs/astra_ordering.md) independently says the same
thing from the other side: on the 16-NPU toy a third of the time is not
bandwidth-proportional.

So this module re-derives the bottleneck with both fixed, and nothing else
changed.  It IMPORTS the existing model and does not modify it — the original
```evaluate()``` stays the reference, and the delta between them is the result.

THE TWO FIXES
─────────────
    fair share   rate(rank) = NIC / max(1, min(sigma, k_rank))
                 k_rank = the rank's number of ACTIVE peers (egress: its row
                 count; ingress: its column count).  A rank talking to two peers
                 is not penalised as if it were contending with sigma of them.
    per-flow cap rate(pair) <= window / RTT(tier),   RTT = 2 * tier latency
                 so a promoted pair carrying one flow lands at
                 window/RTT, not at the full NIC rate.

Both push the OCS gain DOWN in a predictable direction, which is what makes the
fix falsifiable rather than a tuning exercise.

Usage
─────
    python3.12 scripts/contention_before_after.py --workload logs/workload/qwen36 --world-size 32
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.eval.cost_model import (
    CostConfig, DispatchMode, Placement, Tier, Topology, traffic_matrix,
)


@dataclass
class ContentionConfig:
    """The two fixes, each independently switchable so the delta is attributable."""
    fair_share: bool = True
    per_flow_cap: bool = True
    window_bytes: float = 262144.0        # 256 KiB outstanding per flow


def _tier_rate(fabric, tier: Tier) -> float:
    """Nominal per-pair rate at a tier, before fair share and the flow cap."""
    return fabric.bandwidth(tier)


def evaluate_contended(t: "object", placement: Placement, topo: Topology,
                       cost: CostConfig | None = None,
                       mode: DispatchMode = DispatchMode.DEDUP_RANK,
                       n_dp: int | None = None, seed: int = 0,
                       circuits: set | None = None,
                       cc: ContentionConfig | None = None) -> dict:
    """```evaluate()```'s bottleneck, with fair-share contention and a per-flow cap.

    ```circuits``` is a set of ```frozenset({i, j})``` rank pairs to promote: a
    promoted pair is charged as OPTICAL (full NIC rate, optical latency) instead
    of its oversubscribed tier.  That is the same semantics the existing
    ```with_circuits()``` implements.
    """
    cost = cost or CostConfig()
    cc = cc or ContentionConfig()
    circuits = circuits or set()

    tm = traffic_matrix(t, placement, topo, mode, n_dp=n_dp, seed=seed)
    B = cost.hidden_size * cost.dtype_bytes
    bytes_mat = tm.counts * B
    fab = topo.fabric

    W, D = tm.world_size, tm.n_dp
    T = topo.tier_matrix()
    Tb = T[:D, :W]

    loc = np.zeros_like(Tb, dtype=bool)
    loc[:min(D, W), :min(D, W)] = np.eye(min(D, W), dtype=bool)
    net = bytes_mat.copy()
    net[loc] = 0.0

    # ── per-pair tier, after promotion ────────────────────────────────────
    tier_used = Tb.copy()
    for pair in circuits:
        i, j = sorted(pair)
        if i < D and j < W:
            tier_used[i, j] = int(Tier.OPTICAL)
        if j < D and i < W:
            tier_used[j, i] = int(Tier.OPTICAL)

    nvl = (tier_used == int(Tier.INTRA_NODE)) & ~loc

    # ── active peers per rank, which is what "fair share" keys on ─────────
    active = (net > 0)
    k_egress = np.maximum(active.sum(1), 1)      # distinct destinations
    k_ingress = np.maximum(active.sum(0), 1)     # distinct sources

    # ── per-pair service rate ─────────────────────────────────────────────
    # rate = uncontended(tier) / max(1, min(sigma_tier, k_rank))   then the cap.
    # NOTE the per-tier sigma: applying the CORE oversubscription to the POD tier
    # silently drops an intra-pod pair from 50 to 12.5 GB/s, which is a bug the
    # self-check in contention_before_after.py catches (it must reproduce
    # evaluate() exactly with both fixes off).
    # rate    = the rate the REFERENCE model charges (bandwidth() already has /sigma)
    # nominal = the uncontended rate, needed only by the fair-share correction
    rate = np.zeros_like(net)
    nominal = np.zeros_like(net)
    sigma_t = np.ones_like(net)
    for tier in Tier:
        m = (tier_used == int(tier)) & ~loc
        if not m.any():
            continue
        r = _tier_rate(fab, tier)
        s = fab.oversubscription(tier)
        rate[m] = r
        nominal[m] = r * s
        sigma_t[m] = s

    if cc.fair_share:
        # a rank's share is set by how many peers it actually contends with,
        # floored at sigma (never better than uncontended, never worse than NIC/sigma).
        # EXPLICIT axes: sigma_t is (D,W) while k_egress is (D,) -- writing
        # np.minimum(sigma_t, k_egress) broadcasts along the WRONG axis (trailing
        # dims align), silently producing a 3-D rate matrix and garbage drains.
        k_e = np.minimum(sigma_t, k_egress[:, None])     # (D,1) against (D,W)
        k_i = np.minimum(sigma_t, k_ingress[None, :])    # (1,W) against (D,W)
        share_e = nominal / np.maximum(k_e, 1.0)
        share_i = nominal / np.maximum(k_i, 1.0)
        # REPLACE, do not min() against the mean-field rate: when k >= sigma the two
        # agree, and when k < sigma the correction is HIGHER (a rank with few peers
        # contends less than the constant implies).  min() against the old value
        # makes the fix a no-op, which is how the first version silently did nothing.
        rate = np.minimum(share_e, share_i)

    assert rate.shape == net.shape, (
        f"rate {rate.shape} != traffic {net.shape}: a broadcast went wrong")

    if cc.per_flow_cap:
        cap = np.full_like(net, np.inf)
        for tier in Tier:
            m = (tier_used == int(tier)) & ~loc
            if not m.any():
                continue
            rtt_us = 2.0 * fab.latency_us(tier)
            cap[m] = cc.window_bytes / (rtt_us * 1e3)     # bytes/s -> GB/s
        rate = np.minimum(rate, cap)

    with np.errstate(divide="ignore", invalid="ignore"):
        drain = np.where(rate > 0, net / rate, 0.0)
    # intra-node bytes drain through NVLink, not the NIC -- they are counted by
    # egress_nvl/ingress_nvl.  Leaving them in the NIC sum is a 1.1 % error that
    # the self-check caught.
    drain = np.where(nvl, 0.0, drain)

    egress_nvl = (net * nvl).sum(1) / fab.intra_node_gbytes_per_s
    ingress_nvl = (net * nvl).sum(0) / fab.intra_node_gbytes_per_s
    egress_nic = drain.sum(1)
    ingress_nic = drain.sum(0)

    max_tier_present = int(tier_used[~loc].max()) if (~loc).any() else 0
    alpha = float(fab.latency_us(Tier(max_tier_present)))
    mult = (2 if cost.include_combine else 1) * cost.n_microbatches
    bottleneck = alpha + float(max(
        egress_nic.max(initial=0.0), ingress_nic.max(initial=0.0),
        egress_nvl.max(initial=0.0), ingress_nvl.max(initial=0.0))) / 1e3

    return {
        "bottleneck_us": bottleneck * mult,
        "bottleneck_raw_us": bottleneck,
        "binding_us": {
            "egress_nic": float(egress_nic.max(initial=0.0)) / 1e3,
            "ingress_nic": float(ingress_nic.max(initial=0.0)) / 1e3,
            "egress_nvl": float(egress_nvl.max(initial=0.0)) / 1e3,
            "ingress_nvl": float(ingress_nvl.max(initial=0.0)) / 1e3,
        },
        "max_k_egress": int(k_egress.max()),
        "finite_cap_binding_pairs": int(((rate > 0) & (rate < 1e9)).sum()),
        "mean_pair_rate_gbps": float(rate[net > 0].mean()) if (net > 0).any() else 0.0,
    }
