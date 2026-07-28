"""Compute-overlap metrics: measure what fraction of communication
was hidden behind useful computation.

Key metrics:
  - overlap_ratio: time where comm overlapped with compute / total comm time
  - total_wall_time: end-to-end wall clock per step
  - comm_pct: communication time as fraction of total
"""
from __future__ import annotations

from typing import List

from src.utils.timer import TimerEvent


def compute_overlap_ratio(events: List[TimerEvent]) -> float:
    """Estimate the fraction of communication time that overlapped with compute.

    Uses a simple interval-overlap algorithm: for each comm event,
    check if any compute event was active during the same time window.
    """
    comm_events = [e for e in events if e.name.startswith("comm/")]
    compute_events = [e for e in events if e.name.startswith("compute/")]

    if not comm_events:
        return 0.0

    total_comm_ns = sum(e.duration_us for e in comm_events) * 1000
    overlapped_comm_ns = 0.0

    for ce in comm_events:
        ce_start = ce.start_ns
        ce_end = ce.end_ns

        # Check overlap with any compute event
        for xe in compute_events:
            xe_start = xe.start_ns
            xe_end = xe.end_ns

            overlap_start = max(ce_start, xe_start)
            overlap_end = min(ce_end, xe_end)

            if overlap_start < overlap_end:
                overlapped_comm_ns += overlap_end - overlap_start

    if total_comm_ns == 0:
        return 0.0
    return overlapped_comm_ns / total_comm_ns


def step_metrics(events: List[TimerEvent]) -> dict:
    """Aggregate metrics for a single step from its events."""
    if not events:
        return {}

    comm_us = sum(e.duration_us for e in events if "comm" in e.name or "scatter" in e.name or "gather" in e.name)
    compute_us = sum(e.duration_us for e in events if "compute" in e.name)
    route_us = sum(e.duration_us for e in events if "route" in e.name)
    total_us = sum(e.duration_us for e in events)
    overlap = compute_overlap_ratio(events)

    return {
        "total_us": total_us,
        "comm_us": comm_us,
        "compute_us": compute_us,
        "route_us": route_us,
        "overlap_ratio": overlap,
        "num_events": len(events),
    }


# ── OCS-specific metrics ──────────────────────────────────────────────────


def ocs_overlap_ratio(events: List[TimerEvent]) -> float:
    """Measure what fraction of OCS pre-establishment time was hidden behind compute.

    Looks for ocs_pre_establish events and checks their time overlap with
    any compute events. A ratio near 1.0 means circuit reconfig is fully
    hidden; near 0.0 means it's fully exposed on the critical path.
    """
    ocs_events = [e for e in events if "ocs_pre_establish" in e.name]
    compute_events = [e for e in events if "compute" in e.name]

    if not ocs_events:
        return 1.0  # no OCS activity = nothing exposed

    total_ocs_ns = sum(e.duration_us for e in ocs_events) * 1000
    overlapped_ocs_ns = 0.0

    for oe in ocs_events:
        oe_start_ns = oe.start_ns
        oe_end_ns = oe.end_ns

        for ce in compute_events:
            ce_start_ns = ce.start_ns
            ce_end_ns = ce.end_ns

            overlap_start = max(oe_start_ns, ce_start_ns)
            overlap_end = min(oe_end_ns, ce_end_ns)
            if overlap_start < overlap_end:
                overlapped_ocs_ns += overlap_end - overlap_start

    if total_ocs_ns == 0:
        return 1.0
    return overlapped_ocs_ns / total_ocs_ns


def ocs_step_metrics(events: List[TimerEvent]) -> dict:
    """Extend step_metrics with OCS-specific fields."""
    base = step_metrics(events)

    ocs_pre_estab_events = [e for e in events if "ocs_pre_establish" in e.name]
    base["ocs_pre_establish_us"] = sum(e.duration_us for e in ocs_pre_estab_events)
    base["ocs_pre_establish_count"] = len(ocs_pre_estab_events)
    base["ocs_overlap_ratio"] = ocs_overlap_ratio(events)

    # Compute effective overlap: include OCS reconfig as "comm" cost
    comm_us = base["comm_us"] + base["ocs_pre_establish_us"]
    compute_us = base["compute_us"]
    total = comm_us + compute_us
    base["effective_comm_pct"] = (comm_us / total * 100) if total > 0 else 0

    return base


# ── Preset-specific metrics ──────────────────────────────────────────────


def ocs_preset_metrics(
    events: List[TimerEvent],
    ocs_pool_metrics: dict = None,
) -> dict:
    """Metrics specific to OCS preset mode.

    Args:
        events: timer events from a preset-mode run.
        ocs_pool_metrics: dict from transport.get_ocs_metrics().

    Returns dict with:
      - ocs_hit_rate: fraction of A2A target pairs already in OCS pool
      - zero_reconfig_rate: fraction of transfers requiring 0 reconfig time
      - preset_utilization: fraction of pre-established circuits actually used
      - per_step_latency_us: list of per-step total latencies
      - total_reconfig_us: total reconfig time (should be near 0 in preset)
    """
    result = {
        "ocs_hit_rate": 0.0,
        "zero_reconfig_rate": 0.0,
        "preset_utilization": 0.0,
        "per_step_latency_us": [],
        "total_reconfig_us": 0.0,
    }

    if ocs_pool_metrics:
        total_req = max(ocs_pool_metrics.get("total_requests", 1), 1)
        reuse = ocs_pool_metrics.get("circuit_reuses", 0)
        establish = ocs_pool_metrics.get("circuit_establishes", 0)
        active = ocs_pool_metrics.get("active_circuits", 0)
        max_c = ocs_pool_metrics.get("max_circuits", 1)

        result["ocs_hit_rate"] = reuse / total_req
        result["zero_reconfig_rate"] = (total_req - establish) / total_req
        result["preset_utilization"] = active / max_c if max_c > 0 else 0.0
        result["total_reconfig_us"] = ocs_pool_metrics.get(
            "total_reconfig_time_us", 0.0,
        )

    # Per-step latencies
    step_events = {}
    for ev in events:
        parts = ev.name.split("/")
        if len(parts) >= 2 and parts[0].startswith("step"):
            step_key = parts[0]
            if step_key not in step_events:
                step_events[step_key] = {"start": ev.start_ns, "end": ev.end_ns}
            else:
                step_events[step_key]["start"] = min(
                    step_events[step_key]["start"], ev.start_ns,
                )
                step_events[step_key]["end"] = max(
                    step_events[step_key]["end"], ev.end_ns,
                )

    result["per_step_latency_us"] = [
        (v["end"] - v["start"]) / 1000.0
        for v in sorted(step_events.values(), key=lambda x: x["start"])
    ]

    return result

