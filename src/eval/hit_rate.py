"""Operational hit-rate metrics for OCS preset evaluation.

Unlike the statistical hit-rate in affinity_consistency.py (which estimates
from affinity matrices), these measure actual simulator behavior:

  - operational_hit_rate: fraction of OCS requests that found the circuit
    already hot (pre-established). High = plan was accurate.
  - ocs_coverage: fraction of ALL all-to-all rank-pair communications that
    were routed through OCS (not EPS fallback).
  - preset_utilization: fraction of pre-established circuits that were
    actually used at least once during the test run.
  - preset_waste: fraction of pre-established circuits never used.

These are computed from the OCS pool metrics and PathResolver metrics
exported in per-rank trace metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class HitRateReport:
    """Aggregated operational metrics from one or more ranks."""

    total_ocs_requests: int = 0
    ocs_hits: int = 0  # requests that found circuit already hot
    ocs_misses: int = 0  # requests that needed circuit establishment
    eps_requests: int = 0
    total_requests: int = 0  # ocs + eps

    pre_established_count: int = 0  # circuits pre-configured
    pre_established_used: int = 0  # pre-configured circuits that saw >=1 request

    per_rank: List[dict] = field(default_factory=list)

    @property
    def operational_hit_rate(self) -> float:
        """Fraction of OCS requests that hit a hot circuit.

        For ocs_preset mode: this is the fraction of pre-configured
        circuits successfully reused during inference.
        """
        if self.total_ocs_requests == 0:
            return 0.0
        return self.ocs_hits / self.total_ocs_requests

    @property
    def ocs_coverage(self) -> float:
        """Fraction of total rank-pair communications handled by OCS."""
        if self.total_requests == 0:
            return 0.0
        return self.total_ocs_requests / self.total_requests

    @property
    def preset_utilization(self) -> float:
        """Fraction of pre-established circuits that were used."""
        if self.pre_established_count == 0:
            return 0.0
        return self.pre_established_used / self.pre_established_count

    @property
    def preset_waste(self) -> float:
        """Fraction of pre-established circuits never used."""
        if self.pre_established_count == 0:
            return 0.0
        return 1.0 - self.preset_utilization

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "ocs_requests": self.total_ocs_requests,
            "eps_requests": self.eps_requests,
            "ocs_coverage": round(self.ocs_coverage, 4),
            "ocs_hits": self.ocs_hits,
            "ocs_misses": self.ocs_misses,
            "operational_hit_rate": round(self.operational_hit_rate, 4),
            "pre_established_count": self.pre_established_count,
            "pre_established_used": self.pre_established_used,
            "preset_utilization": round(self.preset_utilization, 4),
            "preset_waste": round(self.preset_waste, 4),
        }


def compute_hit_rate_from_metadata(per_rank_metadata: List[dict]) -> HitRateReport:
    """Aggregate hit-rate metrics from per-rank trace metadata.

    Each metadata dict should contain:
      - ocs.metrics: OcsPoolMetrics fields (total_requests, circuit_reuses,
        circuit_establishes, active_circuits, max_circuits)
      - mixed_transport.metrics: PathResolver fields (ocs_requests,
        eps_requests, ocs_fraction, plan_size)

    Args:
        per_rank_metadata: list of metadata dicts, one per rank trace.

    Returns:
        HitRateReport with aggregated metrics.
    """
    report = HitRateReport()

    for rank_idx, meta in enumerate(per_rank_metadata):
        ocs = meta.get("ocs", {})
        ocs_metrics = ocs.get("metrics", {})

        mixed = meta.get("mixed_transport", {})
        mixed_metrics = mixed.get("metrics", {})

        ocs_requests_rank = mixed_metrics.get("ocs_requests", 0)
        eps_requests_rank = mixed_metrics.get("eps_requests", 0)
        ocs_hits_rank = ocs_metrics.get("circuit_reuses", 0)
        ocs_misses_rank = ocs_metrics.get("circuit_establishes", 0)
        pre_established = ocs_metrics.get("active_circuits", 0)

        report.total_ocs_requests += ocs_requests_rank
        report.ocs_hits += ocs_hits_rank
        report.ocs_misses += ocs_misses_rank
        report.eps_requests += eps_requests_rank
        report.total_requests += ocs_requests_rank + eps_requests_rank
        report.pre_established_count += pre_established

        report.per_rank.append({
            "rank": rank_idx,
            "ocs_requests": ocs_requests_rank,
            "eps_requests": eps_requests_rank,
            "ocs_hits": ocs_hits_rank,
            "ocs_misses": ocs_misses_rank,
            "ocs_coverage": (
                ocs_requests_rank / (ocs_requests_rank + eps_requests_rank)
                if (ocs_requests_rank + eps_requests_rank) > 0 else 0.0
            ),
            "operational_hit_rate": (
                ocs_hits_rank / ocs_requests_rank
                if ocs_requests_rank > 0 else 0.0
            ),
            "pre_established": pre_established,
            "max_circuits": ocs.get("max_circuits", 0),
        })

    return report


def compute_hit_rate_from_trace_files(
    trace_paths: List[str],
) -> HitRateReport:
    """Compute hit-rate metrics from per-rank Chrome trace files.

    Reads each trace file, extracts metadata, and aggregates.

    Args:
        trace_paths: list of paths to rank_NN_trace.json files.

    Returns:
        HitRateReport with aggregated metrics.
    """
    import json

    metadata_list = []
    for path in sorted(trace_paths):
        with open(path) as f:
            data = json.load(f)
        # Metadata is in the top-level _metadata field or in metadata
        meta = data.get("_metadata") or data.get("metadata") or {}
        if isinstance(meta, list):
            # Handle case where _metadata is a list
            meta = meta[0] if meta else {}
        metadata_list.append(meta)

    return compute_hit_rate_from_metadata(metadata_list)


def format_report(report: HitRateReport) -> str:
    """Format a HitRateReport as a human-readable string."""
    s = report.summary()
    lines = [
        "=" * 55,
        "OPERATIONAL HIT-RATE REPORT",
        "=" * 55,
        f"Total rank-pair requests:   {s['total_requests']}",
        f"  OCS (circuit):             {s['ocs_requests']} ({s['ocs_coverage']:.1%} coverage)",
        f"  EPS (fallback):            {s['eps_requests']}",
        f"",
        f"OCS hit rate:               {s['operational_hit_rate']:.1%}",
        f"  Hits (already hot):        {s['ocs_hits']}",
        f"  Misses (needed reconfig):  {s['ocs_misses']}",
        f"",
        f"Preset utilization:         {s['preset_utilization']:.1%}",
        f"  Circuits pre-established:  {s['pre_established_count']}",
        f"  Circuits used:             {s['pre_established_used']}",
        f"  Circuits wasted:           {s['preset_waste']:.1%}",
    ]
    if report.per_rank:
        lines.append("")
        lines.append("Per-rank breakdown:")
        lines.append(f"  {'Rank':<6} {'OCS':<8} {'EPS':<8} {'Coverage':<10} {'HitRate':<10}")
        for r in report.per_rank:
            lines.append(
                f"  {r['rank']:<6} {r['ocs_requests']:<8} {r['eps_requests']:<8} "
                f"{r['ocs_coverage']:<10.1%} {r['operational_hit_rate']:<10.1%}"
            )
    lines.append("=" * 55)
    return "\n".join(lines)
