"""Operational hit-rate metrics for OCS preset evaluation.

Two evaluation paths:

1. Simulator-based (original): measures actual OCS pool behavior from
   per-rank trace metadata exported by the simulator.

2. Trace-based (closed loop): directly compares a circuit plan against
   captured routing traces without running the simulator. This closes
   the loop between inference capture and routing prediction.

Simulator-based metrics:
  - operational_hit_rate: fraction of OCS requests that found the circuit
    already hot (pre-established). High = plan was accurate.
  - ocs_coverage: fraction of ALL all-to-all rank-pair communications that
    were routed through OCS (not EPS fallback).
  - preset_utilization: fraction of pre-established circuits that were
    actually used at least once during the test run.
  - preset_waste: fraction of pre-established circuits never used.

Trace-based metrics:
  - expert_pair_hit_rate: fraction of actual co-activated expert pairs
    that were in the plan's predicted set.
  - rank_pair_hit_rate: fraction of actual inter-rank communication pairs
    that the plan predicted (circuit would be pre-established).
  - rank_pair_coverage: fraction of all rank-pair communications covered
    by the plan (upper bound on OCS coverage).
  - plan_utilization: fraction of plan's predicted pairs actually used.
  - token_hit_rate: fraction of tokens where ALL inter-rank pairs were
    pre-covered by the plan.
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


# ═════════════════════════════════════════════════════════════════════
# Trace-based closed-loop hit-rate (no simulator)
# ═════════════════════════════════════════════════════════════════════


@dataclass
class TraceHitRateReport:
    """Hit-rate metrics computed directly from trace-vs-plan comparison.

    No simulator required — pure data analysis of captured routing traces.
    This closes the loop: training trace → plan → test trace → prediction accuracy.
    """

    total_tokens: int = 0
    total_layers: int = 0
    total_expert_instances: int = 0        # sum of top_k across all (token, layer)
    total_expert_pairs: int = 0             # sum of (top_k choose 2) across all events
    total_inter_rank_pairs: int = 0         # pairs where src_rank != dst_rank
    inter_rank_pairs_covered: int = 0       # inter-rank pairs in the plan
    expert_pairs_covered: int = 0           # all (ea, eb) pairs in the plan
    tokens_fully_covered: int = 0           # tokens where ALL inter-rank pairs in plan

    plan_size: int = 0                      # number of (src_rank, dst_rank) in the plan
    plan_pairs_used: int = 0                # plan pairs that appeared in the test trace
    _plan_pairs_used_set: set = field(default_factory=set)  # for multi-trace aggregation

    # Per-layer breakdown
    per_layer: Dict[str, dict] = field(default_factory=dict)

    @property
    def expert_pair_hit_rate(self) -> float:
        """Fraction of co-activated expert pairs predicted by the plan."""
        if self.total_expert_pairs == 0:
            return 0.0
        return self.expert_pairs_covered / self.total_expert_pairs

    @property
    def rank_pair_hit_rate(self) -> float:
        """Fraction of inter-rank communication pairs that the plan predicted."""
        if self.total_inter_rank_pairs == 0:
            return 0.0
        return self.inter_rank_pairs_covered / self.total_inter_rank_pairs

    @property
    def rank_pair_coverage(self) -> float:
        """Fraction of all rank-pair comms covered by plan (incl. same-rank)."""
        total_pairs = self.total_inter_rank_pairs + (
            self.total_expert_pairs - self.total_expert_pairs
        )
        # Coverage relative to inter-rank only (same-rank doesn't need OCS)
        return self.rank_pair_hit_rate

    @property
    def token_hit_rate(self) -> float:
        """Fraction of tokens where ALL inter-rank pairs were in the plan."""
        if self.total_tokens == 0:
            return 0.0
        return self.tokens_fully_covered / self.total_tokens

    @property
    def plan_utilization(self) -> float:
        """Fraction of plan pairs that were actually used in the test trace."""
        if self.plan_size == 0:
            return 0.0
        return self.plan_pairs_used / self.plan_size

    def summary(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "total_layers": self.total_layers,
            "total_expert_instances": self.total_expert_instances,
            "total_expert_pairs": self.total_expert_pairs,
            "total_inter_rank_pairs": self.total_inter_rank_pairs,
            "inter_rank_pairs_covered": self.inter_rank_pairs_covered,
            "expert_pairs_covered": self.expert_pairs_covered,
            "expert_pair_hit_rate": round(self.expert_pair_hit_rate, 4),
            "rank_pair_hit_rate": round(self.rank_pair_hit_rate, 4),
            "rank_pair_coverage": round(self.rank_pair_coverage, 4),
            "token_hit_rate": round(self.token_hit_rate, 4),
            "tokens_fully_covered": self.tokens_fully_covered,
            "plan_size": self.plan_size,
            "plan_pairs_used": self.plan_pairs_used,
            "plan_utilization": round(self.plan_utilization, 4),
        }


def compute_trace_hit_rate(
    plan: List[Tuple[int, int, float]],
    trace_path: str,
    experts_per_rank: int = 1,
) -> TraceHitRateReport:
    """Compute how well a circuit plan predicts a test trace's routing.

    Pure data analysis — no simulator. For every (token, layer) in the trace:
      1. Map selected experts → ranks
      2. Compute all (ea, eb) co-activated expert pairs
      3. Compute all inter-rank (src_rank, dst_rank) pairs
      4. Check which pairs are in the plan's predicted set

    Args:
        plan: list of (src_rank, dst_rank, score) from compute_plan_from_trace(s).
        trace_path: path to a RoutingTrace JSON (test trace).
        experts_per_rank: experts per GPU rank for rank mapping.

    Returns:
        TraceHitRateReport with detailed hit-rate metrics.
    """
    from src.data.routing_schema import RoutingTrace

    trace = RoutingTrace.load(trace_path)
    num_experts = trace.meta.num_experts

    def _rank_of(expert_id: int) -> int:
        return expert_id // experts_per_rank

    # Build plan lookup: set of (src_rank, dst_rank) pairs
    plan_set: set = set()
    for src, dst, _score in plan:
        plan_set.add((src, dst))

    plan_pairs_used: set = set()

    report = TraceHitRateReport(plan_size=len(plan))

    # Deduplicate layers from trace routes
    all_layers = set()
    for route in trace.routes:
        all_layers.update(route.layers.keys())
    report.total_layers = len(all_layers)

    for route in trace.routes:
        report.total_tokens += 1
        token_fully_covered = True

        for layer_id, layer_data in route.layers.items():
            experts = layer_data.experts
            if not experts:
                continue

            report.total_expert_instances += len(experts)

            # Compute all co-activated expert pairs: (ea, eb) for ea != eb
            ranks = [_rank_of(e) for e in experts]
            token_all_inter_rank_covered = True

            for i, ea in enumerate(experts):
                for j, eb in enumerate(experts):
                    if i == j:
                        continue
                    report.total_expert_pairs += 1

                    ra, rb = ranks[i], ranks[j]
                    if ra == rb:
                        # Same-rank: no OCS circuit needed
                        continue

                    report.total_inter_rank_pairs += 1
                    rank_pair = (ra, rb)

                    if rank_pair in plan_set:
                        report.inter_rank_pairs_covered += 1
                        report.expert_pairs_covered += 1
                        plan_pairs_used.add(rank_pair)
                    else:
                        token_all_inter_rank_covered = False
                        token_fully_covered = False

        if token_fully_covered:
            report.tokens_fully_covered += 1

    report.plan_pairs_used = len(plan_pairs_used)
    report._plan_pairs_used_set = plan_pairs_used
    return report


def compute_multi_trace_hit_rate(
    plan: List[Tuple[int, int, float]],
    trace_paths: List[str],
    experts_per_rank: int = 1,
) -> Tuple[TraceHitRateReport, List[TraceHitRateReport]]:
    """Compute hit rates for multiple test traces against a single plan.

    Args:
        plan: circuit plan from training traces.
        trace_paths: list of test trace paths.
        experts_per_rank: experts per GPU rank.

    Returns:
        (aggregated_report, per_trace_reports)
    """
    per_trace = []
    for path in trace_paths:
        per_trace.append(
            compute_trace_hit_rate(plan, path, experts_per_rank)
        )

    # Aggregate
    agg = TraceHitRateReport(plan_size=len(plan))
    all_plan_used: set = set()
    for r in per_trace:
        agg.total_tokens += r.total_tokens
        agg.total_layers = max(agg.total_layers, r.total_layers)
        agg.total_expert_instances += r.total_expert_instances
        agg.total_expert_pairs += r.total_expert_pairs
        agg.total_inter_rank_pairs += r.total_inter_rank_pairs
        agg.inter_rank_pairs_covered += r.inter_rank_pairs_covered
        agg.expert_pairs_covered += r.expert_pairs_covered
        agg.tokens_fully_covered += r.tokens_fully_covered

    # Collect all plan pairs that were used across any test trace
    all_plan_used: set = set()
    for r in per_trace:
        all_plan_used.update(r._plan_pairs_used_set)
    agg.plan_pairs_used = len(all_plan_used)
    agg._plan_pairs_used_set = all_plan_used

    return agg, per_trace


def format_trace_report(report: TraceHitRateReport, label: str = "") -> str:
    """Format a TraceHitRateReport as a human-readable string."""
    s = report.summary()
    header = f"TRACE HIT-RATE REPORT{f' — {label}' if label else ''}"
    lines = [
        "=" * 55,
        header,
        "=" * 55,
        f"Tokens analyzed:            {s['total_tokens']}",
        f"MoE layers per token:       {s['total_layers']}",
        f"Total expert instances:     {s['total_expert_instances']}",
        f"Total expert pairs:         {s['total_expert_pairs']}",
        f"Inter-rank pairs:           {s['total_inter_rank_pairs']}",
        f"",
        f"EXPERT PAIR HIT RATE:       {s['expert_pair_hit_rate']:.1%}",
        f"  (pairs in plan:           {s['expert_pairs_covered']})",
        f"",
        f"RANK PAIR HIT RATE:         {s['rank_pair_hit_rate']:.1%}",
        f"  (inter-rank pairs covered:{s['inter_rank_pairs_covered']})",
        f"  (circuit coverage:        {s['rank_pair_coverage']:.1%})",
        f"",
        f"TOKEN HIT RATE:             {s['token_hit_rate']:.1%}",
        f"  (tokens fully covered:    {s['tokens_fully_covered']})",
        f"",
        f"Plan size:                  {s['plan_size']} circuits",
        f"Plan utilization:           {s['plan_utilization']:.1%}",
        f"  (plan pairs used:         {s['plan_pairs_used']})",
        "=" * 55,
    ]
    return "\n".join(lines)
