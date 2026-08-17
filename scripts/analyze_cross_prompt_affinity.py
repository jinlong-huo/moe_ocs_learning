#!/usr/bin/env python3
"""Cross-prompt affinity analysis: pairwise similarity between routing traces.

Loads all traces captured from the prompt taxonomy, computes pairwise
affinity similarity metrics, and determines whether same-domain prompts
truly share expert routing patterns.

Metrics computed for each pair of traces (A, B):
  - JS divergence:     Jensen-Shannon divergence of expert distributions (0=identical)
  - Jaccard similarity: overlap of expert sets used
  - Affinity correlation: Pearson R between co-activation matrices
  - Plan hit-rate:     estimated OCS hit rate if plan from A is used on B

Usage:
    python scripts/analyze_cross_prompt_affinity.py
    python scripts/analyze_cross_prompt_affinity.py --manifest outputs/experiment_traces/manifest.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))


def load_affinity_matrix(trace) -> np.ndarray:
    """Build the expert co-activation matrix [E, E] from a routing trace.

    Uses ExpertAffinityTracker to accumulate co-activation, then extracts
    the raw co-activation counts as the affinity matrix.
    """
    import torch
    from src.ocs.placement import ExpertAffinityTracker

    num_experts = trace.meta.num_experts
    tracker = ExpertAffinityTracker(num_experts)

    for route in trace.routes:
        for layer_data in route.layers.values():
            experts = layer_data.experts
            weights = layer_data.weights
            if not experts:
                continue
            if isinstance(experts[0], (int, float)):
                eids = torch.tensor([experts], dtype=torch.long)
                wts = torch.tensor([weights], dtype=torch.float32) if weights else torch.ones(1, dtype=torch.float32)
            else:
                eids = torch.tensor(experts, dtype=torch.long)
                wts = torch.tensor(weights, dtype=torch.float32) if weights else torch.ones(len(experts), dtype=torch.float32)
            tracker.record_routing(eids, wts)

    return tracker.co_activation_counts.numpy()


def compute_pairwise_metrics(traces: dict[str, object]) -> list[dict]:
    """Compute all pairwise affinity metrics between traces.

    Args:
        traces: dict mapping group_id → RoutingTrace

    Returns:
        List of dicts, one per pair, with all metrics.
    """
    import torch
    from src.eval.affinity_consistency import (
        js_divergence,
        jaccard_similarity,
        affinity_correlation,
    )

    ids = sorted(traces.keys())
    metrics_list = []

    for i, id_a in enumerate(ids):
        trace_a = traces[id_a]
        num_experts = trace_a.meta.num_experts

        # Build affinity matrix for A
        aff_a = load_affinity_matrix(trace_a)

        # Expert distribution for A
        dist_a = np.zeros(num_experts)
        for route in trace_a.routes:
            for layer_data in route.layers.values():
                for e in layer_data.experts:
                    if 0 <= e < num_experts:
                        dist_a[e] += 1
        dist_a = dist_a / (dist_a.sum() + 1e-12)

        for j, id_b in enumerate(ids):
            if j <= i:
                continue  # upper triangle only (excluding diagonal)

            trace_b = traces[id_b]
            aff_b = load_affinity_matrix(trace_b)

            # Expert distribution for B
            dist_b = np.zeros(num_experts)
            for route in trace_b.routes:
                for layer_data in route.layers.values():
                    for e in layer_data.experts:
                        if 0 <= e < num_experts:
                            dist_b[e] += 1
            dist_b = dist_b / (dist_b.sum() + 1e-12)

            # JS divergence
            js = js_divergence(dist_a, dist_b)

            # Jaccard: flatten expert selections across all routes
            all_experts_a = []
            all_experts_b = []
            for route in trace_a.routes:
                for layer_data in route.layers.values():
                    all_experts_a.extend(layer_data.experts)
            for route in trace_b.routes:
                for layer_data in route.layers.values():
                    all_experts_b.extend(layer_data.experts)
            jac = jaccard_similarity(
                np.array([all_experts_a]), np.array([all_experts_b])
            )

            # Affinity correlation
            try:
                corr = affinity_correlation(aff_a, aff_b)
            except Exception:
                corr = float("nan")

            # Plan hit-rate: plan from A → estimated hit on B
            # Build the plan directly from A's affinity matrix (no file reload needed)
            try:
                # Get rank-pair communication matrix from B
                world_size = 4
                experts_per_rank = max(1, num_experts // world_size)
                rank_comm = trace_b.rank_communication_matrix(
                    experts_per_rank=experts_per_rank
                )
                total_comm = sum(rank_comm.values())
                if total_comm > 0:
                    # Build plan from A's affinity using the tracker API directly
                    from src.ocs.placement import ExpertAffinityTracker
                    tracker = ExpertAffinityTracker(num_experts)
                    tracker.co_activation_counts = torch.tensor(
                        aff_a, dtype=torch.float64
                    )
                    tracker.expert_usage = torch.tensor(
                        aff_a.sum(axis=0), dtype=torch.float64
                    )
                    tracker.total_samples = 1  # non-zero so normalization works
                    plan = tracker.compute_circuit_plan(
                        experts_per_rank=experts_per_rank,
                        world_size=world_size,
                        max_circuits=16,
                    )
                    plan_pairs = {(src, dst) for src, dst, _ in plan}
                    covered = sum(
                        count for (src, dst), count in rank_comm.items()
                        if (src, dst) in plan_pairs
                    )
                    plan_hit_estimate = covered / total_comm
                else:
                    plan_hit_estimate = 0.0
            except Exception:
                plan_hit_estimate = float("nan")

            metrics_list.append({
                "group_a": id_a,
                "group_b": id_b,
                "domain_a": _trace_domain(traces, id_a),
                "domain_b": _trace_domain(traces, id_b),
                "same_domain": _trace_domain(traces, id_a) == _trace_domain(traces, id_b),
                "js_divergence": round(js, 6),
                "jaccard_similarity": round(jac, 6),
                "affinity_correlation": round(corr, 6) if not np.isnan(corr) else None,
                "plan_hit_rate_estimate": round(plan_hit_estimate, 4) if not np.isnan(plan_hit_estimate) else None,
            })

    return metrics_list


def _trace_domain(traces: dict, group_id: str) -> str:
    """Get domain for a group_id. Stored as trace attribute."""
    trace = traces[group_id]
    return getattr(trace, "_domain", "unknown")


def print_analysis(metrics: list[dict]) -> None:
    """Print a readable analysis of pairwise metrics."""
    # Group by same_domain vs cross_domain
    same = [m for m in metrics if m["same_domain"]]
    cross = [m for m in metrics if not m["same_domain"]]

    print()
    print("=" * 70)
    print("PAIRWISE AFFINITY ANALYSIS")
    print("=" * 70)

    # ── Summary statistics ──
    def _mean(values):
        clean = [v for v in values if v is not None]
        return np.mean(clean) if clean else float("nan")

    print(f"\n{'Metric':<28} {'Same-Domain':>14} {'Cross-Domain':>14} {'Delta':>10}")
    print("-" * 70)

    for metric_key, label, higher_better in [
        ("js_divergence", "JS Divergence", False),
        ("jaccard_similarity", "Jaccard Similarity", True),
        ("affinity_correlation", "Affinity Correlation (R)", True),
        ("plan_hit_rate_estimate", "Plan Hit-Rate Estimate", True),
    ]:
        same_vals = [m[metric_key] for m in same if m[metric_key] is not None]
        cross_vals = [m[metric_key] for m in cross if m[metric_key] is not None]
        same_mean = _mean(same_vals)
        cross_mean = _mean(cross_vals)
        delta = same_mean - cross_mean

        # Check if the delta goes in the expected direction
        expected = "↑" if higher_better else "↓"
        direction = "✓" if ((higher_better and delta > 0) or (not higher_better and delta < 0)) else "✗"

        print(f"{label:<28} {same_mean:>14.4f} {cross_mean:>14.4f} {delta:>+9.4f} {direction} {expected}")

    print("-" * 70)
    print(f"  Same-domain pairs:  {len(same)}")
    print(f"  Cross-domain pairs: {len(cross)}")
    print()

    # ── Per-pair detail ──
    print("Per-pair detail:")
    print(f"  {'A':<20} {'B':<20} {'Domain':<14} {'JS':<10} {'Jaccard':<10} {'Corr(R)':<10} {'HitEst':<8}")
    print(f"  {'-'*18} {'-'*18} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for m in sorted(metrics, key=lambda x: (not x["same_domain"], -(x["affinity_correlation"] or -999))):
        same_marker = "SAME" if m["same_domain"] else "cross"
        print(f"  {m['group_a']:<20} {m['group_b']:<20} {same_marker:<14} "
              f"{m['js_divergence']:<10.4f} {m['jaccard_similarity']:<10.4f} "
              f"{m['affinity_correlation'] or 'N/A':<10} {m['plan_hit_rate_estimate'] or 'N/A':<8}")

    print()

    # ── Key finding ──
    same_corr = _mean([m["affinity_correlation"] for m in same])
    cross_corr = _mean([m["affinity_correlation"] for m in cross])
    if same_corr > cross_corr * 1.3 and not np.isnan(same_corr):
        print("✓ HYPOTHESIS SUPPORTED:")
        print(f"  Same-domain affinity correlation ({same_corr:.4f}) is "
              f"{same_corr / max(cross_corr, 0.001):.1f}x higher than "
              f"cross-domain ({cross_corr:.4f}).")
        print(f"  Semantically similar prompts DO share expert routing patterns.")
        print(f"  → Affinity from one prompt SHOULD generalize to similar prompts.")
    else:
        print("⚠ HYPOTHESIS WEAK / NOT SUPPORTED:")
        print(f"  Same-domain correlation ({same_corr:.4f}) vs "
              f"cross-domain ({cross_corr:.4f}).")
        print(f"  Possible causes: temp too high (non-deterministic), prompts too short,")
        print(f"  or Qwen model routing patterns are input-insensitive for this model size.")

    print()
    print("=" * 70)


def export_csv(metrics: list[dict], output_path: str) -> None:
    """Export pairwise metrics to CSV."""
    if not metrics:
        return
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    print(f"[export] CSV → {output_path}")


def export_html_heatmap(
    trace_ids: list[str],
    metrics: list[dict],
    output_path: str,
) -> None:
    """Export an interactive HTML clustermap of pairwise similarities."""
    # Build a full matrix
    n = len(trace_ids)
    idx_map = {tid: i for i, tid in enumerate(trace_ids)}

    # Build affinity correlation matrix
    corr_matrix = np.eye(n)
    js_matrix = np.zeros((n, n))

    for m in metrics:
        i = idx_map[m["group_a"]]
        j = idx_map[m["group_b"]]
        corr_val = m["affinity_correlation"] or 0.0
        corr_matrix[i, j] = corr_val
        corr_matrix[j, i] = corr_val
        js_matrix[i, j] = m["js_divergence"]
        js_matrix[j, i] = m["js_divergence"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cross-Prompt Affinity Clustermap</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #7ec8e3; }}
  table {{ border-collapse: collapse; margin: 10px 0; }}
  th, td {{ padding: 8px 14px; text-align: center; font-size: 12px; }}
  th {{ background: #16213e; color: #7ec8e3; position: sticky; top: 0; }}
  td {{ border: 1px solid #333; }}
  .same {{ background: #1a3a1a; }}
  .cross {{ background: #3a1a1a; }}
  .high {{ font-weight: bold; color: #4caf50; }}
  .mid {{ color: #ff9800; }}
  .low {{ color: #f44336; }}
  .legend {{ display: flex; gap: 20px; margin: 10px 0; font-size: 13px; }}
  .legend span {{ padding: 3px 8px; border-radius: 3px; }}
  .legend .same-bg {{ background: #1a3a1a; }}
  .legend .cross-bg {{ background: #3a1a1a; }}
  .domain-tag {{ display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 10px; margin-left: 4px; }}
</style>
</head>
<body>
<h1>Cross-Prompt Affinity Clustermap</h1>
<div class="legend">
  <span class="same-bg">Same Domain</span>
  <span class="cross-bg">Cross Domain</span>
  <span class="high">R ≥ 0.5 (high)</span>
  <span class="mid">0.2 ≤ R &lt; 0.5 (moderate)</span>
  <span class="low">R &lt; 0.2 (low)</span>
</div>
<p>Pearson R between expert co-activation matrices. Values near 1.0 = nearly identical routing patterns.</p>
<table>
<tr><th></th>"""
    for tid in trace_ids:
        html += f"<th>{tid}</th>"
    html += "</tr>"

    for i, id_i in enumerate(trace_ids):
        html += f"<tr><th>{id_i}</th>"
        for j, id_j in enumerate(trace_ids):
            if i == j:
                html += '<td style="background:#333;">—</td>'
            else:
                val = corr_matrix[i, j]
                same = (i // 3 == j // 3)  # crude: assume groups of 3 per domain
                cls = "same" if same else "cross"
                lvl = "high" if val >= 0.5 else ("mid" if val >= 0.2 else "low")
                html += f'<td class="{cls} {lvl}">{val:.3f}</td>'
        html += "</tr>"
    html += "</table>"
    html += f"<p><small>Generated from {n} traces. Same-domain = same semantic category (ML, history, cooking).</small></p>"
    html += "</body></html>"

    with open(output_path, "w") as f:
        f.write(html)
    print(f"[export] HTML heatmap → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-prompt pairwise affinity analysis"
    )
    parser.add_argument(
        "--manifest",
        default="outputs/experiment_traces/manifest.json",
        help="Path to manifest.json from capture_experiment_traces.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files (default: same dir as manifest)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[error] Manifest not found: {manifest_path}")
        print(f"        Run capture_experiment_traces.py first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    output_dir = Path(args.output_dir) if args.output_dir else manifest_path.parent

    # ── Load all traces ──
    from src.data.routing_schema import RoutingTrace

    results = manifest["results"]
    traces: dict[str, RoutingTrace] = {}
    domain_map: dict[str, str] = {}
    for r in results:
        gid = r["group_id"]
        trace_path = r["trace_path"]
        if not Path(trace_path).exists():
            print(f"[warn] Missing trace: {trace_path} — skipping {gid}")
            continue
        trace = RoutingTrace.load(trace_path)
        trace._domain = r["domain"]  # attach domain for analysis
        traces[gid] = trace
        domain_map[gid] = r["domain"]

    print(f"Loaded {len(traces)} traces across {len(set(domain_map.values()))} domains")
    for domain in sorted(set(domain_map.values())):
        ids = [gid for gid, d in domain_map.items() if d == domain]
        print(f"  {domain}: {', '.join(ids)}")

    if len(traces) < 2:
        print("[error] Need at least 2 traces for pairwise analysis")
        sys.exit(1)

    # ── Compute pairwise metrics ──
    metrics = compute_pairwise_metrics(traces)

    # ── Print analysis ──
    print_analysis(metrics)

    # ── Export ──
    csv_path = output_dir / "pairwise_affinity_matrix.csv"
    export_csv(metrics, str(csv_path))

    html_path = output_dir / "affinity_clustermap.html"
    trace_ids = sorted(traces.keys())
    export_html_heatmap(trace_ids, metrics, str(html_path))

    # ── Domain-level summary ──
    print("\nDomain-level affinity summary:")
    domains = sorted(set(domain_map.values()))
    for da in domains:
        ids_a = [gid for gid, d in domain_map.items() if d == da]
        for db in domains:
            if db < da:
                continue
            ids_b = [gid for gid, d in domain_map.items() if d == db]
            pairs = [m for m in metrics
                     if m["group_a"] in ids_a and m["group_b"] in ids_b
                     or m["group_a"] in ids_b and m["group_b"] in ids_a]
            if not pairs:
                continue
            avg_corr = np.mean([p["affinity_correlation"] for p in pairs if p["affinity_correlation"] is not None])
            avg_hit = np.mean([p["plan_hit_rate_estimate"] for p in pairs if p["plan_hit_rate_estimate"] is not None])
            same_label = "(same)" if da == db else "(cross)"
            print(f"  {da} ↔ {db} {same_label}: R={avg_corr:.4f}, hit_est={avg_hit:.4f}  ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
