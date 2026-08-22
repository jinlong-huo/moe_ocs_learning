"""
analyze.py — Multi-tenant session analysis.

Quantifies what a single-tenant trace cannot: contention.  For each
engine step we know which tenants had tokens computed together, so we can
measure (a) per-tenant delay vs the level of concurrency the tenant
experienced, and (b) expert-dispatch contention — how often two tenants
in the same step routed to the same expert (the "same hardware slot"
collision that serializes compute / bandwidth).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from src.serving.schema import MultiTenantSession, TenantSummary


def _load_session(session_path: Path) -> MultiTenantSession:
    return MultiTenantSession.load(str(session_path))


def _load_tenant_trace(session_dir: Path, tenant: TenantSummary):
    from src.data.routing_schema import RoutingTrace

    return RoutingTrace.load(str(session_dir / tenant.trace_path))


def _concurrency_per_tenant(session: MultiTenantSession) -> dict[str, list[tuple[float, int]]]:
    """For each tenant, the concurrency it saw while its tokens were computed."""
    active: dict[str, list] = defaultdict(list)
    for s in session.steps:
        n = len(s.tokens)
        for req_id in s.tokens:
            active[req_id].append((s.t_s, n))
    return dict(active)


def _expert_contention(session_dir: Path, session: MultiTenantSession) -> dict:
    """Expert-slot collision rate across multi-tenant steps.

    For every step with ≥2 tenants, count how many (step, expert) slots
    were demanded by more than one tenant — those are the expert compute /
    dispatch-bandwidth collisions between tenants.
    """
    by_req: dict[str, set[str]] = {}
    for s in session.steps:
        if len(s.tokens) < 2:
            continue
        per_tenant_experts: dict[str, set[int]] = {}
        for req_id in s.tokens:
            trace = _load_tenant_trace(session_dir, session.by_request_id()[req_id])
            # positions computed in this step: we approximate with all routes
            # whose position range falls in the tenant's computed span; the
            # step-level expert sets are the union over the tenant's trace.
            exps: set[int] = set()
            for route in trace.routes:
                for lr in route.layers.values():
                    exps.update(lr.experts)
            per_tenant_experts[req_id] = exps
        # Per-expert demand: number of tenants touching expert e in this step.
        demand: Counter = Counter()
        for exps in per_tenant_experts.values():
            for e in exps:
                demand[e] += 1
        shared = sum(max(0, c - 1) for c in demand.values())
        total = sum(demand.values())
        by_req[str(s.step)] = {
            "num_tenants": len(s.tokens),
            "shared_expert_demand": shared,
            "total_expert_demand": total,
            "collision_ratio": shared / total if total else 0.0,
        }
    return by_req


def analyze_session(session_dir: str | Path, baseline_dir: str | Path | None = None,
                    plot: bool = False) -> dict:
    """Compute the multi-tenant session report.

    Returns a report dict; optionally saves ``session_report.json`` and
    ``timeline.png`` into the session directory.
    """
    session_dir = Path(session_dir)
    session = _load_session(session_dir / "session.json")

    # ── per-tenant latency ─────────────────────────────────────────
    tenants_report = []
    for t in sorted(session.tenants, key=lambda x: x.arrival_s):
        tenants_report.append({
            "request_id": t.request_id,
            "arrival_s": round(t.arrival_s, 4),
            "ttft_s": round(t.ttft_s, 4),
            "tpot_s": round(t.tpot_s, 4),
            "itl_max_s": round(max(t.itl_s), 4) if t.itl_s else 0.0,
            "finish_s": round(t.finish_s, 4),
            "prompt_len": t.prompt_len,
            "generated_len": t.generated_len,
            "throughput_tok_s": round(t.output_throughput_tok_s, 2),
        })

    # ── delay vs concurrency ───────────────────────────────────────
    conc_per_tenant = _concurrency_per_tenant(session)
    delay_vs_conc = defaultdict(lambda: {"ttft_sum": 0.0, "n": 0})
    for t in session.tenants:
        seen = conc_per_tenant.get(t.request_id, [])
        if not seen:
            peak = 1
        else:
            peak = max(c for _, c in seen)
        delay_vs_conc[peak]["ttft_sum"] += t.ttft_s
        delay_vs_conc[peak]["n"] += 1

    # ── contention ─────────────────────────────────────────────────
    multi_steps = [s for s in session.steps if len(s.tokens) >= 2]
    contention = _expert_contention(session_dir, session)
    collision_ratios = [v["collision_ratio"] for v in contention.values()]

    # ── baseline comparison (sequential run of the same workload) ──
    baseline = None
    if baseline_dir is not None:
        base = _load_session(Path(baseline_dir) / "session.json")
        base_by_idx = {t.tenant_idx: t for t in base.tenants}
        cur_by_idx = {t.tenant_idx: t for t in session.tenants}
        rows = []
        for idx in sorted(cur_by_idx):
            cur, b = cur_by_idx[idx], base_by_idx.get(idx)
            if b is None:
                continue
            rows.append({
                "tenant_idx": idx,
                "ttft_solo_s": round(b.ttft_s, 4),
                "ttft_shared_s": round(cur.ttft_s, 4),
                "slowdown": round(cur.ttft_s / b.ttft_s, 2) if b.ttft_s > 0 else None,
                "tpot_solo_s": round(b.tpot_s, 4),
                "tpot_shared_s": round(cur.tpot_s, 4),
                "tpot_slowdown": round(cur.tpot_s / b.tpot_s, 2) if b.tpot_s > 0 else None,
            })
        baseline = rows

    report = {
        "session": str(session_dir),
        "meta": asdict(session.meta),
        "summary": {
            "num_tenants": len(session.tenants),
            "peak_concurrency": session.peak_concurrency(),
            "total_time_s": round(session.total_time_s(), 3),
            "mean_ttft_s": round(session.mean_ttft_s(), 4),
            "mean_tpot_s": round(session.mean_tpot_s(), 4),
            "aggregate_throughput_tok_s": round(session.aggregate_throughput_tok_s(), 2),
            "multi_tenant_steps": len(multi_steps),
            "total_steps": len(session.steps),
            "mean_expert_collision_ratio": (
                round(sum(collision_ratios) / len(collision_ratios), 4)
                if collision_ratios
                else 0.0
            ),
        },
        "tenants": tenants_report,
        "delay_vs_concurrency": {
            str(k): {"mean_ttft_s": round(v["ttft_sum"] / v["n"], 4), "n": v["n"]}
            for k, v in sorted(delay_vs_conc.items())
        },
        "baseline_comparison": baseline,
    }

    out_report = session_dir / "session_report.json"
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[analyze] Report → {out_report}")

    if plot:
        try:
            _plot_timeline(session, session_dir / "timeline.png")
        except ImportError:
            print("[analyze] matplotlib not available — skipping timeline plot")

    return report


def _plot_timeline(session: MultiTenantSession, out_path: Path) -> None:
    """Gantt-style tenant timeline + per-step concurrency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    tenants = sorted(session.tenants, key=lambda t: t.arrival_s)
    for i, t in enumerate(tenants):
        ax1.barh(i, t.finish_s - t.arrival_s, left=t.arrival_s, height=0.6,
                 color="#4C9AFF", alpha=0.8)
        if t.token_timestamps_s:
            xs = t.token_timestamps_s
            ax1.scatter(xs, [i] * len(xs), s=8, color="#FF5630", zorder=3)
    ax1.set_yticks(range(len(tenants)))
    ax1.set_yticklabels([t.request_id for t in tenants], fontsize=8)
    ax1.set_ylabel("tenant")
    ax1.invert_yaxis()

    xs = [s.t_s for s in session.steps]
    ys = [len(s.tokens) for s in session.steps]
    ax2.step(xs, ys, where="post", color="#36B37E")
    ax2.set_ylabel("tenants in step")
    ax2.set_xlabel("session time (s)")

    fig.suptitle(f"Multi-tenant serving — {session.meta.schedule} "
                 f"({session.meta.num_tenants} tenants, "
                 f"{session.meta.num_experts} experts, top_k={session.meta.top_k})")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[analyze] Timeline → {out_path}")
