#!/usr/bin/env python3
"""EPS vs OCS comparison — the authentic, field-standard cost models.

Runs three configs on the SAME 3-tier fabric (2 pods × 2 nodes × 1 rank,
real Qwen weights + captured routing) and reports a directly comparable
table:

  - EPS baseline   : pure electrical packet switching (tier-aware fabric cost)
  - OCS alpha model: T_ocs = T_eps + T_reconfig × N_switches, T_reconfig ≈ 1 us
                     (fast switch class: SOA / ring-resonator)
  - OCS beta model : T_ocs = T_eps + T_reconfig × N_switches, T_reconfig ≈ 50 us
                     (MEMS beam-steering class)

Both OCS models use the fixed-delay cost model (no LRU circuit cache, no
eviction) so every number is directly comparable with the EPS baseline.

Usage:
    python3 scripts/compare_ocs_models.py
    python3 scripts/compare_ocs_models.py --output outputs/ocs_model_comparison.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent

MODELS = [
    {
        "name": "eps_baseline",
        "label": "EPS baseline",
        "config": "configs/qwen_eps_baseline.yaml",
    },
    {
        "name": "ocs_alpha",
        "label": "OCS alpha (T_reconfig=1us, fast switch)",
        "config": "configs/ocs_alpha_model.yaml",
    },
    {
        "name": "ocs_beta",
        "label": "OCS beta  (T_reconfig=50us, MEMS)",
        "config": "configs/ocs_beta_model.yaml",
    },
]


def run_model(name: str, config: str, trace_dir: Path, verbose: bool = False) -> dict:
    """Run one config via the launcher and return its parsed metrics."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.launcher",
        "--config", str(_repo_root / config),
        "--trace-dir", str(trace_dir),
    ]
    print(f"[compare] running {name} ({config}) ...", flush=True)
    proc = subprocess.run(
        cmd, cwd=_repo_root, capture_output=not verbose, text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:] if not verbose else "")
        print(proc.stderr[-2000:])
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")

    rank0 = trace_dir / "rank_00_trace.json"
    if not rank0.exists():
        raise RuntimeError(f"{name}: missing {rank0}")
    with open(rank0) as f:
        payload = json.load(f)

    events = payload.get("traceEvents", [])
    comm_us = sum(e.get("dur", 0) for e in events if e.get("name", "").startswith("comm"))
    route_us = sum(e.get("dur", 0) for e in events if e.get("name", "").startswith("step"))
    total_event_us = sum(e.get("dur", 0) for e in events)

    meta = payload.get("_metadata", {})
    ocs_meta = meta.get("ocs", {})
    ocs_metrics = ocs_meta.get("metrics", {}) if isinstance(ocs_meta, dict) else {}

    return {
        "name": name,
        "label": next(m["label"] for m in MODELS if m["name"] == name),
        "config": config,
        "total_event_us": total_event_us,
        "comm_us": comm_us,
        "route_us": route_us,
        "ocs_enabled": bool(ocs_meta.get("enabled", False)),
        "ocs_cost_model": ocs_meta.get("cost_model", "lru")
        if isinstance(ocs_meta, dict) else "lru",
        "ocs_circuit_budget": ocs_meta.get("max_circuits", None)
        if isinstance(ocs_meta, dict) else None,
        "ocs_reconfig_total_us": ocs_metrics.get("total_reconfig_time_us", 0.0),
        "ocs_establishes": ocs_metrics.get("circuit_establishes", 0),
        "ocs_reuses": ocs_metrics.get("circuit_reuses", 0),
        "ocs_evictions": ocs_metrics.get("circuit_evictions", 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="EPS vs fixed-delay OCS comparison")
    ap.add_argument("--output", default="outputs/ocs_model_comparison.json")
    ap.add_argument("--trace-root", default="outputs/ocs_model_comparison_traces")
    ap.add_argument("--verbose", action="store_true",
                    help="Stream launcher output instead of capturing it")
    args = ap.parse_args()

    results = []
    for model in MODELS:
        trace_dir = Path(args.trace_root) / model["name"]
        results.append(run_model(model["name"], model["config"], trace_dir, args.verbose))

    eps = results[0]
    print("\n" + "=" * 78)
    print("EPS vs OCS — fixed-delay cost models (real Qwen weights, 2x2x1 fabric)")
    print("=" * 78)
    print(f"{'model':<38s} {'comm_us':>12s} {'reconfig_us':>12s} "
          f"{'budget':>7s} {'est':>5s} {'reuse':>6s} {'reassign':>9s}")
    for r in results:
        budget = "-" if r["ocs_circuit_budget"] is None else str(r["ocs_circuit_budget"])
        print(f"{r['label']:<38s} {r['comm_us']:>12.1f} "
              f"{r['ocs_reconfig_total_us']:>12.1f} "
              f"{budget:>7s} {r['ocs_establishes']:>5d} {r['ocs_reuses']:>6d} "
              f"{r['ocs_evictions']:>9d}")

    print("\nDeltas vs EPS baseline:")
    for r in results[1:]:
        delta = r["comm_us"] - eps["comm_us"]
        extra = r["ocs_reconfig_total_us"]
        print(f"  {r['name']:<12s}: comm {delta:>+10.1f} us vs EPS "
              f"(reconfig {extra:.1f} us over {r['ocs_establishes']} switches, "
              f"{r['ocs_evictions']} port reassignments)")

    report = {
        "experiment": "eps_vs_ocs_fixed_delay",
        "fabric": "2 pods x 2 nodes x 1 rank; NVLink 1us/900GB/s, "
                  "NDR 3us/400Gb/s, core 10us/200Gb/s",
        "cost_models": {
            "eps": "tier-aware electrical packet switching",
            "ocs": "T_ocs = T_eps + T_reconfig x N_switches (fixed delay, "
                   "per-rank circuit budget with FIFO port reassignment)",
            "alpha": "T_reconfig = 1 us, full fan-out WSS (SOA / ring class)",
            "beta": "T_reconfig = 50 us, single-port MEMS beam-steering",
        },
        "models": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[compare] report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
