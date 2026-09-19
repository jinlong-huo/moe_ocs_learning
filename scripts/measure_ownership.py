#!/usr/bin/env python3
"""measure_ownership.py — replace an invented input with a measured one.

THE GAP
───────
```docs/assumptions.md``` A1 establishes that *routing* is a pure function of
(inputs, weights): a token hits the same experts regardless of topology, node
distribution or engine.  Every claim in this repo leans on it.

But the quantity an OCS is priced against is not the expert map — it is the
**rank-pair byte matrix**, and that needs a second fact A1 does not supply:
*which rank owns the token in the first place*.  ```cost_model.token_rank```
answers it with

    src = (run * 1_000_003 + pos) % n_dp

a synthetic hash of (sequence, position) that is uniform over DP ranks by
construction.  It is never measured, never cited, and no gate covers it — yet it
swings the OCS answer by ~148x (docs/ocs_moe_program.md §4.2).

This script measures what is actually on disk: two captured **multi-tenant
serving sessions** (concurrent co-batched requests, real vLLM backend, real
Qwen3.6 routing).  From them it derives the one property the hash gets wrong —
the **source support**: how many distinct owning ranks are live at the same
instant — and reports it against the hash's assumption.

WHAT IS MEASURED, AND WHAT REMAINS ASSUMED
──────────────────────────────────────────
Measured:   which sequences are co-resident in each serving step, how many
            tokens each contributes, the arrival order, and the per-token
            per-layer expert routing of every co-batched sequence.
Assumed:    how a deployment maps those live sequences onto DP replicas.  The
            session does not record replicas, so the script reports the
            **support** (number of live sequences = upper bound on live owning
            ranks) rather than claiming a replica assignment it cannot see.

Usage
─────
    python3.12 scripts/measure_ownership.py \
        --sessions logs/multi_tenant --world-size 32 \
        --out outputs/ownership/multi_tenant_ownership.json
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────

def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0 = perfectly even)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or x.sum() <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * x.sum()) - (n + 1.0) / n)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def load_run(run_dir: Path) -> dict:
    """One captured multi-tenant serving session + its per-tenant traces."""
    sess = json.loads((run_dir / "session.json").read_text())
    traces = {}
    tdir = run_dir / "traces"
    if tdir.is_dir():
        for f in sorted(tdir.glob("*.json")):
            d = json.loads(f.read_text())
            traces[f.stem] = d
    return {"dir": run_dir, "session": sess, "traces": traces}


def token_positions_per_step(session: dict, tenant: str) -> list[tuple[int, int]]:
    """Map each step to the [lo, hi) token-position range the tenant fed it.

    Tokens within a sequence are processed in order, so the cumulative count of
    a tenant's tokens across steps is exactly its token-position timeline.  The
    prefill step carries the whole prompt at once, which the cumsum reproduces.
    """
    out, cur = [], 0
    for s in session["steps"]:
        n = int(s["tokens"].get(tenant, 0))
        out.append((cur, cur + n))
        cur += n
    return out


def expert_sets(trace: dict, layer: str, lo: int, hi: int) -> set:
    """Union of the top-K expert sets of a tenant's tokens over [lo, hi)."""
    acc: set = set()
    for r in trace.get("routes", [])[lo:hi]:
        cell = r.get("layers", {}).get(layer)
        if cell:
            acc.update(int(e) for e in cell.get("experts", []))
    return acc


# ── the measurement ──────────────────────────────────────────────────────

def measure_run(loaded: dict, world_size: int, n_dp: int,
                probe_layers: tuple[str, ...]) -> dict:
    sess = loaded["session"]
    tenants = [t["request_id"] for t in sess["tenants"]]
    order = sorted(sess["tenants"], key=lambda t: t.get("arrival_s", 0.0))
    steps = sess["steps"]

    # ── concurrency: how many sequences are co-resident per step ──────────
    live_counts = np.array([len(s["tokens"]) for s in steps], dtype=np.int64)
    live_hist: dict[int, int] = {}
    for c in live_counts:
        live_hist[int(c)] = live_hist.get(int(c), 0) + 1

    # ── source support: distinct owning ranks live at the same instant ────
    # One sequence per replica (per-sequence ownership) is the standard
    # continuous-batching layout; the measured bound is #live sequences.
    order_index = {t["request_id"]: i for i, t in enumerate(order)}
    replica_of = {tid: (order_index[tid] % max(n_dp, 1)) for tid in tenants}

    support_per_step, tokens_per_step = [], []
    for s in steps:
        reps = {replica_of[t] for t in s["tokens"]}
        support_per_step.append(len(reps))
        tokens_per_step.append(int(sum(s["tokens"].values())))

    support = np.array(support_per_step, dtype=np.int64)
    tok_step = np.array(tokens_per_step, dtype=np.int64)

    # ── per-replica token totals: measured layout vs the hash ─────────────
    total_tokens = int(sum(sum(s["tokens"].values()) for s in steps))
    per_replica_measured = np.zeros(max(n_dp, 1), dtype=np.int64)
    for tid, idx in order_index.items():
        n = int(sum(s["tokens"].get(tid, 0) for s in steps))
        per_replica_measured[idx % max(n_dp, 1)] += n

    per_replica_hash = np.full(max(n_dp, 1), total_tokens / max(n_dp, 1))

    # ── same-replica share: token pairs that never cross a rank boundary ──
    same_share_steps = []
    for s in steps:
        counts = np.array(list(s["tokens"].values()), dtype=np.float64)
        tot = counts.sum()
        if tot <= 0:
            continue
        same = float((counts * (counts - 1.0)).sum())
        same_share_steps.append(same / (tot * (tot - 1.0)) if tot > 1 else 1.0)

    # ── do co-batched sequences hit the same experts? ─────────────────────
    overlap: dict[str, list[float]] = {lay: [] for lay in probe_layers}
    pos = {tid: token_positions_per_step(sess, tid) for tid in tenants}
    for si, s in enumerate(steps):
        act = [t for t in s["tokens"] if t in loaded["traces"]]
        if len(act) < 2:
            continue
        for lay in probe_layers:
            sets = {}
            for tid in act:
                lo, hi = pos[tid][si]
                sets[tid] = expert_sets(loaded["traces"][tid], lay, lo, hi)
            for a, b in combinations(act, 2):
                if sets[a] and sets[b]:
                    overlap[lay].append(jaccard(sets[a], sets[b]))

    return {
        "run": loaded["dir"].name,
        "model": sess["meta"].get("model_id"),
        "backend": sess["meta"].get("backend"),
        "num_experts": sess["meta"].get("num_experts"),
        "top_k": sess["meta"].get("top_k"),
        "num_layers": sess["meta"].get("num_layers"),
        "num_tenants": len(tenants),
        "steps": len(steps),
        "total_tokens": total_tokens,
        "concurrency_hist": {str(k): v for k, v in sorted(live_hist.items())},
        "concurrency_mean": float(live_counts.mean()),
        "concurrency_max": int(live_counts.max()),
        "source_support_measured_mean": float(support.mean()),
        "source_support_measured_max": int(support.max()),
        "source_support_hash": int(n_dp),
        "tokens_per_step_mean": float(tok_step.mean()),
        "tokens_per_step_max": int(tok_step.max()),
        "gini_tokens_per_replica_measured": round(gini(per_replica_measured), 6),
        "gini_tokens_per_replica_hash": round(gini(per_replica_hash), 6),
        "same_replica_token_share_measured": (
            round(float(np.mean(same_share_steps)), 6) if same_share_steps else None),
        "same_replica_token_share_hash": round(1.0 / max(n_dp, 1), 6),
        "expert_overlap_same_step": {
            lay: {
                "n_pairs": len(v),
                "mean_jaccard": round(float(np.mean(v)), 6) if v else None,
            } for lay, v in overlap.items()
        },
    }


# ── reporting ────────────────────────────────────────────────────────────

def report(rows: list[dict], world_size: int, n_dp: int) -> None:
    print(f"== measured ownership from multi-tenant serving sessions "
          f"(world_size={world_size}, n_dp={n_dp})")
    for r in rows:
        print(f"\n-- {r['run']}  ({r['num_tenants']} tenants, {r['steps']} steps, "
              f"{r['total_tokens']} tokens, {r['model']})")
        print(f"   concurrency per step      : {r['concurrency_hist']} "
              f"(mean {r['concurrency_mean']:.2f}, max {r['concurrency_max']})")
        print(f"   live owning ranks (measured): max {r['source_support_measured_max']}, "
              f"mean {r['source_support_measured_mean']:.2f}")
        print(f"   live owning ranks (hash)    : {r['source_support_hash']} "
              f"<- what token_rank() assumes")
        print(f"   tokens per replica Gini   : measured "
              f"{r['gini_tokens_per_replica_measured']:.4f} vs hash "
              f"{r['gini_tokens_per_replica_hash']:.4f}")
        print(f"   same-replica token share  : measured "
              f"{r['same_replica_token_share_measured']:.4f} vs hash "
              f"{r['same_replica_token_share_hash']:.4f}")
        for lay, v in r["expert_overlap_same_step"].items():
            if v["mean_jaccard"] is not None:
                print(f"   expert-set Jaccard, co-batched, layer {lay:<3}: "
                      f"{v['mean_jaccard']:.4f} over {v['n_pairs']} pairs")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=Path, default=Path("logs/multi_tenant"))
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--n-dp", type=int, default=None)
    ap.add_argument("--probe-layers", default="0,20,39")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/ownership/multi_tenant_ownership.json"))
    args = ap.parse_args(argv)

    W = args.world_size
    n_dp = args.n_dp or W
    probe = tuple(x.strip() for x in args.probe_layers.split(",") if x.strip())

    run_dirs = sorted(d for d in args.sessions.glob("run_*") if (d / "session.json").is_file())
    if not run_dirs:
        print(f"error: no session.json under {args.sessions}", file=sys.stderr)
        return 1

    rows = [measure_run(load_run(d), W, n_dp, probe) for d in run_dirs]
    report(rows, W, n_dp)

    doc = {
        "source": "logs/multi_tenant/*/session.json + traces/tenant-*.json",
        "world_size": W,
        "n_dp": n_dp,
        "measured": ["co-batched sequences per step", "tokens each contributes",
                     "arrival order", "per-token per-layer expert routing"],
        "assumed": ["how live sequences map onto DP replicas (support is measured, "
                    "the assignment is not recorded by the capture)"],
        "runs": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1))
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
