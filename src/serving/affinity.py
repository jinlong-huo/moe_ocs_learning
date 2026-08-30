"""
affinity.py — Cross-tenant routing affinity analysis.

Quantifies how similar the expert routing of different tenants is — the
core evidence for OCS preset/affinity scheduling: if similar prompts
route to similar experts, circuits can be pre-established for a prompt
family instead of paying full all-to-all.

Metric definitions follow ``moe_ocs_learning/src/eval/affinity_consistency.py``:

  * top-k overlap    — per-(position, layer) expert-set match rate
  * Jaccard          — expert-set overlap per layer
  * JS divergence    — Jensen-Shannon distance of aggregate expert
                        distributions (0 = identical)
  * affinity corr    — Pearson R between expert co-activation matrices
  * plan hit-rate    — fraction of tenant B's routing cells fully covered
                        by tenant A's expert set (an OCS preset plan)
  * mass intersection — mean Σ min(p_a, p_b) over aligned cells (= 1 − TV
                        distance); a marginal flip costs its gate mass, ~0
  * EMD               — mean 1-D earth-mover's distance Σ|CDF_a − CDF_b|
                        over the expert-id axis; prices near-misses
                        proportionally instead of all-or-nothing
  * Bhattacharyya     — mean Σ √(p_a·p_b); expected routing-mass agreement
  * matched weight MAE / cosine — weight-vector fidelity on cells whose
                        expert sets match; separates "same experts" from
                        "same emphasis"
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from src.serving.schema import MultiTenantSession


# ═══════════════════════════════════════════════════════════════════
# Metric primitives (mirror moe_ocs_learning/src/eval/affinity_consistency.py)
# ═══════════════════════════════════════════════════════════════════

def expert_distribution(expert_ids_list, num_experts: int) -> np.ndarray:
    counts = np.zeros(num_experts, dtype=np.float64)
    for ids in expert_ids_list:
        for e in ids:
            counts[int(e)] += 1.0
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total



def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        a = np.maximum(a, 1e-12)
        b = np.maximum(b, 1e-12)
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def jaccard_of_sets(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def affinity_correlation(ca: np.ndarray, cb: np.ndarray) -> float:
    """Pearson R between two co-activation matrices (flattened)."""
    a = ca.ravel().astype(np.float64)
    b = cb.ravel().astype(np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if np.isnan(c) else c


# ═══════════════════════════════════════════════════════════════════
# Trace → arrays
# ═══════════════════════════════════════════════════════════════════

def _route_cells(trace):
    """Return list of per-(position, layer) expert cells:
    [(pos, token_id, layer, experts, weights)]."""
    cells = []
    for route in trace.routes:
        for lid, lr in route.layers.items():
            cells.append((
                route.token_pos, route.token_id, int(lid),
                list(lr.experts), list(lr.weights),
            ))
    return cells


def _route_array(trace, layers: list[str], top_k: int) -> np.ndarray:
    """Dense [max_pos+1, L, K] expert-id array; -1 pads missing cells."""
    cells = _route_cells(trace)
    if not cells:
        return np.zeros((0, len(layers), top_k), dtype=np.int64) - 1
    max_pos = max(c[0] for c in cells)
    arr = np.full((max_pos + 1, len(layers), top_k), -1, dtype=np.int64)
    for pos, _tid, lid, experts, _ in cells:
        l = layers.index(str(lid))
        for k, e in enumerate(experts[:top_k]):
            arr[pos, l, k] = int(e)
    return arr


def co_activation(trace, num_experts: int) -> np.ndarray:
    """Expert co-activation matrix [E, E]: how often two experts co-route."""
    mat = np.zeros((num_experts, num_experts), dtype=np.float64)
    for _, _, _, experts, _ in _route_cells(trace):
        for i, ea in enumerate(experts):
            for j, eb in enumerate(experts):
                mat[int(ea), int(eb)] += 1.0
    return mat


def used_expert_set(trace) -> set[int]:
    out: set[int] = set()
    for _, _, _, experts, _ in _route_cells(trace):
        out.update(int(e) for e in experts)
    return out


# ═══════════════════════════════════════════════════════════════════
# Weight-aware cell metrics
# ═══════════════════════════════════════════════════════════════════

def _clip_pairs(experts, weights, k: int):
    """First-k (expert, weight) pairs with valid expert ids, zipped."""
    pairs = [
        (int(e), float(w))
        for e, w in zip(list(experts)[:k], list(weights)[:k])
        if int(e) >= 0
    ]
    if not pairs:
        return [], []
    es, ws = zip(*pairs)
    return list(es), list(ws)


def _cell_dist(experts, weights, num_experts: int) -> np.ndarray:
    """Dense [E] routing-mass distribution for one cell.

    Selected experts carry their raw top-k softmax mass; the residual
    ``1 - Σw`` (the gate mass of all non-selected experts, recoverable
    because the logged weights are *un-renormalized* top-k softmax masses)
    is spread uniformly over the unselected experts — the maximally
    agnostic assumption when only top-k is logged.  Renormalized gates
    (``norm_topk_prob``, Σw ≈ 1) degrade gracefully to their own support.
    """
    p = np.zeros(num_experts, dtype=np.float64)
    mask = np.zeros(num_experts, dtype=bool)
    for e, w in zip(experts, weights):
        if 0 <= e < num_experts:
            p[e] += w
            mask[e] = True
    total = float(p.sum())
    if total <= 0.0:
        return np.full(num_experts, 1.0 / max(num_experts, 1))
    if total > 1.0:
        # Renormalized weights (or fp rounding): already a full distribution.
        return p / total
    n_unsel = int((~mask).sum())
    if n_unsel <= 0:
        return p / total
    p[~mask] += (1.0 - total) / n_unsel
    return p


def _mass_metrics(pa: np.ndarray, pb: np.ndarray) -> tuple[float, float, float]:
    """(histogram intersection, 1-D EMD, Bhattacharyya) of two cell dists.

    EMD uses the L1 ground distance on the expert-id axis — the standard
    closed form Σ|CDF_a − CDF_b|.  Expert-id distance is a namespace
    convention, not a semantic metric; it is kept because it prices a
    near-miss smoothly where set-Jaccard charges a total miss.
    """
    inter = float(np.minimum(pa, pb).sum())
    emd = float(np.abs(np.cumsum(pa) - np.cumsum(pb)).sum())
    bhatt = float(np.sqrt(np.maximum(pa * pb, 0.0)).sum())
    return inter, emd, bhatt


def _matched_weight_metrics(ea, wa, eb, wb) -> tuple[float, float]:
    """(MAE, cosine) of weight vectors aligned by shared expert id.

    Only meaningful on set-matched cells; caller guarantees the sets are
    equal and non-empty.
    """
    wa_by = dict(zip(ea, wa))
    wb_by = dict(zip(eb, wb))
    shared = sorted(set(wa_by) & set(wb_by))
    if not shared:
        return 0.0, 1.0
    va = np.asarray([wa_by[e] for e in shared], dtype=np.float64)
    vb = np.asarray([wb_by[e] for e in shared], dtype=np.float64)
    mae = float(np.abs(va - vb).mean())
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    cos = float(va @ vb / (na * nb)) if na > 1e-12 and nb > 1e-12 else 1.0
    return mae, cos


# ═══════════════════════════════════════════════════════════════════
# Pairwise metrics
# ═══════════════════════════════════════════════════════════════════

def pairwise_metrics(trace_a, trace_b, num_experts: int, layers: list[str],
                     top_k: int, k_compare: int | None = None,
                     weight_aware: bool = False) -> dict:
    """All affinity metrics between two traces.

    ``k_compare`` compares only the first k experts of each cell
    (e.g. top-(k-1)) — the marginal expert is numerically noisy on Metal.

    ``weight_aware=True`` additionally computes the mass-based metrics
    (intersection / EMD / Bhattacharyya per aligned cell, and weight-vector
    fidelity on set-matched cells).  The keys it adds are the ones that can
    answer "same experts AND same emphasis?" rather than set identity only.
    """
    k = top_k if k_compare is None else k_compare

    cells_a = _route_cells(trace_a)
    cells_b = _route_cells(trace_b)

    # ── per-cell overlap over aligned (pos, layer) ─────────────────
    key_a = {(pos, lid): (experts, weights) for pos, _t, lid, experts, weights in cells_a}
    key_b = {(pos, lid): (experts, weights) for pos, _t, lid, experts, weights in cells_b}
    common = set(key_a) & set(key_b)

    overlap_hits = 0
    jacc_sum = 0.0
    for key in common:
        ea = [e for e in key_a[key][0][:k] if e >= 0]
        eb = [e for e in key_b[key][0][:k] if e >= 0]
        if set(ea) == set(eb):
            overlap_hits += 1
        jacc_sum += jaccard_of_sets(set(ea), set(eb))
    n_cells = len(common) or 1
    topk_overlap = overlap_hits / n_cells
    mean_jaccard = jacc_sum / n_cells


    # ── same-token cells: (token_id, layer) present in both traces,
    #    regardless of position — how does the same token route in
    #    different prompt contexts? ───────────────────────────────── # this is the key verification point, since we want to examine the token cases, like here, we have the tokens different in different prompt then, so we at least give an identical one, for different models and within a certain noise level; apart from that, we might have the different prompt, like here we should also pay attention to some works that have investigated that whether the math prompt is different the coding expert then, if so we might have design a set of different experiments to figure this out then. 
    
    # And in the mean time, we should have different models to verify this case though. OFC the model should have only one settings, otherwise the exp makes no sense. 
    tok_key_a = {(tid, lid): (experts, weights) for _p, tid, lid, experts, weights in cells_a
                 if tid >= 0}
    tok_key_b = {(tid, lid): (experts, weights) for _p, tid, lid, experts, weights in cells_b
                 if tid >= 0}
    tok_common = set(tok_key_a) & set(tok_key_b)
    tok_hits = 0
    tok_jacc = 0.0
    for key in tok_common:
        ea = [e for e in tok_key_a[key][0][:k] if e >= 0]
        eb = [e for e in tok_key_b[key][0][:k] if e >= 0]
        if set(ea) == set(eb):
            tok_hits += 1
        tok_jacc += jaccard_of_sets(set(ea), set(eb))
    n_tok = len(tok_common) or 1
    same_token_overlap = tok_hits / n_tok
    same_token_jaccard = tok_jacc / n_tok

    # ── per-layer Jaccard on used-expert sets ──────────────────────
    per_layer_j = {}
    for lid in layers:
        ea = {e for pos, _t, l, experts, _ in cells_a if l == int(lid) for e in experts}
        eb = {e for pos, _t, l, experts, _ in cells_b if l == int(lid) for e in experts}
        per_layer_j[lid] = jaccard_of_sets(ea, eb)

    # ── aggregate distributions ────────────────────────────────────
    all_ids_a = [e for _, _, _, experts, _ in cells_a for e in experts]
    all_ids_b = [e for _, _, _, experts, _ in cells_b for e in experts]
    dist_a = expert_distribution([all_ids_a], num_experts)
    dist_b = expert_distribution([all_ids_b], num_experts)
    jsd = js_divergence(dist_a, dist_b)

    # ── affinity correlation ───────────────────────────────────────
    corr = affinity_correlation(
        co_activation(trace_a, num_experts), co_activation(trace_b, num_experts)
    )

    # ── OCS plan hit-rate: A's expert set used as the preset plan ──
    preset = used_expert_set(trace_a)
    hits = 0
    for _, _, _, experts, _ in cells_b:
        if set(int(e) for e in experts).issubset(preset):
            hits += 1
    plan_hit_rate = hits / (len(cells_b) or 1)

    result = {
        "cells_common": len(common),
        "tokens_common": len(tok_common),
        "topk_overlap": round(topk_overlap, 5),
        "mean_cell_jaccard": round(mean_jaccard, 5),
        "same_token_overlap": round(same_token_overlap, 5),
        "same_token_jaccard": round(same_token_jaccard, 5),
        "per_layer_jaccard_mean": round(float(np.mean(list(per_layer_j.values()))), 5),
        "js_divergence": round(jsd, 6),
        "affinity_correlation": round(corr, 5),
        "plan_hit_rate": round(plan_hit_rate, 5),
    }

    # ── weight-aware: compare routing mass, not just set identity ──
    if weight_aware:
        inter_sum = emd_sum = bhatt_sum = 0.0
        matched = 0
        mae_sum = cos_sum = 0.0
        for key in common:
            ea, wa = _clip_pairs(key_a[key][0], key_a[key][1], k)
            eb, wb = _clip_pairs(key_b[key][0], key_b[key][1], k)
            if not ea or not eb:
                continue
            pa = _cell_dist(ea, wa, num_experts)
            pb = _cell_dist(eb, wb, num_experts)
            inter, emd, bhatt = _mass_metrics(pa, pb)
            inter_sum += inter
            emd_sum += emd
            bhatt_sum += bhatt
            if set(ea) == set(eb):
                mae, cos = _matched_weight_metrics(ea, wa, eb, wb)
                mae_sum += mae
                cos_sum += cos
                matched += 1
        n_w = len(common) or 1
        result.update({
            "mean_cell_mass_intersection": round(inter_sum / n_w, 5),
            "mean_cell_emd": round(emd_sum / n_w, 5),
            "mean_cell_bhattacharyya": round(bhatt_sum / n_w, 5),
            "matched_cells": matched,
            "matched_weight_mae":
                round(mae_sum / matched, 5) if matched else None,
            "matched_weight_cosine":
                round(cos_sum / matched, 5) if matched else None,
        })

    return result


# ═══════════════════════════════════════════════════════════════════
# Noise floor from identical-prompt repeats
# ═══════════════════════════════════════════════════════════════════

def load_repeats(workload_dir: str | Path) -> list:
    """Load the role='repeat' traces from a capture_workload output dir.

    Repeats are identical prompts on the same backend (suite role
    ``rep.noise.*``), so any metric difference among them is pure backend
    measurement noise — quantized-GEMM nondeterminism and gate near-ties.
    """
    from src.data.routing_schema import RoutingTrace

    d = Path(workload_dir)
    with open(d / "manifest.json") as f:
        man = json.load(f)
    out = []
    for rec in man.get("records", []):
        if rec.get("role") == "repeat":
            out.append(RoutingTrace.load(d / rec["trace"]))
    return out


def repeat_noise_floor(traces: list, num_experts: int, top_k: int,
                       k_compare: int | None = None,
                       weight_aware: bool = True) -> dict:
    """Per-metric noise floor from pairwise metrics among repeat traces.

    Returns ``{"n_traces", "n_pairs", "metrics": {key: {mean, sd}}}``.
    A cross-backend observation should be reported as a z-score against
    the floor of the backend(s) it came from: an observation within the
    floor is indistinguishable from that backend's own nondeterminism.
    """
    if len(traces) < 2:
        return {"n_traces": len(traces), "n_pairs": 0, "metrics": {}}

    layers = sorted({
        lid
        for t in traces
        for r in t.routes
        for lid in r.layers
    })
    pairs = list(combinations(traces, 2))
    acc: dict[str, list[float]] = {}
    for a, b in pairs:
        m = pairwise_metrics(
            a, b, num_experts, layers, top_k,
            k_compare=k_compare, weight_aware=weight_aware,
        )
        for key, v in m.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                acc.setdefault(key, []).append(float(v))

    metrics = {}
    for key, xs in acc.items():
        arr = np.asarray(xs, dtype=np.float64)
        metrics[key] = {
            "mean": round(float(arr.mean()), 6),
            "sd": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 6),
        }
    return {"n_traces": len(traces), "n_pairs": len(pairs), "metrics": metrics}


def z_score(observed: float, floor: dict) -> float | None:
    """Standardised excess of ``observed`` over a repeat-noise floor."""
    sd = floor.get("sd", 0.0)
    if sd is None or sd <= 1e-12:
        return None
    return (observed - floor.get("mean", 0.0)) / sd


# ═══════════════════════════════════════════════════════════════════
# Session-level report
# ═══════════════════════════════════════════════════════════════════

def _load_traces(session_dir: Path, session: MultiTenantSession):
    from src.data.routing_schema import RoutingTrace

    out = {}
    for t in session.tenants:
        out[t.request_id] = RoutingTrace.load(str(session_dir / t.trace_path))
    return out


def affinity_report(session_dir: str | Path, plot: bool = False) -> dict:
    """Pairwise affinity matrix + (for family=similar) edit-distance curve."""
    session_dir = Path(session_dir)
    session = MultiTenantSession.load(str(session_dir / "session.json"))
    traces = _load_traces(session_dir, session)

    meta = session.meta
    num_experts = meta.num_experts
    top_k = meta.top_k

    # Layer set from the first trace with routes.
    layers: list[str] = []
    for t in traces.values():
        for route in t.routes:
            layers = sorted(route.layers.keys())
            break
        if layers:
            break
    if not layers:
        raise ValueError("No routing layers found in traces")

    ids = sorted(traces.keys())
    pairs = []
    matrix = {}
    for i, a in enumerate(ids):
        row = {}
        for j, b in enumerate(ids):
            if i == j:
                row[b] = None
                continue
            if j < i:
                row[b] = matrix[b][a]
                continue
            m_full = pairwise_metrics(traces[a], traces[b], num_experts, layers, top_k)
            m_core = pairwise_metrics(
                traces[a], traces[b], num_experts, layers, top_k, k_compare=max(1, top_k - 1)
            )
            rec = {"full_topk": m_full, "core_topk_minus_1": m_core}
            row[b] = rec
            pairs.append({
                "a": a, "b": b,
                "slots_a": session.by_request_id()[a].slots_changed,
                "slots_b": session.by_request_id()[b].slots_changed,
                "edit_distance": abs(
                    session.by_request_id()[a].slots_changed
                    - session.by_request_id()[b].slots_changed
                ),
                **m_full,
            })
        matrix[a] = row

    # ── edit-distance curve (similar family) ───────────────────────
    curve = {}
    if session.meta.family == "similar":
        by_dist: dict[int, list[dict]] = {}
        for p in pairs:
            by_dist.setdefault(p["edit_distance"], []).append(p)
        for d in sorted(by_dist):
            rows = by_dist[d]
            curve[str(d)] = {
                "n_pairs": len(rows),
                "topk_overlap": round(float(np.mean([r["topk_overlap"] for r in rows])), 5),
                "same_token_overlap": round(float(np.mean([r["same_token_overlap"] for r in rows])), 5),
                "mean_cell_jaccard": round(float(np.mean([r["mean_cell_jaccard"] for r in rows])), 5),
                "js_divergence": round(float(np.mean([r["js_divergence"] for r in rows])), 6),
                "affinity_correlation": round(float(np.mean([r["affinity_correlation"] for r in rows])), 5),
                "plan_hit_rate": round(float(np.mean([r["plan_hit_rate"] for r in rows])), 5),
            }

    report = {
        "session": str(session_dir),
        "meta": {
            "model_id": meta.model_id,
            "num_experts": num_experts,
            "top_k": top_k,
            "num_tenants": meta.num_tenants,
            "family": meta.family,
            "greedy": meta.temperature == 0.0,
            "prefix_caching": meta.prefix_caching,
        },
        "pairs": pairs,
        "edit_distance_curve": curve,
    }

    out_path = session_dir / "affinity_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[affinity] Report → {out_path}")

    if plot:
        try:
            _plot_affinity(session, session_dir, ids, matrix, top_k)
        except ImportError:
            print("[affinity] matplotlib not available — skipping plot")

    return report


def _plot_affinity(session, session_dir: Path, ids, matrix, top_k) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(ids)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def _grid(metric_path):
        g = np.eye(n)
        for i, a in enumerate(ids):
            for j, b in enumerate(ids):
                if i == j:
                    continue
                rec = matrix[a][b]
                v = rec
                for part in metric_path:
                    v = v[part]
                g[i, j] = v
        return g

    g1 = _grid(("full_topk", "topk_overlap"))
    im = axes[0].imshow(g1, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title(f"top-{top_k} expert-set overlap")
    axes[0].set_xticks(range(n), ids, rotation=45, fontsize=7)
    axes[0].set_yticks(range(n), ids, fontsize=7)
    fig.colorbar(im, ax=axes[0])

    g2 = _grid(("full_topk", "affinity_correlation"))
    im = axes[1].imshow(g2, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("co-activation affinity correlation")
    axes[1].set_xticks(range(n), ids, rotation=45, fontsize=7)
    axes[1].set_yticks(range(n), ids, fontsize=7)
    fig.colorbar(im, ax=axes[1])

    fig.suptitle(f"Routing affinity — {session.meta.model_id} "
                 f"(family={session.meta.family}, greedy={session.meta.temperature == 0.0})")
    fig.tight_layout()
    out = session_dir / "affinity.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"[affinity] Plot → {out}")
