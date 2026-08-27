"""
affinity_graph.py — expert affinity graphs and whether they contain structure.

The question this module answers
────────────────────────────────
An expert affinity graph is only useful if it says something that the expert
*load* vector does not already say.  Popular experts co-occur often purely
because they are popular; a raw co-activation matrix therefore always "looks
clustered".  Every affinity number here is consequently reported against a
**load-preserving null model**:

    null: for each routing cell, resample K experts without replacement from
          that layer's empirical load distribution.

The null keeps the per-layer marginal load exactly (in expectation) and
destroys only the *joint* structure.  Any affinity signal that survives the
comparison is genuine co-selection structure; anything that does not is a
restatement of load skew and must not be sold as affinity.

Affinity definitions
────────────────────
``cooccurrence``  C_ij = # cells whose top-k contains both i and j.
                  This is the pairwise relaxation of the fan-out objective
                  (see ``cost_model.py``) and is the only definition with a
                  direct communication interpretation, so it is the default.
``conditional``   A_ij = P(j in topk | i in topk)  (row-normalised C)
``pmi``           A_ij = log2 P(i,j) / (P(i)P(j))  — popularity-corrected
``jaccard``       A_ij = |cells(i) ∩ cells(j)| / |cells(i) ∪ cells(j)|
``cosine``        cosine similarity of expert indicator vectors over cells
``weighted``      co-occurrence weighted by min(gate_i, gate_j)

Scope
─────
``per_layer=True`` builds one graph per MoE layer.  This is the correct scope:
expert ids are per-layer namespaces (see ``trace_ir``), and a per-layer graph
is what a per-layer placement consumes.  ``per_layer=False`` pools layers and
is provided only to quantify how much signal pooling destroys.
"""

from __future__ import annotations

import numpy as np

from src.eval.trace_ir import CellTable


# ═══════════════════════════════════════════════════════════════════════
# Raw co-occurrence
# ═══════════════════════════════════════════════════════════════════════

def cooccurrence(experts: np.ndarray, num_experts: int,
                 weights: np.ndarray | None = None,
                 include_diagonal: bool = False) -> np.ndarray:
    """[E, E] symmetric co-selection counts from an [N, K] expert array.

    Vectorised: builds the sparse indicator once and takes one gram product,
    instead of the O(N*K^2) Python triple loop the previous implementation used.
    """
    n, k = experts.shape
    if n == 0:
        return np.zeros((num_experts, num_experts))
    ind = np.zeros((n, num_experts), dtype=np.float64)
    rows = np.repeat(np.arange(n), k)
    cols = experts.ravel()
    if weights is None:
        np.add.at(ind, (rows, cols), 1.0)
    else:
        np.add.at(ind, (rows, cols), weights.ravel().astype(np.float64))
    C = ind.T @ ind
    if not include_diagonal:
        np.fill_diagonal(C, 0.0)
    return C


def _indicator(experts: np.ndarray, num_experts: int) -> np.ndarray:
    n, k = experts.shape
    ind = np.zeros((n, num_experts), dtype=np.float64)
    ind[np.repeat(np.arange(n), k), experts.ravel()] = 1.0
    return ind


def affinity_matrix(experts: np.ndarray, num_experts: int, kind: str = "cooccurrence",
                    weights: np.ndarray | None = None) -> np.ndarray:
    """One affinity graph under the requested definition."""
    n = experts.shape[0]
    if n == 0:
        return np.zeros((num_experts, num_experts))

    if kind == "cooccurrence":
        return cooccurrence(experts, num_experts)

    if kind == "weighted":
        # min(gate_i, gate_j) is the natural weight: the pair only matters as
        # much as its weaker member contributes to the token's output.
        ind = _indicator(experts, num_experts)
        wm = np.zeros_like(ind)
        wm[np.repeat(np.arange(n), experts.shape[1]), experts.ravel()] = (
            weights.ravel() if weights is not None else 1.0)
        C = np.minimum(wm[:, :, None], wm[:, None, :]).sum(0) if num_experts <= 96 \
            else (wm.T @ ind) * 0.5 + (ind.T @ wm) * 0.5
        np.fill_diagonal(C, 0.0)
        return C

    C = cooccurrence(experts, num_experts)
    load = np.bincount(experts.ravel(), minlength=num_experts).astype(np.float64)

    if kind == "conditional":
        return C / np.maximum(load[:, None], 1e-12)

    if kind == "pmi":
        tot = C.sum()
        if tot <= 0:
            return np.zeros_like(C)
        pij = C / tot
        pi = load / max(load.sum(), 1e-12)
        denom = np.maximum(pi[:, None] * pi[None, :], 1e-18)
        with np.errstate(divide="ignore", invalid="ignore"):
            M = np.log2(np.maximum(pij, 1e-18) / denom)
        M[pij <= 0] = 0.0
        np.fill_diagonal(M, 0.0)
        return M

    if kind == "jaccard":
        inter = C
        union = load[:, None] + load[None, :] - inter
        J = inter / np.maximum(union, 1e-12)
        np.fill_diagonal(J, 0.0)
        return J

    if kind == "cosine":
        norm = np.sqrt(np.maximum(load, 1e-12))
        S = C / (norm[:, None] * norm[None, :])
        np.fill_diagonal(S, 0.0)
        return S

    raise ValueError(f"unknown affinity kind {kind!r}")


KINDS = ("cooccurrence", "conditional", "pmi", "jaccard", "cosine", "weighted")


# ═══════════════════════════════════════════════════════════════════════
# Per-layer graph stack
# ═══════════════════════════════════════════════════════════════════════

def layer_affinities(t: CellTable, kind: str = "cooccurrence"
                     ) -> dict[int, np.ndarray]:
    """{layer -> [E, E] affinity} — one graph per MoE layer."""
    out = {}
    for l in t.layers:
        m = t.layer == l
        out[int(l)] = affinity_matrix(t.experts[m], t.num_experts, kind,
                                      t.weights[m])
    return out


def pooled_affinity(t: CellTable, kind: str = "cooccurrence") -> np.ndarray:
    """Layer-pooled affinity — the (invalid for statistics, valid only for a
    layer-shared placement) single graph."""
    return affinity_matrix(t.experts, t.num_experts, kind, t.weights)


# ═══════════════════════════════════════════════════════════════════════
# Load-preserving null model
# ═══════════════════════════════════════════════════════════════════════

def null_experts(t: CellTable, rng: np.random.Generator) -> np.ndarray:
    """Resample every cell's top-k from its own layer's empirical load.

    Sampling is *without replacement inside a cell* (a token cannot select the
    same expert twice), which is what makes this a fair null for the fan-out
    objective.  Implemented with the Gumbel top-k trick so the whole layer is
    resampled in one vectorised shot.
    """
    out = np.empty_like(t.experts)
    K = t.top_k
    for l in t.layers:
        m = t.layer == l
        n = int(m.sum())
        if n == 0:
            continue
        load = np.bincount(t.experts[m].ravel(), minlength=t.num_experts).astype(np.float64)
        p = load / max(load.sum(), 1e-12)
        logp = np.log(np.maximum(p, 1e-300))
        g = rng.gumbel(size=(n, t.num_experts))
        out[m] = np.argpartition(-(logp[None, :] + g), K - 1, axis=1)[:, :K]
    return out


def null_table(t: CellTable, rng: np.random.Generator) -> CellTable:
    """A ``CellTable`` twin with identical labels/marginals, destroyed joints."""
    nt = t.select(np.ones(t.n_cells, dtype=bool))
    nt.experts = null_experts(t, rng)
    return nt


# ═══════════════════════════════════════════════════════════════════════
# Does the graph carry structure beyond load?
# ═══════════════════════════════════════════════════════════════════════

def concentration(A: np.ndarray, frac: float = 0.01) -> float:
    """Share of total off-diagonal affinity held by the top ``frac`` of edges.

    A load-only graph spreads mass broadly; genuine clustering concentrates it.
    """
    off = A[~np.eye(A.shape[0], dtype=bool)]
    off = off[off > 0]
    if off.size == 0:
        return 0.0
    k = max(1, int(round(frac * off.size)))
    return float(np.sort(off)[::-1][:k].sum() / off.sum())


def structure_test(t: CellTable, kind: str = "cooccurrence", n_null: int = 20,
                   seed: int = 0, per_layer: bool = True) -> dict:
    """Compare observed affinity concentration against the load-preserving null.

    ``z`` is the standardised excess concentration.  |z| < 2 means the affinity
    graph says nothing the load vector did not already say — in which case
    affinity-aware anything is not justified on this workload.
    """
    rng = np.random.default_rng(seed)

    def conc(tab):
        if per_layer:
            vals = [concentration(A) for A in layer_affinities(tab, kind).values()]
            return float(np.mean(vals)) if vals else 0.0
        return concentration(pooled_affinity(tab, kind))

    obs = conc(t)
    nulls = np.array([conc(null_table(t, rng)) for _ in range(n_null)])
    mu, sd = float(nulls.mean()), float(nulls.std(ddof=1) if n_null > 1 else 0.0)
    return {
        "kind": kind, "per_layer": per_layer,
        "observed_concentration": round(obs, 6),
        "null_mean": round(mu, 6), "null_sd": round(sd, 6), "n_null": n_null,
        "z": round((obs - mu) / sd, 3) if sd > 1e-12 else None,
        "excess_ratio": round(obs / mu, 4) if mu > 1e-12 else None,
        "significant": bool(sd > 1e-12 and abs((obs - mu) / sd) > 2.0),
    }


# ═══════════════════════════════════════════════════════════════════════
# Graph-to-graph similarity (temporal / cross-workload stability)
# ═══════════════════════════════════════════════════════════════════════

def graph_similarity(A: np.ndarray, B: np.ndarray) -> dict:
    """How alike are two affinity graphs?

    Reports three complementary views because they disagree in informative
    ways: Pearson on all edges is dominated by the many near-zero edges,
    Spearman is rank-robust, and top-edge Jaccard is what a circuit planner
    actually consumes (it only ever provisions the strongest edges).
    """
    off = ~np.eye(A.shape[0], dtype=bool)
    a, b = A[off].astype(np.float64), B[off].astype(np.float64)
    res = {}
    if a.std() > 1e-12 and b.std() > 1e-12:
        res["pearson"] = float(np.corrcoef(a, b)[0, 1])
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        res["spearman"] = float(np.corrcoef(ra, rb)[0, 1])
    else:
        res["pearson"] = res["spearman"] = 0.0
    for frac in (0.01, 0.05):
        k = max(1, int(round(frac * a.size)))
        ta = set(np.argsort(a)[::-1][:k].tolist())
        tb = set(np.argsort(b)[::-1][:k].tolist())
        res[f"top{int(frac*100)}pct_jaccard"] = len(ta & tb) / len(ta | tb)
    res["cosine"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))
    return {k: round(v, 6) for k, v in res.items()}


def per_layer_graph_similarity(ta: CellTable, tb: CellTable,
                               kind: str = "cooccurrence") -> dict:
    """Mean per-layer graph similarity between two workload slices."""
    Aa, Ab = layer_affinities(ta, kind), layer_affinities(tb, kind)
    common = sorted(set(Aa) & set(Ab))
    if not common:
        return {"n_layers": 0}
    rows = [graph_similarity(Aa[l], Ab[l]) for l in common]
    keys = rows[0].keys()
    out = {f"mean_{k}": round(float(np.mean([r[k] for r in rows])), 6) for k in keys}
    out["n_layers"] = len(common)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Routing-signature similarity (workload level, not expert-pair level)
# ═══════════════════════════════════════════════════════════════════════

def signature_similarity_matrix(t: CellTable, normalize: bool = True
                                ) -> tuple[np.ndarray, list]:
    """Cosine similarity between per-run per-layer routing signatures.

    Uses the per-layer-concatenated signature so expert-id namespaces are not
    mixed.  This is the object that tests "similar workloads route alike".
    """
    S, infos = t.run_signatures(normalize)
    if S.shape[0] == 0:
        return np.zeros((0, 0)), []
    Sn = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-18)
    return Sn @ Sn.T, infos


def within_between(M: np.ndarray, labels: list[str]) -> dict:
    """Within-group vs between-group similarity, with a permutation p-value.

    The permutation test shuffles the group labels, which is the only honest
    null here: raw within-group similarity is meaningless without knowing what
    an arbitrary grouping of the same runs would have produced.
    """
    n = M.shape[0]
    if n < 4:
        return {"n": n, "insufficient": True}
    lab = np.asarray(labels)
    iu = np.triu_indices(n, 1)
    same = lab[iu[0]] == lab[iu[1]]
    vals = M[iu]
    if same.sum() == 0 or (~same).sum() == 0:
        return {"n": n, "insufficient": True}
    w, b = float(vals[same].mean()), float(vals[~same].mean())
    pooled = np.sqrt((vals[same].var() + vals[~same].var()) / 2) + 1e-18

    rng = np.random.default_rng(0)
    n_perm = 2000
    obs = w - b
    cnt = 0
    for _ in range(n_perm):
        pl = rng.permutation(lab)
        s = pl[iu[0]] == pl[iu[1]]
        if s.sum() == 0 or (~s).sum() == 0:
            continue
        if (vals[s].mean() - vals[~s].mean()) >= obs:
            cnt += 1
    return {
        "n": n, "n_within_pairs": int(same.sum()), "n_between_pairs": int((~same).sum()),
        "within_mean": round(w, 6), "between_mean": round(b, 6),
        "gap": round(obs, 6), "cohens_d": round(obs / pooled, 4),
        "perm_p": round((cnt + 1) / (n_perm + 1), 5),
        "significant": bool((cnt + 1) / (n_perm + 1) < 0.05),
    }
