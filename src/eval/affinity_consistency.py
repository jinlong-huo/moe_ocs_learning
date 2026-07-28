"""Affinity consistency analysis: train vs inference routing correlation.

Quantifies how well training-time expert co-activation patterns predict
inference-time routing behavior. This is the core validation for the
OCS pre-configuration hypothesis: if training affinity is a good predictor
of inference routing, pre-configuring OCS circuits from training data
should achieve high hit rates during inference.

Metrics:
  - JS divergence: per-layer Jensen-Shannon divergence of expert distributions
  - Top-K overlap: fraction of top-K expert assignments that match
  - Jaccard similarity: per-layer expert set overlap
  - Affinity correlation: Pearson R between training and inference co-activation
  - Hit rate prediction: estimated OCS hit rate from affinity overlap

All metrics are computed per-layer and aggregated across layers.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _ensure_numpy(X):
    if hasattr(X, "numpy"):
        return X.numpy()
    return np.asarray(X)


def expert_distribution(
    expert_ids_list: List[List[int]],
    num_experts: int,
) -> np.ndarray:
    """Compute expert selection frequency distribution from routing data.

    Args:
        expert_ids_list: list of [expert_ids_per_token] lists (multi-token).
        num_experts: total number of experts.

    Returns:
        numpy array [num_experts] of selection probabilities summing to 1.
    """
    counts = np.zeros(num_experts, dtype=np.float64)
    for ids in expert_ids_list:
        for e in ids:
            if 0 <= e < num_experts:
                counts[e] += 1.0
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two discrete distributions.

    JS(p,q) = 0.5 * KL(p||m) + 0.5 * KL(q||m) where m = (p+q)/2.
    Bounded in [0, 1] when using base-2 log. 0 = identical, 1 = maximally different.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        a = np.maximum(a, 1e-12)
        b = np.maximum(b, 1e-12)
        return np.sum(a * np.log2(a / b))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def topk_overlap(
    train_ids: np.ndarray,
    infer_ids: np.ndarray,
    k: int,
) -> float:
    """Fraction of tokens where top-K expert assignments match.

    Args:
        train_ids: [T, K] expert IDs from training.
        infer_ids: [T, K] expert IDs from inference.
        k: number of top experts to compare.

    Returns:
        Fraction in [0, 1] where the set of top-K experts is identical.
    """
    if train_ids.shape != infer_ids.shape:
        min_t = min(train_ids.shape[0], infer_ids.shape[0])
        train_ids = train_ids[:min_t]
        infer_ids = infer_ids[:min_t]

    if min_t == 0:
        return 0.0

    k_actual = min(k, train_ids.shape[1])
    matches = 0
    for i in range(min_t):
        train_set = set(train_ids[i, :k_actual].tolist())
        infer_set = set(infer_ids[i, :k_actual].tolist())
        if train_set == infer_set or len(train_set & infer_set) >= k_actual:
            matches += 1
    return matches / min_t


def jaccard_similarity(
    train_ids: np.ndarray,
    infer_ids: np.ndarray,
) -> float:
    """Jaccard similarity of expert sets used in training vs inference.

    Args:
        train_ids: [T, K] expert IDs from training.
        infer_ids: [T, K] expert IDs from inference.

    Returns:
        Jaccard index in [0, 1]: size of intersection / size of union.
    """
    train_set = set(train_ids.flatten().tolist())
    infer_set = set(infer_ids.flatten().tolist())
    union = train_set | infer_set
    if not union:
        return 0.0
    return len(train_set & infer_set) / len(union)


def affinity_correlation(
    train_affinity: np.ndarray,
    infer_affinity: np.ndarray,
) -> float:
    """Pearson correlation between training and inference co-activation matrices.

    Args:
        train_affinity: [E, E] training co-activation matrix.
        infer_affinity: [E, E] inference co-activation matrix.

    Returns:
        Pearson R in [-1, 1]. Values near 1 indicate strong linear correlation.
    """
    t = train_affinity.flatten()
    i = infer_affinity.flatten()
    if t.std() < 1e-12 or i.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(t, i)[0, 1])


def estimated_hit_rate(
    train_affinity: np.ndarray,
    max_circuits: int,
    num_ranks: int,
    experts_per_rank: int,
) -> float:
    """Estimate OCS circuit hit rate from training affinity.

    Computes the fraction of high-affinity rank pairs that would be covered
    by the top max_circuits circuits based on training co-activation patterns.

    This is a theoretical upper bound on the preset OCS hit rate — actual
    inference hit rate may be lower if affinity patterns shift.

    Args:
        train_affinity: [num_experts, num_experts] co-activation matrix.
        max_circuits: maximum circuits in OCS pool.
        num_ranks: number of GPU ranks.
        experts_per_rank: experts per rank.

    Returns:
        Estimated hit rate in [0, 1].
    """
    num_experts = train_affinity.shape[0]
    expert_to_rank = {e: e // experts_per_rank for e in range(num_experts)}

    rank_pair_scores: Dict[tuple, float] = {}
    for ea in range(num_experts):
        ra = expert_to_rank[ea]
        for eb in range(num_experts):
            rb = expert_to_rank[eb]
            if ra == rb:
                continue
            key = (ra, rb)
            rank_pair_scores[key] = max(
                rank_pair_scores.get(key, 0.0), train_affinity[ea, eb],
            )

    sorted_pairs = sorted(
        rank_pair_scores.items(), key=lambda x: x[1], reverse=True,
    )
    top_pairs = set(key for key, _ in sorted_pairs[:max_circuits])

    total_score = sum(s for _, s in sorted_pairs)
    if total_score == 0:
        return 0.0
    covered_score = sum(s for k, s in sorted_pairs if k in top_pairs)
    return covered_score / total_score


def layer_consistency_report(
    train_layers: List[List[List[int]]],
    infer_layers: List[List[List[int]]],
    num_experts: int,
    top_k: int,
) -> Dict:
    """Produce a per-layer consistency report between training and inference.

    Args:
        train_layers: layer → token_list → [expert_ids]. Shape [L][T][K].
        infer_layers: same shape as train_layers, for inference.
        num_experts: total experts.
        top_k: number of top experts to compare.

    Returns dict with per-layer metrics and global aggregates.
    """
    L = min(len(train_layers), len(infer_layers))
    per_layer = []

    for layer_idx in range(L):
        train_dist = expert_distribution(train_layers[layer_idx], num_experts)
        infer_dist = expert_distribution(infer_layers[layer_idx], num_experts)

        js = js_divergence(train_dist, infer_dist)
        jac = jaccard_similarity(
            np.array(train_layers[layer_idx]),
            np.array(infer_layers[layer_idx]),
        )

        per_layer.append({
            "layer": layer_idx,
            "js_divergence": round(js, 6),
            "jaccard_similarity": round(jac, 6),
        })

    avg_js = np.mean([p["js_divergence"] for p in per_layer])
    avg_jac = np.mean([p["jaccard_similarity"] for p in per_layer])

    return {
        "num_layers": L,
        "num_experts": num_experts,
        "top_k": top_k,
        "per_layer": per_layer,
        "global": {
            "mean_js_divergence": round(avg_js, 6),
            "mean_jaccard_similarity": round(avg_jac, 6),
        },
    }
