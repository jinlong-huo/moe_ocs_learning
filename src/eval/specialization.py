"""
specialization.py — does the workload determine the routing, and how strongly?

Three tests, in increasing strength of evidence:

1. ``expert_category_mi``  Mutual information between workload category and
   selected expert, per layer, against a permutation null.

2. ``specialization_index`` Per-expert KL divergence of P(category | expert)
   from the category prior — which experts are specialists, and by how much.

3. ``category_decoding``   Can the category of a HELD-OUT sequence be
   predicted from its routing signature alone?  Leave-one-out nearest-centroid
   against a 1/n_categories chance baseline.  This is the strongest test
   because it is out-of-sample by construction and needs no null model to
   interpret.

The permutation discipline that the previous analysis lacked
────────────────────────────────────────────────────────────
Routing cells within one sequence are massively dependent: a 130-token
sequence contributes ~3000 cells that share a prompt, a topic and a decoding
trajectory.  Permuting *cell* labels therefore produces a null with an
effective sample size ~1000x too large and will declare almost anything
significant.  Every null here permutes at the **run** level — whole sequences
keep their cells and only the category label moves.  Likewise the
leave-one-out in ``category_decoding`` holds out a whole sequence, never a
cell.
"""

from __future__ import annotations

import numpy as np

from src.eval.trace_ir import CellTable


# ═══════════════════════════════════════════════════════════════════════
# 1. Mutual information, category vs expert
# ═══════════════════════════════════════════════════════════════════════

def _joint(cats: np.ndarray, experts: np.ndarray, n_cat: int, n_exp: int
           ) -> np.ndarray:
    """[n_cat, n_exp] selection counts."""
    J = np.zeros((n_cat, n_exp), dtype=np.float64)
    k = experts.shape[1]
    np.add.at(J, (np.repeat(cats, k), experts.ravel()), 1.0)
    return J


def _mi_bits(J: np.ndarray) -> float:
    tot = J.sum()
    if tot <= 0:
        return 0.0
    P = J / tot
    pr = P.sum(1, keepdims=True)
    pc = P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = P * np.log2(P / np.maximum(pr * pc, 1e-300))
    return float(np.nansum(np.where(P > 0, term, 0.0)))


def expert_category_mi(t: CellTable, n_perm: int = 200, seed: int = 0,
                       roles: tuple[str, ...] = ("category",)) -> dict:
    """I(category ; expert) per layer, with a RUN-level permutation null.

    ``mi_bits`` is upper-bounded by log2(n_categories); ``normalized`` divides
    by that bound so layers and models are comparable.  The reported effect is
    ``excess_over_null``: MI always has positive bias at finite sample, so the
    raw value alone is not evidence.
    """
    keep = [r for r in t.runs if r.role in roles]
    if len(keep) < 4:
        return {"insufficient": True, "n_runs": len(keep)}
    sub = t.by_runs([r.uid for r in keep])
    cats = sorted({r.category for r in keep})
    cidx = {c: i for i, c in enumerate(cats)}
    run2cat = np.zeros(max(r.run_idx for r in t.runs) + 1, dtype=np.int32)
    for r in keep:
        run2cat[r.run_idx] = cidx[r.category]

    rng = np.random.default_rng(seed)
    run_ids = np.array([r.run_idx for r in keep])
    labels = np.array([cidx[r.category] for r in keep])

    per_layer, null_layer = [], []
    for l in sub.layers:
        m = sub.layer == l
        ex, rn = sub.experts[m], sub.run[m]
        obs = _mi_bits(_joint(run2cat[rn], ex, len(cats), t.num_experts))
        nulls = []
        for _ in range(n_perm):
            perm = rng.permutation(labels)
            lut = np.zeros_like(run2cat)
            lut[run_ids] = perm
            nulls.append(_mi_bits(_joint(lut[rn], ex, len(cats), t.num_experts)))
        per_layer.append(obs)
        null_layer.append(float(np.mean(nulls)))

    obs = np.asarray(per_layer)
    nul = np.asarray(null_layer)
    bound = np.log2(len(cats))
    return {
        "n_runs": len(keep), "n_categories": len(cats), "n_perm": n_perm,
        "mi_bits_mean": round(float(obs.mean()), 6),
        "mi_bits_max": round(float(obs.max()), 6),
        "mi_null_mean": round(float(nul.mean()), 6),
        "excess_over_null_bits": round(float((obs - nul).mean()), 6),
        "excess_ratio": round(float(obs.mean() / max(nul.mean(), 1e-12)), 4),
        "normalized_mi": round(float(obs.mean() / bound), 6),
        "normalized_excess": round(float((obs - nul).mean() / bound), 6),
        "per_layer_bits": [round(v, 6) for v in obs.tolist()],
        "per_layer_null": [round(v, 6) for v in nul.tolist()],
        "mi_upper_bound_bits": round(float(bound), 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. Per-expert specialisation
# ═══════════════════════════════════════════════════════════════════════

def specialization_index(t: CellTable, roles: tuple[str, ...] = ("category",),
                         min_count: int = 30) -> dict:
    """KL( P(category | expert) || P(category) ) per (layer, expert).

    A generalist expert has KL ~ 0; a specialist concentrates on a few
    categories.  ``top_expert_categories`` names the strongest specialists so
    the claim "coding prompts prefer expert X" can be checked directly rather
    than asserted from a heatmap.
    """
    keep = [r for r in t.runs if r.role in roles]
    if len(keep) < 4:
        return {"insufficient": True}
    sub = t.by_runs([r.uid for r in keep])
    cats = sorted({r.category for r in keep})
    cidx = {c: i for i, c in enumerate(cats)}
    run2cat = np.zeros(max(r.run_idx for r in t.runs) + 1, dtype=np.int32)
    for r in keep:
        run2cat[r.run_idx] = cidx[r.category]

    kls, records = [], []
    for l in sub.layers:
        m = sub.layer == l
        J = _joint(run2cat[sub.run[m]], sub.experts[m], len(cats), t.num_experts)
        prior = J.sum(1) / max(J.sum(), 1e-12)
        cnt = J.sum(0)
        for e in range(t.num_experts):
            if cnt[e] < min_count:
                continue
            p = J[:, e] / cnt[e]
            with np.errstate(divide="ignore", invalid="ignore"):
                kl = float(np.nansum(np.where(p > 0, p * np.log2(p / np.maximum(prior, 1e-300)), 0.0)))
            kls.append(kl)
            records.append((kl, int(l), e, cats[int(np.argmax(p))],
                            float(p.max()), float(prior[int(np.argmax(p))])))
    if not kls:
        return {"insufficient": True, "reason": "no expert met min_count"}
    kls = np.asarray(kls)
    records.sort(reverse=True)
    return {
        "n_scored": int(kls.size), "n_categories": len(cats),
        "kl_bits_mean": round(float(kls.mean()), 5),
        "kl_bits_p90": round(float(np.percentile(kls, 90)), 5),
        "kl_bits_max": round(float(kls.max()), 5),
        "kl_upper_bound_bits": round(float(np.log2(len(cats))), 4),
        "specialist_fraction_kl_gt_0p5": round(float((kls > 0.5).mean()), 4),
        "top_expert_categories": [
            {"layer": l, "expert": e, "kl_bits": round(k, 4),
             "top_category": c, "p_top": round(pt, 4), "prior": round(pr, 4)}
            for k, l, e, c, pt, pr in records[:15]
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. Out-of-sample category decoding — the strongest test
# ═══════════════════════════════════════════════════════════════════════

def category_decoding(t: CellTable, roles: tuple[str, ...] = ("category",),
                      seed: int = 0, n_perm: int = 500) -> dict:
    """Leave-one-sequence-out nearest-centroid decoding of the category.

    Signatures are per-layer-concatenated load vectors (expert namespaces kept
    separate).  For each held-out sequence the class centroids are rebuilt from
    the remaining sequences only, so the number is genuinely out-of-sample.

    The permutation null shuffles category labels across sequences, giving the
    accuracy distribution attainable by chance at this sample size — which for
    ~70 sequences and 12 classes is well above the naive 1/12.
    """
    keep = [r for r in t.runs if r.role in roles]
    if len(keep) < 8:
        return {"insufficient": True, "n_runs": len(keep)}
    sub = t.by_runs([r.uid for r in keep])
    S, infos = sub.run_signatures(normalize=True)
    if S.shape[0] < 8:
        return {"insufficient": True, "n_runs": int(S.shape[0])}
    S = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-18)
    y = np.array([r.category for r in infos])
    cats = sorted(set(y.tolist()))
    yi = np.array([cats.index(c) for c in y])
    n, C = S.shape[0], len(cats)

    def loo_acc(labels: np.ndarray) -> float:
        hit = 0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            cents, ids = [], []
            for c in range(C):
                sel = mask & (labels == c)
                if sel.sum() == 0:
                    continue
                v = S[sel].mean(0)
                cents.append(v / max(np.linalg.norm(v), 1e-18))
                ids.append(c)
            if not cents:
                continue
            pred = ids[int(np.argmax(np.stack(cents) @ S[i]))]
            hit += int(pred == labels[i])
        return hit / n

    acc = loo_acc(yi)
    rng = np.random.default_rng(seed)
    nulls = np.array([loo_acc(rng.permutation(yi)) for _ in range(n_perm)])
    p = float(((nulls >= acc).sum() + 1) / (n_perm + 1))

    # per-layer decoding: which depths carry the workload signal?
    per_layer = {}
    L = sub.n_layers
    E = sub.num_experts
    for i, l in enumerate(sub.layers):
        Sl = S[:, i * E:(i + 1) * E]
        Sl = Sl / np.maximum(np.linalg.norm(Sl, axis=1, keepdims=True), 1e-18)
        hit = 0
        for j in range(n):
            mask = np.ones(n, dtype=bool); mask[j] = False
            cents, ids = [], []
            for c in range(C):
                sel = mask & (yi == c)
                if sel.sum() == 0:
                    continue
                v = Sl[sel].mean(0)
                cents.append(v / max(np.linalg.norm(v), 1e-18)); ids.append(c)
            hit += int(ids[int(np.argmax(np.stack(cents) @ Sl[j]))] == yi[j])
        per_layer[int(l)] = round(hit / n, 4)

    return {
        "n_runs": n, "n_categories": C, "chance_naive": round(1.0 / C, 4),
        "accuracy": round(acc, 4),
        "perm_null_mean": round(float(nulls.mean()), 4),
        "perm_null_p95": round(float(np.percentile(nulls, 95)), 4),
        "perm_p": round(p, 5), "n_perm": n_perm,
        "significant": bool(p < 0.05),
        "per_layer_accuracy": per_layer,
        "best_layer": max(per_layer, key=per_layer.get) if per_layer else None,
    }


# ═══════════════════════════════════════════════════════════════════════
# 4. Semantics vs surface form — the confound the 2-prompt design could not see
# ═══════════════════════════════════════════════════════════════════════

def semantics_vs_lexis(t: CellTable) -> dict:
    """Compare routing similarity across four controlled contrasts.

    ``repeat``            identical prompt, re-run -> measurement noise floor.
    ``paraphrase``        same meaning, minimal shared words.
    ``lexical_control``   same template and shared words, different domain.
    ``between_category``  different meaning and different words.

    Reading the result:
      paraphrase >> lexical_control      routing tracks MEANING
      lexical_control >> paraphrase      routing tracks SURFACE FORM
      both ~ between_category            routing carries no workload signal
      any of them ~ repeat               that contrast is at the noise floor
                                         and cannot support a claim
    """
    from src.eval.affinity_graph import signature_similarity_matrix

    M, infos = signature_similarity_matrix(t)
    if M.shape[0] < 4:
        return {"insufficient": True}
    n = M.shape[0]
    role = np.array([i.role for i in infos])
    grp = np.array([i.group for i in infos])
    cat = np.array([i.category for i in infos])
    iu = np.triu_indices(n, 1)
    a, b = iu
    sim = M[iu]

    def stat(mask):
        v = sim[mask]
        if v.size == 0:
            return None
        return {"n_pairs": int(v.size), "mean": round(float(v.mean()), 6),
                "sd": round(float(v.std()), 6),
                "ci95": [round(float(v.mean() - 1.96 * v.std() / np.sqrt(v.size)), 6),
                         round(float(v.mean() + 1.96 * v.std() / np.sqrt(v.size)), 6)]}

    same_grp = grp[a] == grp[b]
    out = {
        "noise_floor_repeat": stat((role[a] == "repeat") & (role[b] == "repeat")),
        "within_paraphrase_set": stat((role[a] == "paraphrase") &
                                      (role[b] == "paraphrase") & same_grp),
        "within_lexical_control": stat((role[a] == "lexical_control") &
                                       (role[b] == "lexical_control") & same_grp),
        "within_length_ladder": stat((role[a] == "length_ladder") &
                                     (role[b] == "length_ladder") & same_grp),
        "within_category": stat((role[a] == "category") & (role[b] == "category") &
                                (cat[a] == cat[b])),
        "between_category": stat((role[a] == "category") & (role[b] == "category") &
                                 (cat[a] != cat[b])),
    }

    # normalise every contrast onto the noise floor so the numbers are readable
    nf = out["noise_floor_repeat"]
    bc = out["between_category"]
    if nf and bc and nf["mean"] > bc["mean"]:
        span = nf["mean"] - bc["mean"]
        for k, v in out.items():
            if v is None or k in ("noise_floor_repeat", "between_category"):
                continue
            v["fraction_of_noise_floor_span"] = round(
                (v["mean"] - bc["mean"]) / span, 4)

    verdict = None
    p, l = out["within_paraphrase_set"], out["within_lexical_control"]
    if p and l:
        if p["mean"] > l["mean"] + 1.96 * (p["sd"] / max(np.sqrt(p["n_pairs"]), 1)):
            verdict = "semantic_dominant"
        elif l["mean"] > p["mean"] + 1.96 * (l["sd"] / max(np.sqrt(l["n_pairs"]), 1)):
            verdict = "lexical_dominant"
        else:
            verdict = "indistinguishable"
    out["verdict"] = verdict
    return out
