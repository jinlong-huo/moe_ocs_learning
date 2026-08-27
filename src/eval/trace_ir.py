"""
trace_ir.py — the canonical intermediate representation for routing analysis.

Design rule
───────────
Routing is captured ONCE.  Every downstream stage (affinity, placement,
topology, cost, OCS) is a pure function of this IR.  No stage may recompute or
mutate a routing decision; that is what makes the invariance claim structural
rather than asserted.

The central object is ``CellTable``: the flat relation

    (run_idx, layer, token_pos, token_id, phase, expert_0..expert_{K-1},
                                                weight_0..weight_{K-1})

one row per *routing cell*, i.e. per (sequence, token, MoE layer).

Two correctness points the previous code got wrong
──────────────────────────────────────────────────
1. **Expert ids are PER-LAYER namespaces.**  Expert 5 of layer 3 and expert 5
   of layer 30 are different weight matrices computed by different gates.
   Measured on a real trace, per-layer expert-load vectors are essentially
   uncorrelated across layers (Pearson r ~ 0.01).  Pooling all layers into one
   ``num_experts``-wide histogram therefore averages ~L unrelated
   distributions and drives every distributional metric toward uniform — which
   is why two semantically distant prompts previously reported a
   Jensen-Shannon divergence of 0.005.  ``CellTable`` keeps ``layer`` as a
   first-class key and every statistic has an explicit ``per_layer`` form.

2. **Placement may be per-layer.**  Nothing in expert parallelism forces every
   layer to share one expert->rank map.  ``GLOBAL`` and ``PER_LAYER`` placement
   scopes are both first-class here, and the per-layer scope turns out to be
   where essentially all of the exploitable structure lives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RunInfo:
    """Labels for one captured sequence (one row of the design matrix)."""

    run_idx: int
    uid: str
    category: str
    group: str
    role: str
    variant: int
    model_id: str
    prompt_len: int
    generated_len: int
    total_tokens: int
    trace_path: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class CellTable:
    """Column-oriented routing cells over a whole workload.

    Arrays are parallel, length ``n_cells``:
      run     int32   index into ``runs``
      layer   int32   absolute decoder-layer index of the MoE block
      pos     int32   absolute token position within its sequence
      tok     int32   token id (-1 if unresolved)
      phase   uint8   0 = prefill, 1 = decode
      experts int32   [n_cells, K] selected expert ids, sorted by descending
                      gate weight (experts[:,0] is the argmax)
      weights float32 [n_cells, K] top-k softmax mass
    """

    run: np.ndarray
    layer: np.ndarray
    pos: np.ndarray
    tok: np.ndarray
    phase: np.ndarray
    experts: np.ndarray
    weights: np.ndarray
    runs: list[RunInfo]
    num_experts: int
    top_k: int
    model_id: str
    layers: np.ndarray = field(default=None)          # sorted unique layer ids

    def __post_init__(self):
        if self.layers is None:
            self.layers = np.unique(self.layer)

    # ── shape ────────────────────────────────────────────────────────
    @property
    def n_cells(self) -> int:
        return int(self.run.shape[0])

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    @property
    def n_layers(self) -> int:
        return int(self.layers.shape[0])

    def __repr__(self) -> str:
        return (f"CellTable(model={self.model_id} E={self.num_experts} "
                f"K={self.top_k} runs={self.n_runs} layers={self.n_layers} "
                f"cells={self.n_cells})")

    # ── selection (all return views/copies, never mutate) ─────────────
    def select(self, mask: np.ndarray) -> "CellTable":
        return CellTable(
            run=self.run[mask], layer=self.layer[mask], pos=self.pos[mask],
            tok=self.tok[mask], phase=self.phase[mask],
            experts=self.experts[mask], weights=self.weights[mask],
            runs=self.runs, num_experts=self.num_experts, top_k=self.top_k,
            model_id=self.model_id, layers=self.layers,
        )

    def by_runs(self, uids: list[str] | set[str]) -> "CellTable":
        uids = set(uids)
        keep = np.array([r.run_idx for r in self.runs if r.uid in uids],
                        dtype=np.int64)
        return self.select(np.isin(self.run, keep))

    def by_role(self, role: str) -> "CellTable":
        keep = np.array([r.run_idx for r in self.runs if r.role == role],
                        dtype=np.int64)
        return self.select(np.isin(self.run, keep))

    def by_category(self, cat: str) -> "CellTable":
        keep = np.array([r.run_idx for r in self.runs if r.category == cat],
                        dtype=np.int64)
        return self.select(np.isin(self.run, keep))

    def by_layer(self, layer: int) -> "CellTable":
        return self.select(self.layer == layer)

    def decode_only(self) -> "CellTable":
        return self.select(self.phase == 1)

    # ── run label helpers ────────────────────────────────────────────
    def run_info(self) -> dict[int, RunInfo]:
        return {r.run_idx: r for r in self.runs}

    def categories(self) -> list[str]:
        seen = []
        for r in self.runs:
            if r.category not in seen:
                seen.append(r.category)
        return seen

    def uids(self) -> list[str]:
        return [r.uid for r in self.runs]

    def cell_category(self) -> np.ndarray:
        """Per-cell category index + the label list."""
        cats = self.categories()
        idx = {c: i for i, c in enumerate(cats)}
        m = np.zeros(self.n_runs and (max(r.run_idx for r in self.runs) + 1) or 1,
                     dtype=np.int32)
        for r in self.runs:
            m[r.run_idx] = idx[r.category]
        return m[self.run], cats

    # ── core statistics (ALWAYS per-layer aware) ─────────────────────
    def expert_load(self, layer: int | None = None) -> np.ndarray:
        """Selection counts per expert.  ``layer=None`` pools (use only when
        the placement being evaluated is itself layer-shared)."""
        t = self if layer is None else self.by_layer(layer)
        return np.bincount(t.experts.ravel(), minlength=self.num_experts).astype(np.float64)

    def per_layer_load(self) -> np.ndarray:
        """[n_layers, num_experts] selection counts."""
        out = np.zeros((self.n_layers, self.num_experts), dtype=np.float64)
        for i, l in enumerate(self.layers):
            sub = self.experts[self.layer == l]
            if sub.size:
                out[i] = np.bincount(sub.ravel(), minlength=self.num_experts)
        return out

    def cross_layer_load_correlation(self) -> dict:
        """Pearson r between per-layer load vectors.

        This is the diagnostic that invalidates layer-pooled expert statistics:
        if r ~ 0 the layers carry independent expert-id namespaces.
        """
        L = self.per_layer_load()
        Ln = L / np.maximum(L.sum(1, keepdims=True), 1e-12)
        rs = []
        for i in range(Ln.shape[0]):
            for j in range(i + 1, Ln.shape[0]):
                a, b = Ln[i], Ln[j]
                if a.std() > 1e-12 and b.std() > 1e-12:
                    rs.append(float(np.corrcoef(a, b)[0, 1]))
        rs = np.asarray(rs) if rs else np.zeros(1)
        return {"n_pairs": int(rs.size), "mean_r": float(rs.mean()),
                "p05": float(np.percentile(rs, 5)),
                "p95": float(np.percentile(rs, 95)),
                "abs_mean_r": float(np.abs(rs).mean())}

    def load_balance(self, layer: int | None = None) -> dict:
        """Skew of the expert-load distribution.

        ``max_over_uniform`` is the quantity that actually bounds all-to-all
        completion time: the collective finishes when the busiest destination
        finishes, so a 15x-overloaded expert is a 15x-overloaded rank if it is
        placed alone.
        """
        load = self.expert_load(layer)
        tot = load.sum()
        if tot <= 0:
            return {"max_over_uniform": 0.0, "gini": 0.0, "cv": 0.0,
                    "top_eighth_share": 0.0, "unused_fraction": 1.0}
        p = load / tot
        u = 1.0 / self.num_experts
        srt = np.sort(p)
        n = p.size
        gini = float((2 * np.arange(1, n + 1) - n - 1).dot(srt) / (n * srt.sum() + 1e-18))
        k8 = max(1, n // 8)
        return {
            "max_over_uniform": float(p.max() / u),
            "gini": gini,
            "cv": float(p.std() / (p.mean() + 1e-18)),
            "top_eighth_share": float(np.sort(p)[::-1][:k8].sum()),
            "unused_fraction": float((load == 0).mean()),
        }

    # ── signatures for similarity work ───────────────────────────────
    def layer_signature(self, normalize: bool = True) -> np.ndarray:
        """[n_layers * num_experts] concatenated per-layer load vector.

        This is the correct "routing fingerprint" of a workload: it keeps each
        layer's expert namespace separate, unlike a pooled histogram.
        """
        L = self.per_layer_load()
        if normalize:
            L = L / np.maximum(L.sum(1, keepdims=True), 1e-12)
        return L.ravel()

    def run_signatures(self, normalize: bool = True
                       ) -> tuple[np.ndarray, list[RunInfo]]:
        """[n_runs, n_layers*num_experts] one signature per sequence."""
        infos, rows = [], []
        for r in self.runs:
            sub = self.select(self.run == r.run_idx)
            if sub.n_cells == 0:
                continue
            infos.append(r)
            rows.append(sub.layer_signature(normalize))
        return (np.stack(rows) if rows else np.zeros((0, 1))), infos


# ═══════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════

def load_workload(manifest_path: str | Path,
                  roles: list[str] | None = None,
                  max_runs: int | None = None,
                  decode_only: bool = False) -> CellTable:
    """Build a ``CellTable`` from a ``capture_workload.py`` manifest."""
    mp = Path(manifest_path)
    root = mp.parent
    man = json.load(open(mp))

    recs = [r for r in man["records"] if roles is None or r["role"] in roles]
    if max_runs:
        recs = recs[:max_runs]

    runs: list[RunInfo] = []
    run_a, lay_a, pos_a, tok_a, ph_a, exp_a, w_a = [], [], [], [], [], [], []
    E = K = 0
    model_id = man.get("model", "")

    for i, rec in enumerate(recs):
        raw = json.load(open(root / rec["trace"]))
        meta = raw["meta"]
        E = E or int(meta["num_experts"])
        K = K or int(meta["top_k"])
        model_id = meta.get("model_id", model_id)
        runs.append(RunInfo(
            run_idx=i, uid=rec["uid"], category=rec["category"],
            group=rec["group"], role=rec["role"], variant=rec.get("variant", 0),
            model_id=meta.get("model_id", ""), prompt_len=meta["prompt_len"],
            generated_len=meta["generated_len"], total_tokens=meta["total_tokens"],
            trace_path=rec["trace"], meta=rec.get("meta", {}) or {},
        ))
        for route in raw["routes"]:
            is_dec = 1 if route["phase"] == "decode" else 0
            if decode_only and not is_dec:
                continue
            for lid, lr in route["layers"].items():
                ex = lr["experts"]
                if len(ex) != K:
                    continue
                run_a.append(i); lay_a.append(int(lid))
                pos_a.append(route["token_pos"]); tok_a.append(route["token_id"])
                ph_a.append(is_dec); exp_a.append(ex); w_a.append(lr["weights"])

    if not run_a:
        raise ValueError(f"no routing cells loaded from {mp}")

    return CellTable(
        run=np.asarray(run_a, dtype=np.int32),
        layer=np.asarray(lay_a, dtype=np.int32),
        pos=np.asarray(pos_a, dtype=np.int32),
        tok=np.asarray(tok_a, dtype=np.int32),
        phase=np.asarray(ph_a, dtype=np.uint8),
        experts=np.asarray(exp_a, dtype=np.int32),
        weights=np.asarray(w_a, dtype=np.float32),
        runs=runs, num_experts=E, top_k=K, model_id=model_id,
    )


def load_single_trace(trace_path: str | Path, uid: str = "single",
                      category: str = "single") -> CellTable:
    """Wrap one standalone ``RoutingTrace`` JSON as a 1-run ``CellTable``."""
    raw = json.load(open(trace_path))
    meta = raw["meta"]
    rows = []
    for route in raw["routes"]:
        is_dec = 1 if route["phase"] == "decode" else 0
        for lid, lr in route["layers"].items():
            if len(lr["experts"]) != meta["top_k"]:
                continue
            rows.append((0, int(lid), route["token_pos"], route["token_id"],
                         is_dec, lr["experts"], lr["weights"]))
    if not rows:
        raise ValueError(f"no cells in {trace_path}")
    return CellTable(
        run=np.array([r[0] for r in rows], dtype=np.int32),
        layer=np.array([r[1] for r in rows], dtype=np.int32),
        pos=np.array([r[2] for r in rows], dtype=np.int32),
        tok=np.array([r[3] for r in rows], dtype=np.int32),
        phase=np.array([r[4] for r in rows], dtype=np.uint8),
        experts=np.array([r[5] for r in rows], dtype=np.int32),
        weights=np.array([r[6] for r in rows], dtype=np.float32),
        runs=[RunInfo(0, uid, category, category, "single", 0,
                      meta.get("model_id", ""), meta["prompt_len"],
                      meta["generated_len"], meta["total_tokens"],
                      str(trace_path))],
        num_experts=meta["num_experts"], top_k=meta["top_k"],
        model_id=meta.get("model_id", ""),
    )


# ═══════════════════════════════════════════════════════════════════════
# Cell-level identity — the primitive behind every invariance check
# ═══════════════════════════════════════════════════════════════════════

def cell_key_map(t: CellTable) -> dict[tuple[int, int, int], tuple[int, ...]]:
    """{(run, pos, layer): sorted expert tuple} — the logical routing map."""
    out = {}
    for i in range(t.n_cells):
        out[(int(t.run[i]), int(t.pos[i]), int(t.layer[i]))] = tuple(
            sorted(int(e) for e in t.experts[i]))
    return out


def routing_identical(a: CellTable, b: CellTable) -> dict:
    """Bit-compare the logical token->expert map of two cell tables.

    Returns the exact cell counts, never a bare boolean, so a partial
    mismatch is visible as a rate rather than collapsing to False.
    """
    ka, kb = cell_key_map(a), cell_key_map(b)
    common = set(ka) & set(kb)
    if not common:
        return {"n_common": 0, "identical": False, "match_rate": 0.0,
                "mismatched": 0, "only_a": len(ka), "only_b": len(kb)}
    mism = sum(1 for k in common if ka[k] != kb[k])
    return {
        "n_common": len(common),
        "mismatched": mism,
        "match_rate": 1.0 - mism / len(common),
        "identical": mism == 0 and len(ka) == len(kb) == len(common),
        "only_a": len(ka) - len(common),
        "only_b": len(kb) - len(common),
    }
