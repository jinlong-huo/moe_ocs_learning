"""
placement_opt.py — expert placement generators and their optimisation targets.

Placement families
──────────────────
``linear``            expert e -> rank e // (E/W).  The default in every MoE
                      framework; included as the *deployed* baseline.
``random``            seeded uniform permutation.  The correct null for
                      "does the placement matter at all?" — linear is not a
                      null, it is a specific (and as it turns out, essentially
                      random-equivalent) choice.
``load_balanced``     LPT bin-packing on measured expert load.  Targets
                      ingress imbalance, NOT affinity.  Included because
                      measured load skew is large (top 1/8 of experts can carry
                      ~2/3 of tokens) and any affinity claim must be shown to
                      beat this much simpler intervention.
``affinity_global``   greedy co-activation clustering with one map for all
                      layers.
``affinity_layer``    the same, but an independent map per MoE layer.
``fanout_layer``      direct local search on the true objective (mean distinct
                      destination ranks per cell), per layer.  This is the
                      honest optimiser: affinity clustering is only a pairwise
                      *relaxation* of it, so reporting affinity results without
                      the direct optimum leaves the headroom unknown.
``hierarchical``      two-level: first partition experts into node-sized groups
                      to minimise inter-node fan-out, then place groups.
                      Matches the real cost structure (intra-node is ~9x the
                      bandwidth of inter-node) rather than treating all ranks
                      as equidistant.
``adversarial``       maximise fan-out.  Gives the upper end of the achievable
                      range so a reported gain can be read as a fraction of
                      what is actually available.

Every generator takes ONLY a fit slice of the workload.  Evaluation happens on
a disjoint slice in ``verify_live_invariance.py``; nothing here ever sees the
evaluation data.
"""

from __future__ import annotations

import numpy as np

from src.eval.affinity_graph import affinity_matrix, layer_affinities, pooled_affinity
from src.eval.cost_model import Placement
from src.eval.trace_ir import CellTable


# ═══════════════════════════════════════════════════════════════════════
# Objective: mean distinct destination ranks per routing cell
# ═══════════════════════════════════════════════════════════════════════

def mean_fanout(experts: np.ndarray, expert_to_rank: np.ndarray) -> float:
    """Vectorised mean |{rank(e) : e in topk}| over cells."""
    if experts.shape[0] == 0:
        return 0.0
    r = np.sort(expert_to_rank[experts], axis=1)
    new = np.ones_like(r, dtype=bool)
    new[:, 1:] = r[:, 1:] != r[:, :-1]
    return float(new.sum(1).mean())


def group_fanout(experts: np.ndarray, expert_to_rank: np.ndarray,
                 gpus_per_node: int) -> float:
    """Fan-out measured in NODES rather than ranks (inter-node volume)."""
    if experts.shape[0] == 0:
        return 0.0
    r = np.sort(expert_to_rank[experts] // gpus_per_node, axis=1)
    new = np.ones_like(r, dtype=bool)
    new[:, 1:] = r[:, 1:] != r[:, :-1]
    return float(new.sum(1).mean())


def dedup_ingress(experts: np.ndarray, expert_to_rank: np.ndarray,
                  world_size: int) -> np.ndarray:
    """[W] messages arriving at each rank under dedup dispatch.

    This — not the expert *selection* count — is the quantity that sets the
    all-to-all completion time.  The two differ sharply: a rank that owns one
    very popular expert receives a message from nearly every token regardless
    of what else it owns, so message counts saturate while selection counts do
    not.  Balancing selection counts (classic LPT load balancing) therefore
    does NOT balance the collective, which is why ``load_balanced_layer``
    can make the bottleneck worse.
    """
    if experts.shape[0] == 0:
        return np.zeros(world_size)
    r = np.sort(expert_to_rank[experts], axis=1)
    new = np.ones_like(r, dtype=bool)
    new[:, 1:] = r[:, 1:] != r[:, :-1]
    return np.bincount(r[new], minlength=world_size).astype(np.float64)


def bottleneck_objective(experts: np.ndarray, expert_to_rank: np.ndarray,
                         world_size: int, volume_weight: float = 0.15) -> float:
    """max-ingress (the collective's critical path) plus a volume term.

    ``volume_weight`` keeps total traffic in the objective so the optimiser
    does not buy balance by inflating fan-out.  Both terms are normalised by
    the number of cells so the value is comparable across workload slices.
    """
    ing = dedup_ingress(experts, expert_to_rank, world_size)
    n = max(experts.shape[0], 1)
    return float(ing.max() / n + volume_weight * ing.sum() / (n * world_size))


class IngressOracle:
    """Fast dedup-ingress evaluator for local search.

    Each expert's set of routing cells is stored as a packed bitset, so the
    messages arriving at a rank are ``popcount(OR of its experts' bitsets)`` —
    the union is exactly the dedup semantics (a token that selects three
    experts on one rank still sends one message).  This turns an objective
    evaluation from an O(N*K) sort into a handful of word operations, which is
    what makes swap-based local search tractable at E=256 over 40 layers.
    """

    __slots__ = ("bits", "nwords", "n_cells", "E", "W", "volume_weight", "_lut")

    def __init__(self, experts: np.ndarray, num_experts: int, world_size: int,
                 volume_weight: float = 0.15):
        n, k = experts.shape
        self.n_cells, self.E, self.W = n, num_experts, world_size
        self.volume_weight = volume_weight
        self.nwords = (n + 63) // 64
        bits = np.zeros((num_experts, self.nwords), dtype=np.uint64)
        rows = np.repeat(np.arange(n, dtype=np.int64), k)
        cols = experts.ravel().astype(np.int64)
        word = rows >> 6
        shift = (rows & 63).astype(np.uint64)
        one = np.uint64(1)
        # np.bitwise_or.at is not available; group by expert and OR in place
        order = np.argsort(cols, kind="stable")
        cs, ws, sh = cols[order], word[order], shift[order]
        bnd = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
        for i, st in enumerate(bnd):
            en = bnd[i + 1] if i + 1 < len(bnd) else len(cs)
            e = int(cs[st])
            np.bitwise_or.at(bits[e], ws[st:en], one << sh[st:en])
        self.bits = bits
        self._lut = np.unpackbits(
            np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int64)

    def _popcount(self, words: np.ndarray) -> np.ndarray:
        """Popcount along the last axis of a uint64 array."""
        return self._lut[words.view(np.uint8).reshape(*words.shape[:-1], -1)].sum(-1)

    def ingress(self, p: np.ndarray) -> np.ndarray:
        """[W] arriving message count per rank.

        Assumes a *balanced* placement (exactly E/W experts per rank), which
        every generator and every swap in this module preserves.  That lets the
        whole reduction be one vectorised ``bitwise_or`` over a
        [W, E/W, nwords] view instead of a Python loop over ranks.
        """
        order = np.argsort(p, kind="stable")
        epr = self.E // self.W
        if order.size != self.E or self.E % self.W:
            return self._ingress_slow(p)
        blocks = self.bits[order].reshape(self.W, epr, self.nwords)
        acc = np.bitwise_or.reduce(blocks, axis=1)
        counts = self._popcount(acc)
        out = np.zeros(self.W, dtype=np.int64)
        out[p[order[::epr]]] = counts
        return out

    def _ingress_slow(self, p: np.ndarray) -> np.ndarray:
        out = np.zeros(self.W, dtype=np.int64)
        for r in range(self.W):
            m = np.flatnonzero(p == r)
            if m.size:
                acc = np.bitwise_or.reduce(self.bits[m], axis=0)
                out[r] = int(self._popcount(acc))
        return out

    def objective(self, p: np.ndarray) -> float:
        """max-ingress (critical path) + a volume term.

        The volume term keeps total traffic in the objective so the optimiser
        cannot buy balance by inflating fan-out.
        """
        ing = self.ingress(p)
        n = max(self.n_cells, 1)
        return float(ing.max() / n
                     + self.volume_weight * ing.sum() / (n * self.W))

    def fanout(self, p: np.ndarray) -> float:
        """Mean distinct destination ranks per cell.

        Identity worth noting: the sum of dedup ingress over ranks IS the total
        message count, so mean fan-out comes free from the same reduction.
        """
        return float(self.ingress(p).sum() / max(self.n_cells, 1))


# ═══════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════

def _linear_map(E: int, W: int) -> np.ndarray:
    epr = E // W
    return (np.arange(E) // epr).astype(np.int32)


def _random_map(E: int, W: int, rng: np.random.Generator) -> np.ndarray:
    epr = E // W
    perm = rng.permutation(E)
    out = np.empty(E, dtype=np.int32)
    out[perm] = np.arange(E) // epr
    return out


def _lpt_map(load: np.ndarray, W: int, capacity: int | None = None) -> np.ndarray:
    """Longest-processing-time first: equalise total load per rank."""
    E = load.shape[0]
    cap = capacity or (E // W)
    out = np.empty(E, dtype=np.int32)
    tot = np.zeros(W)
    cnt = np.zeros(W, dtype=np.int64)
    for e in np.argsort(-load):
        avail = np.where(cnt < cap)[0]
        r = int(avail[np.argmin(tot[avail])])
        out[e] = r
        tot[r] += load[e]
        cnt[r] += 1
    return out


def _greedy_cluster(A: np.ndarray, W: int) -> np.ndarray:
    """Seed-and-fill greedy affinity clustering (the previous algorithm,
    vectorised and with balanced capacity enforced)."""
    E = A.shape[0]
    cap = E // W
    out = np.full(E, -1, dtype=np.int32)
    remaining = np.ones(E, dtype=bool)
    for r in range(W):
        if not remaining.any():
            break
        rem = np.flatnonzero(remaining)
        sub = A[np.ix_(rem, rem)]
        seed = rem[int(np.argmax(sub.sum(1)))]
        grp = [seed]
        remaining[seed] = False
        while len(grp) < cap and remaining.any():
            rem = np.flatnonzero(remaining)
            sc = A[np.ix_(rem, grp)].sum(1) + A[np.ix_(grp, rem)].sum(0)
            nxt = rem[int(np.argmax(sc))]
            grp.append(nxt)
            remaining[nxt] = False
        out[np.asarray(grp)] = r
    leftovers = np.flatnonzero(out < 0)
    for i, e in enumerate(leftovers):
        out[e] = i % W
    return out


def _local_search(experts: np.ndarray, init: np.ndarray, W: int,
                  rng: np.random.Generator, objective, n_sweeps: int = 6,
                  n_cand: int = 24) -> np.ndarray:
    """Swap-based local search on an arbitrary placement objective.

    Only *swaps* are proposed, so the balanced experts-per-rank invariant is
    preserved by construction.  Objective evaluations dominate the cost, so
    candidates are sampled rather than exhaustive: O(E * n_cand) per sweep.
    """
    p = init.copy()
    best = objective(p)
    E = p.shape[0]
    for _ in range(n_sweeps):
        improved = False
        for a in rng.permutation(E):
            cand = rng.choice(E, size=min(E, n_cand), replace=False)
            for b in cand:
                if p[a] == p[b]:
                    continue
                p[a], p[b] = p[b], p[a]
                v = objective(p)
                if v < best - 1e-12:
                    best, improved = v, True
                else:
                    p[a], p[b] = p[b], p[a]
        if not improved:
            break
    return p


def _fanout_local_search(experts: np.ndarray, init: np.ndarray, W: int,
                         rng: np.random.Generator, n_sweeps: int = 6,
                         group_size: int = 1) -> np.ndarray:
    obj = ((lambda m: mean_fanout(experts, m)) if group_size == 1
           else (lambda m: group_fanout(experts, m, group_size)))
    return _local_search(experts, init, W, rng, obj, n_sweeps)


def _capped_cluster(A: np.ndarray, experts: np.ndarray, W: int,
                    ingress_cap_ratio: float = 1.25) -> np.ndarray:
    """Affinity clustering with a hard cap on per-rank dedup ingress.

    Pure affinity clustering co-locates the *popular* experts (they co-occur
    with everything), which is exactly what destroys ingress balance.  The cap
    forbids a rank from exceeding ``ingress_cap_ratio`` times the ideal share,
    so the optimiser keeps the affinity gain only where it is affordable.
    """
    E = A.shape[0]
    cap_slots = E // W
    out = np.full(E, -1, dtype=np.int32)
    remaining = np.ones(E, dtype=bool)
    # per-expert token reach: how many cells would a rank owning e hear from
    reach = np.bincount(experts.ravel(), minlength=E).astype(np.float64)
    n_cells = max(experts.shape[0], 1)
    ideal = reach.sum() / W
    cap = ingress_cap_ratio * ideal

    for r in range(W):
        if not remaining.any():
            break
        rem = np.flatnonzero(remaining)
        sub = A[np.ix_(rem, rem)]
        seed = rem[int(np.argmax(sub.sum(1)))]
        grp = [seed]
        remaining[seed] = False
        acc = reach[seed]
        while len(grp) < cap_slots and remaining.any():
            rem = np.flatnonzero(remaining)
            sc = A[np.ix_(rem, grp)].sum(1) + A[np.ix_(grp, rem)].sum(0)
            # forbid candidates that would blow the ingress cap, unless we must
            ok = (acc + reach[rem]) <= cap
            slots_left = cap_slots - len(grp)
            if ok.any() and remaining.sum() > slots_left:
                sc = np.where(ok, sc, -np.inf)
            nxt = rem[int(np.argmax(sc))]
            grp.append(nxt)
            remaining[nxt] = False
            acc += reach[nxt]
        out[np.asarray(grp)] = r
    for i, e in enumerate(np.flatnonzero(out < 0)):
        out[e] = i % W
    return out


# ═══════════════════════════════════════════════════════════════════════
# Generators
# ═══════════════════════════════════════════════════════════════════════

def make_placement(kind: str, fit: CellTable, world_size: int, *,
                   seed: int = 0, gpus_per_node: int = 8,
                   affinity_kind: str = "cooccurrence",
                   n_sweeps: int = 4) -> Placement:
    """Build one placement from a FIT slice of the workload."""
    E, W = fit.num_experts, world_size
    if E % W:
        raise ValueError(f"E={E} not divisible by W={W}")
    rng = np.random.default_rng(seed)

    if kind == "linear":
        return _check(Placement(_linear_map(E, W), W, "global", name="linear"), W)

    if kind == "random":
        return _check(Placement(_random_map(E, W, rng), W, "global", name=f"random.s{seed}"), W)

    if kind == "load_balanced":
        return _check(Placement(_lpt_map(fit.expert_load(), W), W, "global",
                         name="load_balanced"), W)

    if kind == "load_balanced_layer":
        L = fit.per_layer_load()
        m = np.stack([_lpt_map(L[i], W) for i in range(fit.n_layers)])
        return _check(Placement(m, W, "per_layer", layers=fit.layers,
                         name="load_balanced_layer"), W)

    if kind == "affinity_global":
        A = pooled_affinity(fit, affinity_kind)
        return _check(Placement(_greedy_cluster(A, W), W, "global",
                         name=f"affinity_global.{affinity_kind}"), W)

    if kind == "affinity_layer":
        As = layer_affinities(fit, affinity_kind)
        m = np.stack([_greedy_cluster(As[int(l)], W) for l in fit.layers])
        return _check(Placement(m, W, "per_layer", layers=fit.layers,
                         name=f"affinity_layer.{affinity_kind}"), W)

    if kind == "fanout_layer":
        As = layer_affinities(fit, "cooccurrence")
        rows = []
        for l in fit.layers:
            ex = fit.experts[fit.layer == l]
            orc = IngressOracle(ex, E, W)
            init = _greedy_cluster(As[int(l)], W)
            rows.append(_local_search(ex, init, W, rng, orc.fanout,
                                      n_sweeps, n_cand=12))
        return _check(Placement(np.stack(rows), W, "per_layer", layers=fit.layers,
                         name="fanout_layer"), W)

    if kind == "balanced_affinity_layer":
        # affinity clustering constrained by a per-rank dedup-ingress cap
        As = layer_affinities(fit, "cooccurrence")
        rows = []
        for l in fit.layers:
            ex = fit.experts[fit.layer == l]
            rows.append(_capped_cluster(As[int(l)], ex, W))
        return _check(Placement(np.stack(rows), W, "per_layer", layers=fit.layers,
                         name="balanced_affinity_layer"), W)

    if kind in ("bottleneck_layer", "affinity_coordinated_layer"):
        # Coordinated per-layer optimisation.
        #
        # The subtlety that makes independent per-layer optimisation wrong: the
        # collective's critical path is  max_s  SUM_l ingress_l(s)  — a rank's
        # ingress accumulated over every MoE layer — but optimising each layer
        # in isolation minimises  SUM_l  max_s ingress_l(s).  Those differ, and
        # independent optimisers happily pile their respective hot spots onto
        # the same rank, which is exactly how per-layer affinity clustering
        # ended up ~2x WORSE than random despite cutting volume by a third.
        #
        # So layers are optimised in sequence against a running accumulator:
        # each layer sees the load its predecessors already committed.
        As = layer_affinities(fit, "cooccurrence")
        affinity_init = (kind == "affinity_coordinated_layer")
        acc = np.zeros(W, dtype=np.float64)
        rows = []
        for l in fit.layers:
            ex = fit.experts[fit.layer == l]
            orc = IngressOracle(ex, E, W)
            n = max(ex.shape[0], 1)
            init = (_greedy_cluster(As[int(l)], W) if affinity_init
                    else _capped_cluster(As[int(l)], ex, W))

            def obj(p, _o=orc, _acc=acc, _n=n):
                ing = _o.ingress(p)
                tot = _acc + ing
                # primary: the accumulated critical path.
                # secondary: this layer's own volume (fan-out), so balance is
                # not bought by fanning every token out to every rank.
                return float(tot.max() / _n + 0.15 * ing.sum() / (_n * W))

            p_l = _local_search(ex, init, W, rng, obj, n_sweeps, n_cand=12)
            acc += orc.ingress(p_l)
            rows.append(p_l)
        return _check(Placement(np.stack(rows), W, "per_layer", layers=fit.layers,
                         name=kind), W)

    if kind == "hierarchical_layer":
        # Level 1: minimise NODE fan-out (inter-node bytes dominate the clock).
        # Level 2: inside a node, spread load so no single GPU becomes the
        # ingress bottleneck.
        n_nodes = max(1, int(np.ceil(W / gpus_per_node)))
        As = layer_affinities(fit, "cooccurrence")
        L = fit.per_layer_load()
        rows = []
        for i, l in enumerate(fit.layers):
            ex = fit.experts[fit.layer == l]
            if n_nodes <= 1:
                rows.append(_lpt_map(L[i], W))
                continue
            node_map = _greedy_cluster(As[int(l)], n_nodes)
            orc_nd = IngressOracle(ex, E, n_nodes)
            node_map = _local_search(ex, node_map, n_nodes, rng,
                                     orc_nd.fanout, n_sweeps, n_cand=12)
            out = np.empty(E, dtype=np.int32)
            reach = np.bincount(ex.ravel(), minlength=E).astype(np.float64)
            for nd in range(n_nodes):
                members = np.flatnonzero(node_map == nd)
                if members.size == 0:
                    continue
                # The last node may be partially populated (W need not be a
                # multiple of gpus_per_node), so the local width is clamped to
                # the ranks that actually exist there. Without this the mapping
                # can emit a rank id >= world_size.
                base = nd * gpus_per_node
                width = max(1, min(gpus_per_node, W - base))
                # balance TOKEN REACH inside the node, not selection count:
                # under dedup the arriving message count is what saturates
                local = _lpt_map(reach[members], width,
                                 capacity=int(np.ceil(members.size / width)))
                out[members] = base + local
            rows.append(out)

        return _check(Placement(np.stack(rows), W, "per_layer", layers=fit.layers,
                         name="hierarchical_layer"), W)

    if kind == "adversarial":
        # maximise fan-out: spread the strongest co-activated pairs apart
        As = layer_affinities(fit, "cooccurrence")
        rows = []
        for l in fit.layers:
            A = As[int(l)]
            p = _greedy_cluster(-A, W)      # cluster on NEGATIVE affinity
            rows.append(p)
        return _check(Placement(np.stack(rows), W, "per_layer", layers=fit.layers,
                         name="adversarial"), W)

    raise ValueError(f"unknown placement kind {kind!r}")


def _check(p: "Placement", W: int) -> "Placement":
    """Reject any generator that emits an out-of-range rank id.

    Cheap invariant, but it caught a real bug: the hierarchical builder mapped
    node-local slots as ``node*gpus_per_node + local`` without clamping the
    last, partially-populated node, producing rank ids >= world_size whenever
    world_size was not a multiple of gpus_per_node.
    """
    m = p.expert_to_rank
    if m.min() < 0 or m.max() >= W:
        raise ValueError(f"{p.name}: rank ids outside [0,{W}) "
                         f"(min={m.min()}, max={m.max()})")
    return p


PLACEMENT_KINDS = (
    "linear", "random", "load_balanced", "load_balanced_layer",
    "affinity_global", "affinity_layer", "fanout_layer",
    "balanced_affinity_layer", "bottleneck_layer",
    "affinity_coordinated_layer",
    "hierarchical_layer", "adversarial",
)


# ═══════════════════════════════════════════════════════════════════════
# Controlled perturbations — for the placement-invariance experiment
# ═══════════════════════════════════════════════════════════════════════

def perturb(base: Placement, kind: str, fit: CellTable, seed: int = 0
            ) -> Placement:
    """Systematic perturbations of an existing placement.

    Used by the invariance experiment: routing must be bit-identical across all
    of these while cost must move, so each one is a separate observation of
    "same routing, different cost".
    """
    rng = np.random.default_rng(seed)
    m = base.expert_to_rank.copy()
    W = base.world_size

    def _apply(row: np.ndarray) -> np.ndarray:
        row = row.copy()
        E = row.shape[0]
        if kind == "swap_two":
            a, b = rng.choice(E, 2, replace=False)
            row[a], row[b] = row[b], row[a]
        elif kind == "swap_many":
            for _ in range(max(1, E // 8)):
                a, b = rng.choice(E, 2, replace=False)
                row[a], row[b] = row[b], row[a]
        elif kind == "full_permute":
            row = row[rng.permutation(E)]
        elif kind == "move_hottest":
            load = fit.expert_load()
            hot = int(np.argmax(load))
            targets = np.flatnonzero(row != row[hot])
            if targets.size:
                b = int(rng.choice(targets))
                row[hot], row[b] = row[b], row[hot]
        elif kind == "split_top_pair":
            A = pooled_affinity(fit, "cooccurrence")
            i, j = np.unravel_index(np.argmax(A), A.shape)
            if row[i] == row[j]:
                targets = np.flatnonzero(row != row[i])
                if targets.size:
                    b = int(rng.choice(targets))
                    row[j], row[b] = row[b], row[j]
        else:
            raise ValueError(f"unknown perturbation {kind!r}")
        return row

    if m.ndim == 1:
        m = _apply(m)
    else:
        m = np.stack([_apply(r) for r in m])
    return _check(Placement(m, W, base.scope, layers=base.layers,
                     name=f"{base.name}+{kind}.s{seed}"), W)


PERTURBATIONS = ("swap_two", "swap_many", "full_permute", "move_hottest",
                 "split_top_pair")
