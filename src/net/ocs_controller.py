"""ocs_controller.py -- who gets a circuit, and when the configuration changes.

The controller is the only component that decides G, and the policies are deliberately
separated by *what they are allowed to know*, because that difference is the whole
question:

    none            no circuits ever.  The pure-EPS control.
    static_hot      fit G once on the calibration collectives, then hold it forever.
                    The deployed reading: configure once, reuse across the run.
    epoch_hot       refit G every `epoch_collectives` from the traffic actually seen
                    since the last decision -- no foresight.
    oracle_hot      refit G on the collective that is about to run.  An upper bound,
                    not a deployable policy.
    threshold       the screening rule: rank candidate pairs by the congestion they
                    actually suffered (bytes x the rate they lost), and admit only
                    while the port budget lasts.

Every policy returns a *set of pairs*; the fabric turns that into an epoch and charges
the reconfiguration.  A policy that returns the same set changes nothing and costs
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.eval.cost_model import Tier


@dataclass
class ControllerConfig:
    policy: str = "static_hot"
    n_circuits: int = 16
    ports_per_rank: int = 2
    epoch_collectives: int = 0          # 0 = configure once and hold
    calibration_collectives: int = 1
    promote_from: tuple = (Tier.CROSS_POD,)
    promotable: frozenset = frozenset()   # pairs a circuit can actually serve


@dataclass
class OcsController:
    cfg: ControllerConfig = field(default_factory=ControllerConfig)
    _accum: dict = field(default_factory=dict)      # pair -> bytes since last decision
    _loss: dict = field(default_factory=dict)       # pair -> bytes x lost-rate-fraction
    _fitted: set = field(default_factory=set)
    decisions: list = field(default_factory=list)

    # ── observation ──────────────────────────────────────────────────
    def promotable_pairs(self, T) -> frozenset:
        """Which unordered pairs a circuit could serve, from the tier matrix."""
        W = T.shape[0]
        allowed = {int(x) for x in self.cfg.promote_from}
        return frozenset(frozenset((a, b)) for a in range(W) for b in range(a + 1, W)
                         if int(T[a, b]) in allowed)

    def observe_path_rates(self, flows) -> None:
        """Accumulate what each pair actually suffered on EPS this collective."""
        for f in flows:
            if f.path != "EPS" or f.tier != Tier.CROSS_POD.name:
                continue
            key = f.pair
            self._accum[key] = self._accum.get(key, 0.0) + f.bytes
            lost = getattr(f, "lost_fraction", 0.0)
            self._loss[key] = self._loss.get(key, 0.0) + f.bytes * lost

    def observe_matrix(self, mat: np.ndarray) -> None:
        """Accumulate demand **only on pairs a circuit could serve**.

        Without this filter the largest pairs in a real MoE trace are intra-node
        (NVLink) pairs; a bytes-ranked top-k then spends the whole circuit budget on
        pairs that are not promotable, and the OCS carries almost nothing.
        """
        for a in range(mat.shape[0]):
            for b in range(a + 1, mat.shape[0]):
                key = frozenset((a, b))
                if self.cfg.promotable and key not in self.cfg.promotable:
                    continue
                v = float(mat[a, b] + mat[b, a])
                if v > 0:
                    self._accum[key] = self._accum.get(key, 0.0) + v

    # ── decisions ────────────────────────────────────────────────────
    def _rank(self, pairs_bytes: dict, metric: str) -> list:
        items = list(pairs_bytes.items())
        if metric == "loss":
            items.sort(key=lambda kv: -self._loss.get(kv[0], 0.0))
        else:
            items.sort(key=lambda kv: -kv[1])
        return [p for p, _ in items]

    def _budgeted(self, ranked: list, degree: dict = None) -> set:
        """Apply the two real constraints: circuit budget and ports per rank."""
        degree = {} if degree is None else degree
        chosen = set()
        for p in ranked:
            if len(chosen) >= self.cfg.n_circuits:
                break
            a, b = min(p), max(p)
            if degree.get(a, 0) >= self.cfg.ports_per_rank:
                continue
            if degree.get(b, 0) >= self.cfg.ports_per_rank:
                continue
            chosen.add(p)
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        return chosen

    def decide(self, k: int, current_pairs: set, current_mat: np.ndarray | None) -> set | None:
        """Return the configuration to hold for collective k, or None to keep it."""
        pol = self.cfg.policy
        if pol == "none":
            return set() if current_pairs else None
        if pol == "oracle_hot" and current_mat is not None:
            ranked = self._rank({frozenset((a, b)): float(current_mat[a, b] + current_mat[b, a])
                                 for a in range(current_mat.shape[0])
                                 for b in range(a + 1, current_mat.shape[0])
                                 if current_mat[a, b] + current_mat[b, a] > 0
                                 and (not self.cfg.promotable
                                      or frozenset((a, b)) in self.cfg.promotable)}, "bytes")
            return self._budgeted(ranked)
        if pol == "static_hot":
            if k < self.cfg.calibration_collectives:
                return None
            if self._fitted:
                return None
            self._fitted = self._budgeted(self._rank(self._accum, "bytes"))
            if not self._fitted:
                return None
            self.decisions.append({"collective": k, "n": len(self._fitted)})
            return set(self._fitted)
        if pol in ("epoch_hot", "threshold"):
            period = max(1, self.cfg.epoch_collectives)
            if k < self.cfg.calibration_collectives or (k % period) != 0:
                return None
            metric = "loss" if pol == "threshold" else "bytes"
            ranked = self._rank(self._accum, metric)
            new = self._budgeted(ranked)
            self._accum, self._loss = {}, {}
            if new == current_pairs:
                return None
            self.decisions.append({"collective": k, "n": len(new)})
            return new
        return None


def make_controller(cfg: ControllerConfig) -> OcsController:
    return OcsController(cfg=cfg)
