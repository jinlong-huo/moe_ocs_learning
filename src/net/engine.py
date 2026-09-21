"""engine.py -- the inference execution loop, with the network inside it.

    layer -> compute -> collective -> network -> collective finishes -> next layer

Time is nanoseconds.  Between two events the active flow set is fixed, so each flow's
rate is fixed, and the fluid integration is exact: the engine jumps straight to the
next completion (or to the next sampling boundary) instead of stepping.

Per flow, the rate is max-min fair over four resources:

    egress port at the source      NIC 50 GB/s (EPS/OCS) | NVLink 450 | pod 50
    ingress port at the destination same table
    path resource                  EPS: the pooled core  | OCS: the circuit
                                   = port_rate * W / sigma   = full NIC rate

The EPS core is pooled and shared, so its utilisation is `min(1, demand/capacity)`:
under light load a flow runs at full port rate, under saturation it is scaled down --
which is precisely the behaviour the static cost model cannot represent.  A circuit
carries its pair at the full NIC rate and **does not draw on the pool**, so promoting a
pair removes its bytes from the core rather than adding bandwidth to it.

Everything observed is written out: GPU states as they change, fabric state at every
sampling boundary, circuit epochs, and one record per collective.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from src.net.collective import Flow, flows_for_matrix
from src.net.fabric import EpsFabric, OcsFabric


@dataclass
class EngineConfig:
    compute_us_per_layer: float = 0.0     # 0 = communication-only, as in completion.py
    include_combine: bool = True
    sample_ns: float = 2_000.0            # fabric state sampling interval (2 us)
    record_gpu_states: bool = True
    setup_free: bool = True               # the first establishment happens before t=0
    wall_seconds_per_layer_guess: float = 0.0


@dataclass
class RunResult:
    gpu_states: list = field(default_factory=list)      # rank, t_ns, state, collective, n_send, n_recv
    fabric_samples: list = field(default_factory=list)  # time series of aggregate state
    collectives: list = field(default_factory=list)
    circuits: list = field(default_factory=list)        # per epoch per circuit
    epochs: list = field(default_factory=list)
    reconfigurations: list = field(default_factory=list)
    makespan_ns: float = 0.0
    per_rank_busy_ns: list = field(default_factory=list)
    per_rank_bytes: list = field(default_factory=list)


class InferenceNetEngine:
    def __init__(self, W: int, T: np.ndarray, eps: EpsFabric, ocs: OcsFabric,
                 controller, cfg: EngineConfig | None = None):
        self.W, self.T = W, T
        self.eps, self.ocs = eps, ocs
        self.ctrl = controller
        self.cfg = cfg or EngineConfig()
        self.t = 0.0
        self.res = RunResult()
        self.res.per_rank_busy_ns = [0.0] * W
        self.res.per_rank_bytes = [0.0] * W
        self._gpu_state = ["IDLE"] * W
        self._next_sample = 0.0
        self._coll = 0
        self._fid = 0
        self._core_bytes_inflight = 0.0

    # ── small helpers ────────────────────────────────────────────────
    def _port_rate(self, path: str) -> float:
        if path == "NVLINK":
            return self.eps.nvlink_gbytes_per_s
        if path == "POD":
            return self.eps.pod_gbytes_per_s
        return self.eps.port_gbytes_per_s

    def _gpu_set(self, active, state: str, coll: int):
        """Recompute every rank's state; append a row only where it changed."""
        if not self.cfg.record_gpu_states:
            return
        n_send = Counter(f.src for f in active)
        n_recv = Counter(f.dst for f in active)
        for r in range(self.W):
            if state == "COMPUTE":
                s = "COMPUTE"
            elif n_send.get(r, 0) and n_recv.get(r, 0):
                s = "SEND+RECV"
            elif n_send.get(r, 0):
                s = "SEND"
            elif n_recv.get(r, 0):
                s = "RECV"
            else:
                s = "IDLE"
            if s != self._gpu_state[r]:
                self._gpu_state[r] = s
                self.res.gpu_states.append({"rank": r, "t_ns": round(self.t, 1), "state": s,
                                            "collective": coll, "n_send": n_send.get(r, 0),
                                            "n_recv": n_recv.get(r, 0)})

    def _rates(self, active) -> dict:
        """Max-min fair rate per flow over egress, ingress and path resources."""
        by_src = Counter(f.src for f in active)
        by_dst = Counter(f.dst for f in active)
        idl = {}
        for f in active:
            r = self._port_rate(f.path) / by_src[f.src]
            r = min(r, self._port_rate(f.path) / by_dst[f.dst])
            idl[f.fid] = r
        # EPS: pooled core, scaled only if the offered demand exceeds the pool
        core = [f for f in active if f.path == "EPS"]
        demand = sum(idl[f.fid] for f in core)
        cap = self.eps.core_gbytes_per_s
        scale = min(1.0, cap / demand) if demand > 0 else 1.0
        rates = {}
        per_circuit = Counter(f.pair for f in active if f.path == "OCS")
        for f in active:
            if f.path == "EPS":
                rates[f.fid] = idl[f.fid] * scale
            elif f.path == "OCS":
                rates[f.fid] = min(idl[f.fid],
                                   self.ocs.cfg.rate_gbytes_per_s / per_circuit[f.pair])
            else:
                rates[f.fid] = idl[f.fid]
        return rates, scale

    def _sample(self, active, rates, coll: int, label: str = ""):
        core = [f for f in active if f.path == "EPS"]
        ocs = [f for f in active if f.path == "OCS"]
        core_bw = sum(rates[f.fid] for f in core)
        ocs_bw = sum(rates[f.fid] for f in ocs)
        rec = {
            "t_ns": round(self.t, 1), "collective": coll, "label": label,
            "n_active_flows": len(active),
            "n_eps_flows": len(core), "n_ocs_flows": len(ocs),
            "core_bw_gbps": round(core_bw, 4),
            "core_utilisation": round(min(1.0, core_bw / self.eps.core_gbytes_per_s), 4),
            "core_queue_mb": round(sum(f.remaining for f in core) / 1e6, 4),
            "ocs_bw_gbps": round(ocs_bw, 4),
            "ocs_bytes_carried_mb": round(sum(f.bytes - f.remaining for f in ocs) / 1e6, 4),
            "port_queue_mb_max": round(max((sum(g.remaining for g in active if g.src == r)
                                            for r in range(self.W)), default=0.0) / 1e6, 4),
            "n_dark": 1 if self.ocs.is_dark(self.t) else 0,
        }
        self.res.fabric_samples.append(rec)
        return rec

    # ── one collective, possibly split by a reconfiguration ──────────
    def run_collective(self, mat: np.ndarray, phase: str, cid: int, current_mat=None) -> dict:
        started = self.t
        flows = flows_for_matrix(mat, self.T, cid, self.t, self.ocs,
                                 self.cfg.include_combine, fid_start=self._fid)
        self._fid += len(flows)
        if phase == "combine":
            flows = [f for f in flows if f.phase == "combine"]
        else:
            flows = [f for f in flows if f.phase == "dispatch"]
        bytes_total = sum(f.bytes for f in flows)
        peak_core_util, peak_queue = 0.0, 0.0
        stage_deadline = None
        if self.ocs.is_dark(self.t):
            stage_deadline = self.ocs.dark_until_ns

        active = list(flows)
        for f in active:
            f.remaining = f.bytes
            f.start_ns = self.t
        self._gpu_set(active, "NET", cid)
        while active:
            rates, scale = self._rates(active)
            for f in active:
                f.lost_fraction = 1.0 - rate_fraction(self.eps, f, rates[f.fid])
            # units: GB/s == bytes/ns, so dt is already ns and rate*dt is bytes
            dt_flow = min(f.remaining / rates[f.fid] for f in active
                          if rates[f.fid] > 0)
            dt_flow = max(dt_flow, 1e-9)
            dt_bound = float("inf")
            if stage_deadline is not None:
                dt_bound = max(0.0, stage_deadline - self.t)
            if self.t + dt_flow > self._next_sample:
                dt_bound = min(dt_bound, self._next_sample - self.t)
            dt = min(dt_flow, dt_bound)
            if dt <= 0:
                dt = dt_flow
            for f in active:
                f.remaining -= rates[f.fid] * dt
                self.res.per_rank_bytes[f.src] += rates[f.fid] * dt
                if f.path == "OCS":
                    cs = self.ocs.circuits.get(f.pair)
                    if cs is not None:
                        cs.bytes_carried += rates[f.fid] * dt
                        cs.busy_ns += dt
                        cs.last_active_ns = self.t + dt
                self.res.per_rank_busy_ns[f.src] += dt if rates[f.fid] > 0 else 0.0
            self.t += dt
            cur = self._sample(active, rates, cid)
            peak_core_util = max(peak_core_util, cur["core_utilisation"])
            peak_queue = max(peak_queue, cur["port_queue_mb_max"])
            done = [f for f in active if f.remaining <= 1e-6]
            for f in done:
                f.finish_ns = self.t
            if done:
                active = [f for f in active if f.remaining > 1e-6]
            if self.t >= self._next_sample:
                self._next_sample = self.t + self.cfg.sample_ns
            # a reconfiguration that came due mid-collective: everything OCS is dark,
            # so unfinished flows fall back to EPS and are re-routed when it lifts
            if stage_deadline is not None and self.t >= stage_deadline - 1e-9:
                stage_deadline = None
                for f in active:
                    if f.path == "OCS" and not self.ocs.has_circuit(f.src, f.dst, self.t):
                        f.path = "EPS"
                self._gpu_set(active, "NET", cid)
        self._gpu_set(active, "NET", cid)
        eps_bytes = sum(f.bytes for f in flows if f.path == "EPS")
        ocs_bytes = sum(f.bytes for f in flows if f.path == "OCS")
        rec = {"collective": cid, "phase": phase, "start_ns": round(started, 1),
               "finish_ns": round(self.t, 1), "network_ns": round(self.t - started, 1),
               "bytes": bytes_total, "n_flows": len(flows),
               "eps_bytes": eps_bytes, "ocs_bytes": ocs_bytes,
               "peak_core_utilisation": round(peak_core_util, 4),
               "peak_port_queue_mb": round(peak_queue, 4)}
        self.res.collectives.append(rec)
        return rec

    # ── the inference execution loop ─────────────────────────────────
    def run(self, layer_matrices: list, passes: int = 1) -> RunResult:
        compute_ns = self.cfg.compute_us_per_layer * 1e3
        for p in range(passes):
            for li, mat in enumerate(layer_matrices):
                self.t += compute_ns
                self._gpu_set([], "COMPUTE", self._coll)
                for phase in ("dispatch", "combine"):
                    if phase == "combine" and not self.cfg.include_combine:
                        continue
                    want = self.ctrl.decide(self._coll, self.ocs.active_pairs(), mat)
                    if want is not None:
                        first = self.ocs.generation == 0
                        info = self.ocs.reconfigure(
                            want, self.t, free=self.cfg.setup_free and first)
                        if info["changed"]:
                            self.res.reconfigurations.append(
                                {"collective": self._coll, "t_ns": round(self.t, 1),
                                 **{k: (v if not isinstance(v, float) else round(v, 1))
                                    for k, v in info.items()},
                                 "pairs": sorted(tuple(sorted(x)) for x in want)})
                    self.run_collective(mat, phase, self._coll)
                    self.ctrl.observe_matrix(mat)
                    self._coll += 1
        self.res.makespan_ns = self.t
        self._close_epochs()
        return self.res

    def _close_epochs(self) -> None:
        for gen_info in self.ocs.epochs:
            for pr in gen_info["pairs"]:
                key = frozenset(pr)
                cs = None
                self.res.circuits.append({"generation": gen_info["generation"],
                                          "src": pr[0], "dst": pr[1],
                                          "epoch_start_ns": gen_info["start_ns"],
                                          "epoch_end_ns": gen_info["end_ns"]})
        for p, cs in self.ocs.circuits.items():
            a, b = sorted(p)
            self.res.circuits.append({"generation": cs.epoch, "src": a, "dst": b,
                                      "epoch_start_ns": cs.established_ns,
                                      "epoch_end_ns": round(self.t, 1),
                                      "bytes_carried": round(cs.bytes_carried, 1),
                                      "busy_ns": round(cs.busy_ns, 1)})


def rate_fraction(eps: EpsFabric, flow, rate_gbps: float) -> float:
    """What fraction of its unconstrained rate the flow is actually getting."""
    nominal = {"NVLINK": eps.nvlink_gbytes_per_s,
               "POD": eps.pod_gbytes_per_s}.get(flow.path, eps.port_gbytes_per_s)
    return min(1.0, rate_gbps / nominal) if nominal else 1.0
