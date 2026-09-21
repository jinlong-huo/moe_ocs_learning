#!/usr/bin/env python3
"""gpu_state_status.py -- the per-GPU status report for one MoE inference execution.

The question this answers
-------------------------
scripts/inference_net_trace.py runs the inference execution loop with the fabric
inside it and dumps the raw event stream (a change log: a row only when some rank
changed state).  That is the right raw record and the wrong shape for reading:
nobody wants a change log, they want "what was rank 17 doing, and for how long".

This script turns the event stream into the report:

  per rank   time and fraction spent in COMPUTE / SEND / RECV / SEND+RECV / IDLE,
             bytes and flows sent, and a strip chart over the whole execution
  per collective  how many ranks were sending, receiving, both, or *idle*, plus
             the fabric state (EPS core utilisation, OCS bytes carried)
  sensitivity  the whole report again at a different per-layer compute time,
             because compute is a parameter this repo measures nowhere

Units are microseconds throughout the report; the engine works in nanoseconds.

Every number is produced by the real captured routing of the named model (the
--workload manifest), a stated placement, and the stated fabric constants.  The
one input that is *not* measured is --compute-us, the per-layer compute time; the
repo has no kernel timings, so it is swept rather than asserted, and every row
says which value produced it.

Usage
    python3.12 scripts/gpu_state_status.py --workload logs/workload/qwen36 \
        --world-size 32 --layers 0,1,2,3 --policies none,oracle_hot \
        --compute-sweep 0,100,1650 --out outputs/net_status
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "scripts"))

import inference_net_trace as intrace  # noqa: E402

STATES = ("COMPUTE", "SEND", "RECV", "SEND+RECV", "IDLE")
STRIP = {"COMPUTE": "C", "SEND": "S", "RECV": "R", "SEND+RECV": "B", "IDLE": "."}


# -- turning the change log into intervals ------------------------------------
def compute_windows(res, compute_ns):
    """The windows in which every rank computes, reconstructed from the schedule.

    The engine advances the clock by the per-layer compute time and records the
    COMPUTE state *at the moment the following collective starts*, so the change
    log alone cannot date the compute intervals.  They are exactly the gaps
    between one collective's finish and the next dispatch's start, which the
    collective records do carry -- so they are reconstructed here rather than
    assumed, and the report stays correct whatever the engine logs.
    """
    if compute_ns <= 0:
        return []
    wins, prev_finish = [], 0.0
    for c in res.collectives:
        # a compute window opens where the previous collective ended -- the
        # previous collective being the combine, not the previous dispatch
        if c["phase"] == "dispatch" and c["start_ns"] > prev_finish:
            wins.append((prev_finish, c["start_ns"]))
        prev_finish = max(prev_finish, c["finish_ns"])
    if res.makespan_ns > prev_finish:
        wins.append((prev_finish, res.makespan_ns))
    return wins


def merge_with_compute(ivals, windows):
    """Cut a rank's state intervals so that compute windows win."""
    if not windows:
        return list(ivals)
    bounds = sorted({t for iv in ivals for t in iv[:2]} | {t for w in windows for t in w})
    out = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        if any(w0 <= a < w1 for w0, w1 in windows):
            st = "COMPUTE"
        else:
            st = state_at(ivals, a)
        if out and out[-1][2] == st and abs(out[-1][1] - a) < 1e-9:
            out[-1] = (out[-1][0], b, st)
        else:
            out.append((a, b, st))
    return out


def per_rank_intervals(res, W):
    """[[(t0, t1, state), ...] for each rank] -- the change log, stitched."""
    changes = {r: [] for r in range(W)}
    for row in res.gpu_states:
        changes[row["rank"]].append((row["t_ns"], row["state"]))
    out = []
    for r in range(W):
        rows = sorted(changes[r])
        ivals = []
        for i, (t, st) in enumerate(rows):
            t1 = rows[i + 1][0] if i + 1 < len(rows) else res.makespan_ns
            if t1 > t:
                ivals.append((t, t1, st))
        out.append(ivals)
    return out


def state_at(ivals, t):
    cur = "IDLE"
    for t0, t1, st in ivals:
        if t0 <= t < t1:
            return st
        if t0 > t:
            break
        cur = st
    return cur


def occupancy(res, W, windows=None):
    """Per rank: time in each state, plus the derived columns."""
    ivals = per_rank_intervals(res, W)
    if windows:
        ivals = [merge_with_compute(iv, windows) for iv in ivals]
    span = res.makespan_ns if res.makespan_ns > 0 else 1.0
    rows = []
    for r in range(W):
        acc = {s: 0.0 for s in STATES}  # noqa: E501
        for t0, t1, st in ivals[r]:
            acc[st if st in acc else "IDLE"] += t1 - t0
        busy = acc["COMPUTE"] + acc["SEND"] + acc["RECV"] + acc["SEND+RECV"]
        rows.append({
            "rank": r,
            **{s.lower().replace("+", "_") + "_us": round(acc[s] / 1e3, 3) for s in STATES},
            **{s.lower().replace("+", "_") + "_pct": round(100.0 * acc[s] / span, 3) for s in STATES},
            "busy_pct": round(100.0 * busy / span, 3),
            "bytes_sent_mb": round(res.per_rank_bytes[r] / 1e6, 3),
        })
    return rows, ivals


def collective_census(res, W, ivals):
    """Who was doing what, per collective."""
    out = []
    for c in res.collectives:
        if c["finish_ns"] <= c["start_ns"]:
            continue
        t = 0.5 * (c["start_ns"] + c["finish_ns"])
        counts = {s: 0 for s in STATES}
        for r in range(W):
            counts[state_at(ivals[r], t)] += 1
        out.append({
            "collective": c["collective"], "phase": c["phase"],
            "start_us": round(c["start_ns"] / 1e3, 3),
            "network_us": round(c["network_ns"] / 1e3, 3),
            "n_flows": c["n_flows"],
            "ranks_send": counts["SEND"] + counts["SEND+RECV"],
            "ranks_recv": counts["RECV"] + counts["SEND+RECV"],
            "ranks_send_recv": counts["SEND+RECV"],
            "ranks_idle": counts["IDLE"],
            "bytes_mb": round(c["bytes"] / 1e6, 3),
            "ocs_bytes_mb": round(c["ocs_bytes"] / 1e6, 3),
            "peak_core_util": c["peak_core_utilisation"],
        })
    return out


def strip_chart(ivals, W, width=100, t_end=None):
    """One line per rank: what that GPU was doing, as a function of time."""
    t_end = t_end or max((iv[-1][1] for iv in ivals if iv), default=1.0)
    lines = []
    for r in range(W):
        chars = []
        for k in range(width):
            t = t_end * (k + 0.5) / width
            chars.append(STRIP.get(state_at(ivals[r], t), "?"))
        lines.append("".join(chars))
    return lines, t_end


def fabric_extremes(res):
    s = res.fabric_samples
    if not s:
        return {}
    core = [x["core_utilisation"] for x in s]
    ocs = [x["ocs_bw_gbps"] for x in s]
    fl = [x["n_active_flows"] for x in s]
    dark = sum(x["n_dark"] for x in s)
    return {
        "n_samples": len(s),
        "core_util_mean": round(float(np.mean(core)), 4),
        "core_util_max": round(float(np.max(core)), 4),
        "core_util_saturated_pct": round(100.0 * float(np.mean([c >= 0.999 for c in core])), 2),
        "ocs_gbps_mean": round(float(np.mean(ocs)), 4),
        "ocs_gbps_max": round(float(np.max(ocs)), 4),
        "active_flows_mean": round(float(np.mean(fl)), 1),
        "active_flows_max": int(np.max(fl)),
        "samples_in_dark": int(dark),
    }


# -- the counterfactual that decides whether a circuit can help at all ---------
def install_port_exemption():
    """Make an OCS circuit use its own port instead of the electrical NIC port.

    The stock engine shares one 50 GB/s egress port across *every* flow a rank has
    in flight, so a promoted pair still waits behind the same port and promotion
    cannot change the finish time -- which is what the runs below measure.  This
    variant charges OCS flows to a dedicated optical port (the budget
    ports_per_rank already models), leaving the electrical flows on the NIC.  The
    difference between the two is the value of a circuit, in this model.
    """
    from collections import Counter
    from src.net.engine import InferenceNetEngine

    class PortExemptEngine(InferenceNetEngine):
        def _rates(self, active):
            elec = [f for f in active if f.path != "OCS"]
            opt = [f for f in active if f.path == "OCS"]
            share_src = Counter(f.src for f in elec)
            share_dst = Counter(f.dst for f in elec)
            ocs_src = Counter(f.src for f in opt)
            ocs_dst = Counter(f.dst for f in opt)
            idl = {}
            for f in elec:
                r = self._port_rate(f.path) / share_src[f.src]
                idl[f.fid] = min(r, self._port_rate(f.path) / share_dst[f.dst])
            for f in opt:
                r = self._port_rate("OCS") / ocs_src[f.src]
                idl[f.fid] = min(r, self._port_rate("OCS") / ocs_dst[f.dst])
            core = [f for f in elec if f.path == "EPS"]
            demand = sum(idl[f.fid] for f in core)
            cap = self.eps.core_gbytes_per_s
            scale = min(1.0, cap / demand) if demand > 0 else 1.0
            per_circuit = Counter(f.pair for f in opt)
            rates = {}
            for f in active:
                if f.path == "EPS":
                    rates[f.fid] = idl[f.fid] * scale
                elif f.path == "OCS":
                    rates[f.fid] = min(idl[f.fid],
                                       self.ocs.cfg.rate_gbytes_per_s / per_circuit[f.pair])
                else:
                    rates[f.fid] = idl[f.fid]
            return rates, scale

    intrace.InferenceNetEngine = PortExemptEngine


def load_layers_phase(args, phase):
    """Real trace -> one byte matrix per MoE layer, for the named pass.

    Mirrors inference_net_trace.load_layers, with the one difference that the
    phase is selectable: a prefill pass and a decode pass are not the same
    collective (prefill carries every prompt token, decode carries one token per
    sequence), so the GPU status they produce is not the same either.
    """
    import src.eval.cost_model as cm
    from src.eval.cost_model import (CostConfig, DispatchMode, FabricConfig, Tier,
                                     Topology, hierarchy_for, traffic_matrix)
    from src.eval.placement_opt import make_placement
    from src.eval.trace_ir import load_workload
    from ownership_ocs_envelope import patch_token_rank

    W = args.world_size
    cm.token_rank_original = cm.token_rank
    patch_token_rank(args.source_model, args.k)
    t_all = load_workload(args.workload / "manifest.json", decode_only=False)
    if phase == "prefill":
        t_all = t_all.select(t_all.phase == 0)
    elif phase == "decode":
        t_all = t_all.select(t_all.phase == 1)
    base = hierarchy_for(W, "multi_pod")
    fab = FabricConfig(core_oversubscription=args.sigma, pod_oversubscription=1.0)
    topo = Topology(W, base.gpus_per_node, base.nodes_per_pod, fab, set(),
                    base.rank_to_slot, (Tier.CROSS_POD,))
    pl = make_placement(args.placement, t_all, W, seed=args.seed)
    cost = CostConfig(hidden_size=2048)
    B = cost.hidden_size * cost.dtype_bytes
    layers = [int(x) for x in np.sort(np.unique(t_all.layers))]
    if args.layers:
        want = [int(x) for x in args.layers.split(",") if x.strip() != ""]
        layers = [x for x in layers if x in want] or layers[:len(want)]
    mats = []
    for L in layers:
        sub = t_all.by_layer(L)
        tm = traffic_matrix(sub, pl, topo, DispatchMode.DEDUP_RANK, n_dp=W, seed=args.seed)
        m = (tm.counts * B).astype(np.float64)
        np.fill_diagonal(m, 0)
        mats.append(m)
    n_cells = int(t_all.n_cells)
    return topo, mats, layers, n_cells


def make_args(ns):
    base = dict(world_size=32, source_model="measured_packed_4", placement="linear",
                k=4, layers="0,1,2,3", passes=1, n_circuits=16, ports_per_rank=2,
                sigma=4.0, port_rate=50.0, reconfig_us=0.0, epoch_collectives=8,
                calibration=1, compute_us=0.0, sample_ns=2000.0, no_combine=False,
                charge_setup=False, policies="", seed=0)
    base.update(ns)
    return SimpleNamespace(**base)


def analyse(policy, args, topo, mats, out_dir, tag_prefix=""):
    out = intrace.run_one(policy, args, topo, mats)
    res, summary = out["res"], out["summary"]
    W = args.world_size
    windows = compute_windows(res, args.compute_us * 1e3)
    occ, ivals = occupancy(res, W, windows)
    census = collective_census(res, W, ivals)
    lines, t_end = strip_chart(ivals, W)
    fab = fabric_extremes(res)
    tag = tag_prefix + policy + "_eps" + str(args.n_circuits) + "_c" + str(int(args.compute_us))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / (tag + ".occupancy.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(occ[0].keys()))
        w.writeheader(); w.writerows(occ)
    with open(out_dir / (tag + ".collective_census.csv"), "w", newline="") as fh:
        if census:
            w = csv.DictWriter(fh, fieldnames=list(census[0].keys()))
            w.writeheader(); w.writerows(census)
    return {"tag": tag, "policy": policy, "compute_us": args.compute_us,
            "summary": summary, "fabric": fab, "occupancy": occ,
            "census": census, "strip": lines, "t_end_ns": t_end, "res": res}


def report(block):
    s, occ, census = block["summary"], block["occupancy"], block["census"]
    L = []
    L.append("-- " + block["tag"] + "  (compute_us_per_layer = " + str(block["compute_us"]) + ")")
    L.append("   execution window   {:,.1f} us over {} collectives ({:,.2f} ms network)".format(
        block["t_end_ns"] / 1e3, s["n_collectives"], s["network_total_us"] / 1e3))
    L.append("   network per coll.  mean {:,.1f} us   p50 {:,.1f}   max {:,.1f}".format(
        s["network_mean_us"], s["network_p50_us"], s["network_max_us"]))
    L.append("   bytes              EPS {:,.1f} MB   OCS {:,.1f} MB   share {:.1f}%".format(
        s["eps_bytes"] / 1e6, s["ocs_bytes"] / 1e6, 100 * s["ocs_share"]))
    dark_us = s.get("dark_wall_us", s.get("reconfig_sum_us", s.get("reconfig_total_us", 0.0)))
    L.append("   reconfigurations   {} ({:,.3f} ms dark wall, {:.1f}% of the run)".format(
        s["n_reconfigurations"], dark_us / 1e3,
        100.0 * s.get("dark_fraction_of_run", 0.0)))
    f = block["fabric"]
    if f:
        L.append("   fabric             EPS core util mean {:.3f}, max {:.3f}, saturated in {:.1f}% of samples; "
                 "active flows mean {}, max {}".format(f["core_util_mean"], f["core_util_max"],
                                                       f["core_util_saturated_pct"],
                                                       f["active_flows_mean"], f["active_flows_max"]))
        L.append("                      OCS carried mean {:.2f} GB/s, max {:.2f} GB/s, dark samples {}".format(
            f["ocs_gbps_mean"], f["ocs_gbps_max"], f["samples_in_dark"]))
    L.append("   {:>4} {:>9} {:>8} {:>8} {:>8} {:>8} {:>7} {:>9}".format(
        "rank", "COMPUTE", "SEND", "RECV", "S+R", "IDLE", "busy%", "MB sent"))
    for r in occ:
        L.append("   {:>4} {:>8.1f}% {:>7.1f}% {:>7.1f}% {:>7.1f}% {:>7.1f}% {:>6.1f}% {:>9.3f}".format(
            r["rank"], r["compute_pct"], r["send_pct"], r["recv_pct"],
            r["send_recv_pct"], r["idle_pct"], r["busy_pct"], r["bytes_sent_mb"]))
    idle = [r["rank"] for r in occ if r["idle_pct"] > 99.0]
    full = [r["rank"] for r in occ if r["busy_pct"] > 99.0]
    L.append("   ranks idle >99% of the run: {}{}".format(
        len(idle), ("  " + str(idle[:12]) + ("..." if len(idle) > 12 else "")) if idle else ""))
    L.append("   ranks busy >99% of the run: {}{}".format(
        len(full), ("  " + str(full[:12]) + ("..." if len(full) > 12 else "")) if full else ""))
    if census:
        L.append("   {:>4} {:>8} {:>10} {:>9} {:>6} {:>5} {:>5} {:>5} {:>5} {:>8} {:>5}".format(
            "coll", "phase", "start us", "net us", "flows", "send", "recv", "both", "IDLE", "MB", "core"))
        for c in census:
            L.append("   {:>4} {:>8} {:>10,.1f} {:>9,.1f} {:>6} {:>5} {:>5} {:>5} {:>5} {:>8.3f} {:>5.2f}".format(
                c["collective"], c["phase"], c["start_us"], c["network_us"], c["n_flows"],
                c["ranks_send"], c["ranks_recv"], c["ranks_send_recv"], c["ranks_idle"],
                c["bytes_mb"], c["peak_core_util"]))
    L.append("   strip chart (C compute, S send, R recv, B both, . idle):")
    for r, line in enumerate(block["strip"]):
        L.append("   r{:>2} |{}|".format(r, line))
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--phase", choices=["prefill", "decode", "both"], default="decode")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--policies", default="none,oracle_hot")
    ap.add_argument("--compute-sweep", default="0")
    ap.add_argument("--placement", default="linear")
    ap.add_argument("--source-model", default="measured_packed_4")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-circuits", type=int, default=16)
    ap.add_argument("--ports-per-rank", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--port-rate", type=float, default=50.0)
    ap.add_argument("--reconfig-us", type=float, default=0.0)
    ap.add_argument("--epoch-collectives", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ocs-dedicated-port", action="store_true",
                    help="counterfactual: a circuit gets its own port, not the NIC's")
    ap.add_argument("--out", type=Path, default=_repo_root / "outputs" / "net_status")
    args = ap.parse_args(argv)

    ns = make_args(dict(workload=args.workload, world_size=args.world_size,
                        source_model=args.source_model,
                        placement=args.placement, k=args.k, layers=args.layers,
                        passes=args.passes, n_circuits=args.n_circuits,
                        ports_per_rank=args.ports_per_rank, sigma=args.sigma,
                        port_rate=args.port_rate, reconfig_us=args.reconfig_us,
                        epoch_collectives=args.epoch_collectives, seed=args.seed))
    if args.ocs_dedicated_port:
        install_port_exemption()
        print("   NOTE: OCS port-exemption counterfactual is ON "
              "(circuits no longer share the electrical NIC port)")
    intrace.OUT = args.out / "raw"
    topo, mats, layers, n_cells = load_layers_phase(ns, args.phase)
    print("== GPU status during MoE inference execution")
    print("   workload {}  W={}  placement {}  phase {}".format(
        args.workload, args.world_size, args.placement, args.phase))
    print("   MoE layers {}  passes {}  cells {:,}  bytes/layer {:,}".format(
        layers, args.passes, n_cells, int(mats[0].sum())))
    print("   fabric: port {} GB/s, sigma {}, {} circuits, reconfig {} us\n".format(
        args.port_rate, args.sigma, args.n_circuits, args.reconfig_us))

    blocks = []
    for cu in [float(x) for x in args.compute_sweep.split(",") if x.strip() != ""]:
        for pol in [p.strip() for p in args.policies.split(",") if p.strip()]:
            a = make_args(dict(vars(ns), compute_us=cu))
            b = analyse(pol, a, topo, mats, args.out)
            blocks.append(b)
            print(report(b))
            print()
            sys.stdout.flush()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "gpu_state_summary.json").write_text(json.dumps({
        "workload": str(args.workload), "world_size": args.world_size,
        "placement": args.placement, "layers": layers, "passes": args.passes,
        "fabric": {"port_gbps": args.port_rate, "sigma": args.sigma,
                   "n_circuits": args.n_circuits, "reconfig_us": args.reconfig_us},
        "blocks": [{k: v for k, v in b.items() if k not in ("res", "strip")}
                   for b in blocks],
    }, indent=1))
    print("-> " + str(args.out / "gpu_state_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
