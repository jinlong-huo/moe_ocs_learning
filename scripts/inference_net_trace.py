#!/usr/bin/env python3
"""inference_net_trace.py -- run the inference execution, watch the network.

WHAT THIS IS
────────────
Not "give the simulator a workload and measure its average load".  Here the network
runs *inside* the inference loop:

    layer -> compute -> collective (dispatch) -> network -> collective (combine)
          -> next layer

and at every instant of that execution the instrument records who is doing what:

  GPU        COMPUTE / SEND / RECV / SEND+RECV / IDLE, per rank, on change
  EPS        active flows, pooled-core utilisation, core queue, per-port queue
  OCS        configuration G(t), circuit ACTIVE / IDLE / DARK, bytes carried
  collective participants, start, finish, network time, EPS/OCS byte split

The point of contrast is the timescale of adaptation: EPS re-shares every instant
(statistical multiplexing, modelled as max-min fair over a pooled core), while the OCS
holds a configuration G until it is reconfigured, and pays a DARK interval when it
changes.  So the interesting output is not a single latency but the interaction:
which links saturate *while this collective runs*, which circuits are actually matched
to the demand, how much falls back to EPS, and what the reconfiguration costs.

NEW FILES ONLY
──────────────
    src/net/{__init__,fabric,collective,ocs_controller,engine}.py
    scripts/inference_net_trace.py
    docs/inference_driven_net.md
Nothing under src/eval, src/ocs, src/comm is modified.

Usage
    python3.12 scripts/inference_net_trace.py --workload logs/workload/qwen36 \
        --world-size 32 --layers 0,1,2,3 --passes 2 \
        --policies none,static_hot,epoch_hot,oracle_hot --n-circuits 16
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "scripts"))

import src.eval.cost_model as cm  # noqa: E402
from src.eval.cost_model import (  # noqa: E402
    CostConfig, DispatchMode, FabricConfig, Tier, Topology, hierarchy_for, traffic_matrix,
)
from src.eval.placement_opt import make_placement  # noqa: E402
from src.eval.trace_ir import load_workload  # noqa: E402
from src.net import (  # noqa: E402
    ControllerConfig, EngineConfig, EpsFabric, InferenceNetEngine, OcsConfig, OcsFabric,
    OcsController, make_controller,
)
from ownership_ocs_envelope import patch_token_rank  # noqa: E402

OUT = _repo_root / "outputs" / "net"


def load_layers(args):
    """The real trace, one byte matrix per MoE layer -- the inference events."""
    W = args.world_size
    cm.token_rank_original = cm.token_rank
    patch_token_rank(args.source_model, args.k)
    t_all = load_workload(args.workload / "manifest.json", decode_only=True)
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
    return topo, mats, [int(x) for x in layers]


def run_one(policy, args, topo, mats) -> dict:
    W = args.world_size
    eps = EpsFabric(port_gbytes_per_s=args.port_rate, sigma=args.sigma, world_size=W)
    ocs = OcsFabric(cfg=OcsConfig(n_circuits=args.n_circuits,
                                  ports_per_rank=args.ports_per_rank,
                                  rate_gbytes_per_s=args.port_rate,
                                  reconfig_us=args.reconfig_us,
                                  epoch_collectives=args.epoch_collectives))
    T = topo.tier_matrix()
    ctrl = make_controller(ControllerConfig(
        policy=policy, n_circuits=args.n_circuits, ports_per_rank=args.ports_per_rank,
        epoch_collectives=args.epoch_collectives,
        calibration_collectives=args.calibration,
        promotable=OcsController(ControllerConfig(promote_from=(Tier.CROSS_POD,))
                                 ).promotable_pairs(T)))
    eng = InferenceNetEngine(W, T, eps, ocs, ctrl,
                             EngineConfig(compute_us_per_layer=args.compute_us,
                                          include_combine=not args.no_combine,
                                          sample_ns=args.sample_ns,
                                          setup_free=not args.charge_setup))
    res = eng.run(mats, passes=args.passes)
    colls = res.collectives
    net = np.array([c["network_ns"] for c in colls]) / 1e3
    eps_b = sum(c["eps_bytes"] for c in colls)
    ocs_b = sum(c["ocs_bytes"] for c in colls)
    # dark WALL time is the union of the dark intervals, not the sum: overlapping
    # switchovers do not each cost their full duration
    iv = sorted((s, min(e, res.makespan_ns)) for s, e in ocs.dark_intervals)
    merged = []
    for s, e in iv:
        if e <= s:
            continue
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    dark_ns = sum(e - s for s, e in merged)
    tag = f"{policy}_eps{args.n_circuits}"
    write_csvs(tag, res)
    summary = {
        "policy": policy, "n_collectives": len(colls),
        "simulated_ns": round(res.makespan_ns, 1),
        "simulated_us": round(res.makespan_ns / 1e3, 3),
        "network_total_us": round(float(net.sum()), 3),
        "network_mean_us": round(float(net.mean()), 3) if len(net) else 0.0,
        "network_p50_us": round(float(np.percentile(net, 50)), 3) if len(net) else 0.0,
        "network_max_us": round(float(net.max()), 3) if len(net) else 0.0,
        "eps_bytes": eps_b, "ocs_bytes": ocs_b,
        "ocs_share": round(ocs_b / (eps_b + ocs_b), 4) if (eps_b + ocs_b) else 0.0,
        "n_reconfigurations": len(res.reconfigurations),
        "reconfig_sum_us": round(sum(c["dark_ns"] for c in res.reconfigurations) / 1e3, 3),
        "dark_wall_us": round(dark_ns / 1e3, 3),
        "dark_fraction_of_run": round(dark_ns / res.makespan_ns, 4) if res.makespan_ns else 0.0,
        "n_epochs": len(ocs.epochs),
        "circuits_carried_bytes": round(sum(c.get("bytes_carried", 0.0) for c in res.circuits), 1),
        "circuit_busy_us_total": round(sum(c.get("busy_ns", 0.0) for c in res.circuits) / 1e3, 3),
    }
    return {"summary": summary, "res": res}


def write_csvs(tag, res) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{tag}.gpu.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rank", "t_ns", "state", "collective", "n_send", "n_recv"])
        w.writeheader(); w.writerows(res.gpu_states)
    with open(OUT / f"{tag}.fabric.csv", "w", newline="") as fh:
        if res.fabric_samples:
            w = csv.DictWriter(fh, fieldnames=list(res.fabric_samples[0].keys()))
            w.writeheader(); w.writerows(res.fabric_samples)
    with open(OUT / f"{tag}.collectives.csv", "w", newline="") as fh:
        if res.collectives:
            w = csv.DictWriter(fh, fieldnames=list(res.collectives[0].keys()))
            w.writeheader(); w.writerows(res.collectives)
    with open(OUT / f"{tag}.circuits.csv", "w", newline="") as fh:
        if res.circuits:
            keys = sorted({k for c in res.circuits for k in c})
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(res.circuits)
    with open(OUT / f"{tag}.reconfig.csv", "w", newline="") as fh:
        if res.reconfigurations:
            w = csv.DictWriter(fh, fieldnames=["collective", "t_ns", "changed", "dark_ns",
                                               "torn_down", "established"],
                               extrasaction="ignore")
            w.writeheader(); w.writerows(res.reconfigurations)


def ascii_trace(tag, res, n_instants=10) -> str:
    """The picture, in text: who is doing what, and what the fabric is doing."""
    lines = []
    if not res.collectives:
        return ""
    t1 = res.makespan_ns
    lines.append(f"   {n_instants} instants across the whole run (0 - {t1/1e3:.1f} us); "
                 f"'coll' = the collective that is in flight")
    for k in range(n_instants):
        t = t1 * k / max(1, n_instants - 1)
        states = {}
        for row in res.gpu_states:
            if row["t_ns"] <= t:
                states[row["rank"]] = row["state"]
            else:
                break
        busy = {s: sorted(r for r, v in states.items() if v == s)
                for s in ("SEND", "RECV", "SEND+RECV", "COMPUTE")}
        s = res.fabric_samples
        idx = max(0, min(len(s) - 1, int(np.searchsorted([x["t_ns"] for x in s], t) - 1))) if s else None
        fab = s[idx] if idx is not None else {}
        coll = next((c for c in res.collectives
                     if c["start_ns"] <= t <= c["finish_ns"]), None)
        label = f"coll {coll['collective']:>2} {coll['phase']:<8}" if coll else "coll  -  (gap)     "
        lines.append(f"   t={t/1e3:9.1f} us  {label} | "
                     f"SEND {len(busy['SEND']):>2} RECV {len(busy['RECV']):>2} "
                     f"SR {len(busy['SEND+RECV']):>2} COMPUTE {len(busy['COMPUTE']):>2} "
                     f"IDLE {sum(1 for v in states.values() if v == 'IDLE'):>2} | "
                     f"core {fab.get('core_utilisation', 0)*100:>5.1f}% "
                     f"q={fab.get('core_queue_mb', 0):>6.1f}MB "
                     f"OCS {fab.get('ocs_bw_gbps', 0):>5.1f}GB/s"
                     + ("  [DARK]" if fab.get("n_dark") else ""))
    return "\n".join(lines)


def figure(tags, summaries, args) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    colors = {"SEND": "#c44e52", "RECV": "#4c72b0", "SEND+RECV": "#8172b3",
              "COMPUTE": "#55a868", "IDLE": "#dddddd"}
    for tag in tags:
        rows = list(csv.DictReader(open(OUT / f"{tag}.gpu.csv")))
        if not rows:
            continue
        span = max(float(r["t_ns"]) for r in rows)
        prev = {}
        for r in rows:
            rank, t, st = int(r["rank"]), float(r["t_ns"]) / 1e3, r["state"]
            if rank in prev:
                axes[0].barh(rank, t - prev[rank][0], left=prev[rank][0], height=0.7,
                             color=colors.get(prev[rank][1], "grey"), alpha=0.85)
            prev[rank] = (t, st)
        axes[0].set_title(f"GPU state during inference execution (last policy: {tag})")
        axes[0].set_ylabel("rank")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    axes[0].legend(handles, colors.keys(), ncol=5, fontsize=7, loc="lower right")
    for tag in tags:
        rows = list(csv.DictReader(open(OUT / f"{tag}.fabric.csv")))
        if not rows:
            continue
        t = [float(r["t_ns"]) / 1e3 for r in rows]
        axes[1].plot(t, [float(r["core_utilisation"]) for r in rows], lw=0.9, label=f"{tag} EPS core util")
        axes[2].plot(t, [float(r["ocs_bw_gbps"]) for r in rows], lw=0.9, label=f"{tag} OCS GB/s")
    axes[1].set_ylabel("EPS core utilisation"); axes[1].set_ylim(0, 1.05)
    axes[2].set_ylabel("OCS carried (GB/s)"); axes[2].set_xlabel("time (us)")
    for ax in axes:
        ax.grid(alpha=.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    p = OUT / "inference_net_trace.png"
    fig.savefig(p, dpi=130)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--source-model", default="measured_packed_4")
    ap.add_argument("--placement", default="linear")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--policies", default="none,static_hot,oracle_hot,epoch_hot")
    ap.add_argument("--n-circuits", type=int, default=16)
    ap.add_argument("--ports-per-rank", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--port-rate", type=float, default=50.0)
    ap.add_argument("--reconfig-us", type=float, default=10_000.0)
    ap.add_argument("--epoch-collectives", type=int, default=8)
    ap.add_argument("--calibration", type=int, default=1)
    ap.add_argument("--compute-us", type=float, default=0.0)
    ap.add_argument("--sample-ns", type=float, default=2_000.0)
    ap.add_argument("--no-combine", action="store_true")
    ap.add_argument("--charge-setup", action="store_true",
                    help="charge the first OCS establishment inside the run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    topo, mats, layers = load_layers(args)
    print(f"== inference-driven network trace: {args.workload.name} W={args.world_size} "
          f"{args.source_model} {args.placement}")
    print(f"   {len(layers)} MoE layers as inference events: {layers}, passes={args.passes}, "
          f"sigma={args.sigma}, port={args.port_rate} GB/s")
    print(f"   bytes per layer {int(mats[0].sum()):,}\n")

    tags, summaries, results = [], {}, {}
    for pol in [p.strip() for p in args.policies.split(",") if p.strip()]:
        out = run_one(pol, args, topo, mats)
        s = out["summary"]
        summaries[pol] = s
        results[pol] = out["res"]
        tags.append(f"{pol}_eps{args.n_circuits}")
        print(f"   {pol:<12} collectives {s['n_collectives']:>4}  "
              f"network total {s['network_total_us']:>10.1f} us  mean {s['network_mean_us']:>8.1f}  "
              f"max {s['network_max_us']:>8.1f}  OCS share {s['ocs_share']*100:>5.1f}%  "
              f"reconfigs {s['n_reconfigurations']}  dark {s['dark_wall_us']/1e3:.2f} ms "
              f"({s['dark_fraction_of_run']*100:.1f}% of the run)")
        sys.stdout.flush()

    base = summaries.get("none", {}).get("network_total_us")
    if base:
        print("\n   vs pure EPS (network time, lower is better):")
        for pol, s in summaries.items():
            d = 100.0 * (1 - s["network_total_us"] / base)
            print(f"     {pol:<12} {d:+7.2f}%")

    last = [p.strip() for p in args.policies.split(",") if p.strip()][-1]
    print()
    print(ascii_trace(f"{last}_eps{args.n_circuits}", results[last]))
    p = figure(tags, summaries, args)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inference_net_summary.json").write_text(json.dumps({
        "workload": str(args.workload), "world_size": args.world_size,
        "source_model": args.source_model, "placement": args.placement,
        "layers": layers, "passes": args.passes,
        "fabric": {"sigma": args.sigma, "port_gbps": args.port_rate,
                   "n_circuits": args.n_circuits, "ports_per_rank": args.ports_per_rank,
                   "reconfig_us": args.reconfig_us, "epoch_collectives": args.epoch_collectives},
        "policies": summaries}, indent=1))
    print(f"\n-> {OUT}/inference_net_summary.json")
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
