#!/usr/bin/env python3
"""spine_probe.py -- real-trace MoE traffic through ASTRA-sim, plus a per-node
communication makespan and a per-link (EPS / OCS) state timeline.

Two stages, one traffic matrix:

  astra     writes the per-pair byte matrix as a Chakra custom collective exactly
            as scripts/astra_placement_order.py does, runs ASTRA-sim (both
            backends), and parses the PER-RANK "Wall time" -- so the straggler
            rank, not just the max, is available.

  timeline  replays the same message list (the same per-rank FIFO order the ET
            files encode) against the tier rates, giving every message a start and
            an end.  From that: per-rank communication makespan, per-pair link
            busy/idle intervals, per-tier (EPS core / OCS circuits) occupancy, and
            a Gantt figure.

Promotion is the repo's own semantics: a circuit promotes one unordered rank pair
from CROSS_POD (12.5 GB/s, 12 us) to OPTICAL (50 GB/s, 6 us).  In the ASTRA stage a
promoted pair is REMOVED from the simulated fabric and charged analytically -- the
hybrid documented in docs/hot_spine_ocs.md section 9; the removal is exactly what a
real circuit does to the shared core, and the charge is the repo's own arithmetic.
In the timeline stage promotion is native (the tier matrix carries it).

Usage
    python3.12 scripts/spine_probe.py --stage astra    --configs eps,hot16,cold16
    python3.12 scripts/spine_probe.py --stage timeline --configs eps,hot16,cold16
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "scripts"))

DEFAULT_ASTRA = Path.home() / "astra-sim"
NETCFG = "configs/astra_sim/eps_baseline_2tier.yml"     # Switch 8 @450ns / Ring 4 @12.5, lat 2000/12000
OUT = _repo_root / "outputs" / "spine"


# ── traffic ────────────────────────────────────────────────────────────────────
def build(args):
    """Real trace -> placement -> per-pair byte matrix, plus the circuit configs."""
    import src.eval.cost_model as cm
    from src.eval.cost_model import (CostConfig, DispatchMode, FabricConfig, Tier,
                                     Topology, hierarchy_for, traffic_matrix)
    from src.eval.placement_opt import make_placement
    from src.eval.trace_ir import load_workload
    from ownership_ocs_envelope import patch_token_rank

    W = args.world_size
    cm.token_rank_original = cm.token_rank
    patch_token_rank(args.source_model, args.k)

    t_all = load_workload(args.workload / "manifest.json", decode_only=True)
    layer = t_all.by_layer(int(np.sort(np.unique(t_all.layers))[0]))
    cost = CostConfig(hidden_size=2048)
    base = hierarchy_for(W, "multi_pod")
    fab = FabricConfig(core_oversubscription=4.0, pod_oversubscription=1.0)

    def topo(circuits=()):
        return Topology(W, base.gpus_per_node, base.nodes_per_pod, fab, set(circuits),
                        base.rank_to_slot, (Tier.CROSS_POD,))

    pl = make_placement(args.placement, layer, W, seed=args.seed)
    tm = traffic_matrix(layer, pl, topo(), DispatchMode.DEDUP_RANK, n_dp=W, seed=args.seed)
    mat = (tm.counts * (cost.hidden_size * cost.dtype_bytes)).astype(np.int64)
    np.fill_diagonal(mat, 0)

    T = topo().tier_matrix()
    promotable = {int(x) for x in topo().promote_from
                  if fab.oversubscription(Tier(int(x))) > 1.0}
    pairs = sorted(((int(mat[a, b] + mat[b, a]), frozenset((a, b)))
                    for a in range(W) for b in range(a + 1, W)
                    if int(T[a, b]) in promotable and (mat[a, b] + mat[b, a]) > 0),
                   reverse=True, key=lambda x: x[0])

    n = args.n_circuits
    sets = {
        "eps": set(),
        "hot16": {p for _, p in pairs[:n]},
        "cold16": {p for _, p in pairs[-n:]} if len(pairs) >= n else set(),
        "all": {p for _, p in pairs},
    }
    return {
        "layer": layer, "mat": mat, "topo": topo, "pairs": pairs, "sets": sets,
        "fabric": fab, "W": W, "tier": T, "promotable": promotable,
        "restore": cm.token_rank_original,
    }


# ── stage: ASTRA-sim ───────────────────────────────────────────────────────────
def stage_astra(ctx, args):
    from src.eval.cost_model import Tier
    sys.path.insert(0, str(args.astra))
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
        ALL_TO_ALL, COMM_COLL_NODE, COMM_RECV_NODE, COMM_SEND_NODE, GlobalMetadata, Node)
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import AttributeProto as Attr
    from extern.graph_frontend.chakra.src.third_party.utils.protolib import encodeMessage

    W, mat = ctx["W"], ctx["mat"]
    out_root = _repo_root / "outputs/astra_sim/et/spine"
    results = {}

    # promotion encoding: "scale" keeps every rank non-empty by shrinking a
    # promoted pair's simulated bytes by the promotion factor (50/12.5 = 4x
    # cheaper on the shared core); "remove" takes them off the fabric entirely
    # (which empties the ET of any rank whose only traffic was promoted, and the
    # ASTRA reader rejects a node-less ET).
    factor = float(ctx["fabric"].bandwidth(Tier.CROSS_POD)) / float(
        ctx["fabric"].bandwidth(Tier.OPTICAL))

    for cfg in args.configs:
        circ = ctx["sets"][cfg]
        dropped = set()
        for p in circ:
            a, b = tuple(p)
            dropped.add((a, b)); dropped.add((b, a))
        m = mat.copy()
        for (a, b) in dropped:
            # the floor matters: a message small enough to take zero cycles makes
            # ASTRA's event queue assert (EventQueue.cpp:33 needs strictly
            # increasing event times), so a promoted pair never drops below 64 KiB.
            m[a, b] = 0 if args.encode == "remove" else max(65536, int(round(m[a, b] * factor)))

        out = out_root / cfg
        out.mkdir(parents=True, exist_ok=True)
        prefix = f"{cfg}_{args.source_model}"
        node_id = 0
        bodies = {}
        for r in range(W):
            sends = sorted(((int(m[r, d]), int(d)) for d in range(W) if m[r, d] > 0), reverse=True)
            recvs = sorted(((int(m[s, r]), int(s)) for s in range(W) if m[s, r] > 0), reverse=True)
            body = []
            for i in range(max(len(sends), len(recvs))):
                if i < len(sends):
                    body.append(("send", sends[i]))
                if i < len(recvs):
                    body.append(("recv", recvs[i]))
            bodies[r] = body
            with open(out / f"{prefix}.{r}.et", "wb") as f:
                encodeMessage(f, GlobalMetadata(version="0.0.4"))
                for what, (size, peer) in body:
                    n = Node()
                    n.id = node_id; n.name = f"{what}_{r}_{peer}"
                    n.type = COMM_SEND_NODE if what == "send" else COMM_RECV_NODE
                    n.attr.append(Attr(name="is_cpu_op", bool_val=False))
                    n.attr.append(Attr(name="comm_size", int64_val=size))
                    n.attr.append(Attr(name="comm_src", int64_val=r if what == "send" else peer))
                    n.attr.append(Attr(name="comm_dst", int64_val=peer if what == "send" else r))
                    n.attr.append(Attr(name="comm_tag", int64_val=0))
                    encodeMessage(f, n); node_id += 1
            with open(out / f"{prefix}.wl.{r}.et", "wb") as f:
                encodeMessage(f, GlobalMetadata(version="0.0.4"))
                n = Node(); n.id = 0; n.name = f"all_to_all_W{W}"; n.type = COMM_COLL_NODE
                n.attr.append(Attr(name="is_cpu_op", bool_val=False))
                n.attr.append(Attr(name="comm_type", int64_val=ALL_TO_ALL))
                n.attr.append(Attr(name="comm_size", int64_val=1 << 20))
                encodeMessage(f, n)
        (out / f"{prefix}.system.json").write_text(json.dumps({
            "scheduling-policy": "FIFO", "preferred-dataset-splits": 1,
            "all-to-all-implementation-custom": [f"outputs/astra_sim/et/spine/{cfg}/{prefix}"],
            "local-mem-bw": 3350}, indent=2))

        for backend in args.backends:
            binary = args.astra / f"build/astra_analytical/build/bin/AstraSim_Analytical_{backend}"
            cmd = [str(binary),
                   f"--workload-configuration=outputs/astra_sim/et/spine/{cfg}/{prefix}.wl",
                   f"--system-configuration=outputs/astra_sim/et/spine/{cfg}/{prefix}.system.json",
                   f"--remote-memory-configuration={args.astra}/examples/remote_memory/analytical/no_memory_expansion.json",
                   f"--network-configuration={args.netcfg}"]
            p = subprocess.run(cmd, cwd=_repo_root, capture_output=True, text=True)
            times = [int(x) for x in re.findall(r"Wall time: (\d+)", p.stdout)]
            if not times:
                results.setdefault(cfg, {})[backend] = {"ok": False,
                                                        "stderr_tail": p.stderr[-400:]}
                print(f"   {cfg:<8}{backend:<22}FAILED"); sys.stdout.flush(); continue
            arr = np.array(times)
            rows = sorted(((int(t), r) for r, t in enumerate(times)), reverse=True)
            results.setdefault(cfg, {})[backend] = {
                "ok": True, "n_ranks_reported": len(times),
                "makespan_cycles": int(arr.max()), "argmax_rank": int(arr.argmax()),
                "min_cycles": int(arr.min()), "mean_cycles": float(arr.mean()),
                "per_rank": [int(x) for x in times],
                "top5_stragglers": [{"rank": r, "cycles": t} for t, r in rows[:5]],
                "bytes_simulated": int(m.sum()),
                "bytes_removed": int(mat.sum() - m.sum()),
                "n_circuits": len(circ),
            }
            print(f"   {cfg:<8}{backend:<22}makespan {arr.max():>9,} cyc  "
                  f"straggler rank {int(arr.argmax()):>2}  bytes {m.sum():>12,}")
            sys.stdout.flush()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = Path(args.netcfg).stem
    (OUT / f"astra_spine.{tag}.json").write_text(json.dumps({
        "workload": str(args.workload), "world_size": W, "source_model": args.source_model,
        "placement": args.placement, "unit": "one MoE layer, custom collective",
        "network": NETCFG, "note": "promoted pairs removed from the simulated fabric",
        "results": results}, indent=1))
    print(f"-> {OUT/f'astra_spine.{tag}.json'}")
    return results


# ── stage: timeline ────────────────────────────────────────────────────────────
def _bodies(mat, W):
    bodies = {}
    for r in range(W):
        sends = sorted(((int(mat[r, d]), int(d)) for d in range(W) if mat[r, d] > 0), reverse=True)
        recvs = sorted(((int(mat[s, r]), int(s)) for s in range(W) if mat[s, r] > 0), reverse=True)
        body = []
        for i in range(max(len(sends), len(recvs))):
            if i < len(sends):
                body.append(("send", sends[i][1], sends[i][0]))
            if i < len(recvs):
                body.append(("recv", recvs[i][1], recvs[i][0]))
        bodies[r] = body
    return bodies


def stage_timeline(ctx, args):
    from src.eval.cost_model import Tier  # noqa: F401
    W, mat, fab = ctx["W"], ctx["mat"], ctx["fabric"]
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}

    for cfg in args.configs:
        circ = ctx["sets"][cfg]
        topo = ctx["topo"](circ)
        T = topo.tier_matrix()
        bodies = _bodies(mat, W)

        egress = np.zeros(W); ingress = np.zeros(W)          # NIC ports
        egress_nv = np.zeros(W); ingress_nv = np.zeros(W)    # NVLink ports
        done = set()
        msgs = []
        for r in range(W):
            for what, peer, size in bodies[r]:
                if what == "recv":
                    continue
                aid, bid = min(r, peer), max(r, peer)
                if (aid, bid) in done:
                    continue
                done.add((aid, bid))
                tier = Tier(int(T[r, peer]))
                bw = fab.bandwidth(tier)                      # GB/s
                lat_ns = fab.latency_us(tier) * 1e3
                if tier == Tier.INTRA_NODE:
                    start = max(egress_nv[r], ingress_nv[peer])
                    end = start + size / bw + lat_ns
                    egress_nv[r] = end; ingress_nv[peer] = end
                else:
                    start = max(egress[r], ingress[peer])
                    end = start + size / bw + lat_ns
                    egress[r] = end; ingress[peer] = end
                msgs.append({"src": int(r), "dst": int(peer), "bytes": int(size),
                             "tier": tier.name, "bw_gbps": float(bw),
                             "latency_us": float(fab.latency_us(tier)),
                             "start_ns": float(start), "end_ns": float(end),
                             "dur_ns": float(end - start)})

        span = max(m["end_ns"] for m in msgs) if msgs else 0.0
        rank_end = {}
        for m in msgs:
            rank_end[m["src"]] = max(rank_end.get(m["src"], 0.0), m["end_ns"])
            rank_end[m["dst"]] = max(rank_end.get(m["dst"], 0.0), m["end_ns"])
        for r in range(W):
            rank_end.setdefault(r, 0.0)
        straggler = max(rank_end.items(), key=lambda kv: kv[1])

        # per-pair links
        links = {}
        for m in msgs:
            k = (m["src"], m["dst"])
            links.setdefault(k, {"src": k[0], "dst": k[1], "tier": m["tier"],
                                 "bytes": 0, "n": 0, "intervals": []})
            links[k]["bytes"] += m["bytes"]
            links[k]["n"] += 1
            links[k]["intervals"].append([m["start_ns"], m["end_ns"]])
        link_rows = []
        for k, v in sorted(links.items()):
            iv = sorted(v["intervals"])
            merged, busy = [], 0.0
            for s, e in iv:
                if merged and s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            busy = sum(e - s for s, e in merged)
            idle = max(0.0, span - busy)
            gaps = [[merged[i][1], merged[i + 1][0]] for i in range(len(merged) - 1)
                    if merged[i + 1][0] > merged[i][1] + 1e-9]
            link_rows.append({**{kk: v[kk] for kk in ("src", "dst", "tier", "bytes", "n")},
                              "busy_ns": round(busy, 1),
                              "utilisation": round(busy / span, 4) if span else 0.0,
                              "idle_ns": round(idle, 1),
                              "n_busy_intervals": len(merged),
                              "n_idle_gaps": len(gaps),
                              "largest_idle_gap_ns": round(max((e - s for s, e in gaps), default=0.0), 1),
                              "intervals": [[round(s, 1), round(e, 1)] for s, e in merged]})

        # per-tier occupancy (the shared EPS core vs the OCS circuits)
        tiers = {}
        for m in msgs:
            tiers.setdefault(m["tier"], []).append((m["start_ns"], m["end_ns"]))
        tier_rows = []
        for tier, iv in sorted(tiers.items()):
            u = sorted(iv)
            merged = []
            for s, e in u:
                if merged and s <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            busy = sum(e - s for s, e in merged)
            idle_gaps = [[merged[i][1], merged[i + 1][0]] for i in range(len(merged) - 1)
                         if merged[i + 1][0] > merged[i][1] + 1e-9]
            tier_rows.append({"tier": tier, "n_links": len(tiers[tier]),
                              "busy_ns": round(busy, 1),
                              "occupancy_of_span": round(busy / span, 4) if span else 0.0,
                              "idle_ns": round(max(0.0, span - busy), 1),
                              "n_idle_gaps": len(idle_gaps),
                              "largest_idle_gap_ns": round(max((e - s for s, e in idle_gaps), default=0.0), 1),
                              "intervals": [[round(s, 1), round(e, 1)] for s, e in merged]})

        prom = ctx["promotable"]
        (OUT / f"{cfg}.messages.csv").write_text(
            "collective,src,dst,bytes,means,tier,bw_gbps,start_ns,end_ns,dur_ns\n" +
            "\n".join(
                f"all_to_all_dispatch,{m['src']},{m['dst']},{m['bytes']},"
                f"{'OCS' if m['tier']=='OPTICAL' else m['tier']},{m['tier']},{m['bw_gbps']},"
                f"{m['start_ns']:.1f},{m['end_ns']:.1f},{m['dur_ns']:.1f}" for m in msgs) + "\n")
        (OUT / f"{cfg}.links.json").write_text(json.dumps(link_rows, indent=1))
        (OUT / f"{cfg}.tiers.json").write_text(json.dumps(tier_rows, indent=1))
        (OUT / f"{cfg}.ranks.csv").write_text(
            "rank,makespan_ns,egress_nic_ns,ingress_nic_ns,egress_nvlink_ns,ingress_nvlink_ns,n_send,bytes_send\n" +
            "\n".join(
                f"{r},{rank_end[r]:.1f},{egress[r]:.1f},{ingress[r]:.1f},"
                f"{egress_nv[r]:.1f},{ingress_nv[r]:.1f},"
                f"{sum(1 for m in msgs if m['src']==r)},"
                f"{sum(m['bytes'] for m in msgs if m['src']==r)}" for r in range(W)) + "\n")

        summary[cfg] = {
            "n_messages": len(msgs), "n_links": len(link_rows),
            "makespan_ns": round(span, 1), "makespan_us": round(span / 1e3, 3),
            "straggler_rank": int(straggler[0]), "straggler_ns": round(straggler[1], 1),
            "tiers": {t["tier"]: {"busy_ns": t["busy_ns"], "occupancy": t["occupancy_of_span"],
                                  "n_links": t["n_links"], "largest_idle_gap_ns": t["largest_idle_gap_ns"]}
                      for t in tier_rows},
            "promoted_pairs": len(circ),
            "promoted_bytes": int(sum(m["bytes"] for m in msgs
                                      if frozenset((m["src"], m["dst"])) in circ)),
            "straggler_is_promoted": bool(any(frozenset((m["src"], m["dst"])) in circ
                                              for m in msgs
                                              if m["end_ns"] >= straggler[1] - 1e-9)),
            "eps_core_idle_ns": next((t["idle_ns"] for t in tier_rows if t["tier"] == "CROSS_POD"), None),
        }
        # ── link state over time: ASCII strips + a binned occupancy matrix ──
        nb = 72
        edges = np.linspace(0.0, span if span else 1.0, nb + 1)
        strips = {}
        for t in tier_rows:
            occ = np.zeros(nb)
            for s, e in t["intervals"]:
                i0 = int(np.searchsorted(edges, s, "right") - 1)
                i1 = int(np.searchsorted(edges, e, "left"))
                for b in range(max(0, i0), min(nb, max(i1, i0 + 1))):
                    occ[b] += 1.0
            strips[t["tier"]] = "".join("#" if x > 0 else "." for x in occ)
            t["ascii"] = strips[t["tier"]]
        top_links = sorted(link_rows, key=lambda r: -r["bytes"])[:6]
        print(f"   {cfg:<8} makespan {span/1e3:>10.3f} us  straggler rank {straggler[0]:>2}  "
              f"msgs {len(msgs):>5}  links {len(link_rows):>4}  promoted {len(circ)}")
        print(f"            span {span/1e3:.1f} us, {nb} bins; '#' = carrying traffic, '.' = idle")
        for t in sorted(tier_rows, key=lambda x: -x["busy_ns"]):
            print(f"            {t['tier']:<11} {t['ascii']}  occ {t['occupancy_of_span']:.2f} "
                  f"idle_gaps {t['n_idle_gaps']} max_gap {t['largest_idle_gap_ns']/1e3:.1f} us")
        for r in top_links:
            occ = np.zeros(nb)
            for s, e in r["intervals"]:
                i0 = int(np.searchsorted(edges, s, "right") - 1)
                i1 = int(np.searchsorted(edges, e, "left"))
                for b in range(max(0, i0), min(nb, max(i1, i0 + 1))):
                    occ[b] += 1.0
            r["ascii"] = "".join("#" if x > 0 else "." for x in occ)
            r["means"] = "OCS" if r["tier"] == "OPTICAL" else r["tier"]
            print(f"            link {r['src']:>2}->{r['dst']:<2} {r['means']:<10}{r['ascii']} "
                  f"occ {r['utilisation']:.2f} {r['bytes']/1e6:.1f} MB")
        sys.stdout.flush()

    (OUT / "timeline_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {OUT/'timeline_summary.json'}")
    return summary


def make_figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"INTRA_NODE": "#4c72b0", "INTRA_POD": "#dd8452",
              "CROSS_POD": "#c44e52", "OPTICAL": "#55a868"}
    cfgs = [c for c in args.configs]
    fig, axes = plt.subplots(len(cfgs) + 1, 1, figsize=(13, 3.1 * (len(cfgs) + 1)))
    if len(cfgs) + 1 == 1:
        axes = [axes]
    import csv as _csv
    for ax, cfg in zip(axes, cfgs):
        with open(OUT / f"{cfg}.messages.csv") as fh:
            for r in _csv.DictReader(fh):
                ax.barh(int(r["src"]), float(r["dur_ns"]) / 1e3, left=float(r["start_ns"]) / 1e3,
                        height=0.7, color=colors.get(r["tier"], "grey"))
        ax.set_title(f"{cfg}: per-rank communication makespan (bar = one message, colour = means)")
        ax.set_ylabel("rank"); ax.grid(alpha=.25, axis="x")
    ax = axes[-1]
    for cfg in cfgs:
        t = json.loads((OUT / f"{cfg}.tiers.json").read_text())
        core = next((x for x in t if x["tier"] == "CROSS_POD"), None)
        if core:
            for s, e in core["intervals"]:
                ax.barh(cfg, (e - s) / 1e3, left=s / 1e3, height=0.5, color="#c44e52",
                        label="EPS core busy" if cfg == cfgs[0] else None)
        oc = [x for x in t if x["tier"] == "OPTICAL"]
        for x in oc:
            for s, e in x["intervals"]:
                ax.barh(cfg, (e - s) / 1e3, left=s / 1e3, height=0.5, color="#55a868",
                        label="OCS circuit busy" if cfg == cfgs[0] else None)
    ax.set_title("link state: when the EPS core and the OCS circuits are carrying traffic (gap = idle)")
    ax.set_xlabel("time (us)"); ax.grid(alpha=.25, axis="x")
    handles = [Patch(color=v, label=k) for k, v in colors.items()]
    axes[0].legend(handles=handles, loc="lower right", fontsize=7, ncol=4)
    fig.tight_layout()
    p = OUT / "comm_makespan_gantt.png"
    fig.savefig(p, dpi=130)
    print(f"-> {p}")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--source-model", default="measured_packed_4")
    ap.add_argument("--placement", default="linear")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-circuits", type=int, default=16)
    ap.add_argument("--configs", default="eps,hot16,cold16")
    ap.add_argument("--stage", default="all", choices=("astra", "timeline", "figure", "all"))
    ap.add_argument("--encode", default="scale", choices=("scale", "remove"),
                    help="how a promoted pair is encoded for ASTRA-sim")
    ap.add_argument("--netcfg", default=NETCFG)
    ap.add_argument("--backends", default="Congestion_Aware,Congestion_Unaware")
    ap.add_argument("--astra", type=Path, default=DEFAULT_ASTRA)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    args.backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    ctx = build(args)
    print(f"== spine probe: {args.workload.name} W={args.world_size} "
          f"{args.source_model} {args.placement} layer0")
    print(f"   bytes/layer {int(ctx['mat'].sum()):,}  promotable pairs {len(ctx['pairs'])}  "
          f"top pair {ctx['pairs'][0][0]:,} B  bottom {ctx['pairs'][-1][0]:,} B\n")
    try:
        if args.stage in ("astra", "all"):
            stage_astra(ctx, args)
        if args.stage in ("timeline", "all"):
            stage_timeline(ctx, args)
        if args.stage in ("figure", "all"):
            make_figure(args)
    finally:
        import src.eval.cost_model as cm
        cm.token_rank = ctx["restore"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
