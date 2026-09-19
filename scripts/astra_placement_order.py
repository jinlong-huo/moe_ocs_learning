#!/usr/bin/env python3
"""astra_placement_order.py — Phase 4a/4 capstone: validate the PLACEMENT ordering
against ASTRA-sim.

WHY THIS IS THE ONE THAT MATTERS
────────────────────────────────
`docs/astra_ordering.md` validated the FABRIC axis (sigma, promotion, latency) on a
uniform all-to-all, which says nothing about placement.  Our substantive claims are
placement orderings on an uneven, placement-dependent dispatch matrix.  This script
runs exactly that: for each placement it writes the per-pair byte matrix as a
Chakra custom collective (scripts/chakra_from_traffic.py's machinery, inlined
here), executes it in ASTRA-sim, and correlates the resulting times against the
closed-form model's bottleneck.

WHAT IT COMPARES
────────────────
    model   evaluate() on the SAME single layer -> bottleneck_us
    astra   ASTRA-sim analytical backend, congestion-unaware -> max over ranks

    verdict Spearman rank correlation + the sign of every pairwise comparison.

The claim is ordering, never equality: 10-40 % absolute error is expected and is
not a failure.  A REVERSAL is the outcome that would invalidate the cost model.

Usage
─────
    python3.12 scripts/astra_placement_order.py --workload logs/workload/qwen36 --world-size 32
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
PLACEMENTS = ("linear", "load_balanced_layer", "affinity_layer",
              "affinity_coordinated_layer", "promote_aware_layer")


def spearman(a, b) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 1.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--source-model", default="measured_packed_4")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--placements", default=",".join(PLACEMENTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--astra", type=Path, default=DEFAULT_ASTRA)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    import src.eval.cost_model as cm  # noqa: E402
    from src.eval.cost_model import (  # noqa: E402
        CostConfig, DispatchMode, FabricConfig, Tier, Topology, evaluate,
        hierarchy_for, traffic_matrix,
    )
    from src.eval.ocs_eval import OcsConfig  # noqa: E402
    from src.eval.placement_opt import make_placement  # noqa: E402
    from src.eval.trace_ir import load_workload  # noqa: E402
    from ownership_ocs_envelope import patch_token_rank, restore_token_rank  # noqa: E402

    sys.path.insert(0, str(args.astra))
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (  # noqa: E402
        ALL_TO_ALL, COMM_COLL_NODE, COMM_RECV_NODE, COMM_SEND_NODE, GlobalMetadata,
        Node,
    )
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (  # noqa: E402
        AttributeProto as Attr,
    )
    from extern.graph_frontend.chakra.src.third_party.utils.protolib import (  # noqa: E402
        encodeMessage,
    )

    W = args.world_size
    binary = args.astra / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
    if not binary.is_file():
        raise SystemExit(f"error: no ASTRA-sim binary at {binary}")

    cm.token_rank_original = cm.token_rank
    t = load_workload(args.workload / "manifest.json", decode_only=True)
    layer0 = t.by_layer(int(np.sort(np.unique(t.layers))[0]))
    cost = CostConfig(hidden_size=2048)
    base = hierarchy_for(W, "multi_pod")
    fab = FabricConfig(core_oversubscription=4.0, pod_oversubscription=1.0)
    topo = Topology(W, base.gpus_per_node, base.nodes_per_pod, fab, set(),
                    base.rank_to_slot, (Tier.CROSS_POD,))

    print(f"== placement ordering vs ASTRA-sim: {args.workload.name} W={W} "
          f"source={args.source_model}")
    print(f"   one MoE layer, per-pair send/recv via the custom-collective path\n")

    rows: dict = {}
    try:
        patch_token_rank(args.source_model, args.k)
        for kind in [p.strip() for p in args.placements.split(",") if p.strip()]:
            # promote_aware_layer needs the fabric and circuit budget: its objective
            # is the bottleneck REMAINING after promotion, so it is topology-aware.
            pl = make_placement(kind, t, W, seed=args.seed, topo=topo,
                                ocs_cfg=OcsConfig(n_circuits=max(4, W // 2),
                                                  ports_per_rank=2))
            tm = traffic_matrix(layer0, pl, topo, DispatchMode.DEDUP_RANK,
                                n_dp=W, seed=args.seed)
            B = cost.hidden_size * cost.dtype_bytes
            mat = (tm.counts * B).astype(np.int64)
            np.fill_diagonal(mat, 0)
            model_us = evaluate(layer0, pl, topo, cost, DispatchMode.DEDUP_RANK,
                                n_dp=W, seed=args.seed)["bottleneck_us"]

            out = _repo_root / f"outputs/astra_sim/et/order_{kind}"
            out.mkdir(parents=True, exist_ok=True)
            prefix = f"{kind}.{args.source_model}"
            node_id = 0
            for r in range(W):
                sends = sorted(((int(mat[r, d]), int(d)) for d in range(W)
                                if mat[r, d] > 0), reverse=True)
                recvs = sorted(((int(mat[s, r]), int(s)) for s in range(W)
                                if mat[s, r] > 0), reverse=True)
                body = []
                for i in range(max(len(sends), len(recvs))):
                    if i < len(sends):
                        body.append(("send", sends[i]))
                    if i < len(recvs):
                        body.append(("recv", recvs[i]))
                with open(out / f"{prefix}.{r}.et", "wb") as f:
                    encodeMessage(f, GlobalMetadata(version="0.0.4"))
                    for what, (size, peer) in body:
                        n = Node()
                        n.id = node_id
                        n.name = f"{what}_{r}_{peer}"
                        n.type = COMM_SEND_NODE if what == "send" else COMM_RECV_NODE
                        n.attr.append(Attr(name="is_cpu_op", bool_val=False))
                        n.attr.append(Attr(name="comm_size", int64_val=size))
                        n.attr.append(Attr(name="comm_src",
                                           int64_val=r if what == "send" else peer))
                        n.attr.append(Attr(name="comm_dst",
                                           int64_val=peer if what == "send" else r))
                        n.attr.append(Attr(name="comm_tag", int64_val=0))
                        encodeMessage(f, n)
                        node_id += 1
                with open(out / f"{prefix}.wl.{r}.et", "wb") as f:
                    encodeMessage(f, GlobalMetadata(version="0.0.4"))
                    n = Node()
                    n.id = 0
                    n.name = f"all_to_all_W{W}"
                    n.type = COMM_COLL_NODE
                    n.attr.append(Attr(name="is_cpu_op", bool_val=False))
                    n.attr.append(Attr(name="comm_type", int64_val=ALL_TO_ALL))
                    n.attr.append(Attr(name="comm_size", int64_val=1 << 20))
                    encodeMessage(f, n)
            (out / f"{prefix}.system.json").write_text(json.dumps({
                "scheduling-policy": "FIFO", "preferred-dataset-splits": 1,
                "all-to-all-implementation-custom": [
                    f"outputs/astra_sim/et/order_{kind}/{prefix}"],
                "local-mem-bw": 3350}, indent=2))

            cmd = [str(binary),
                   f"--workload-configuration=outputs/astra_sim/et/order_{kind}/{prefix}.wl",
                   f"--system-configuration=outputs/astra_sim/et/order_{kind}/{prefix}.system.json",
                   f"--remote-memory-configuration={args.astra}/examples/remote_memory/"
                   "analytical/no_memory_expansion.json",
                   "--network-configuration=configs/astra_sim/eps_baseline_2tier.yml"]
            p = subprocess.run(cmd, cwd=_repo_root, capture_output=True, text=True)
            times = [int(x) for x in re.findall(r"Wall time: (\d+)", p.stdout)]
            astra = max(times) if times else None
            rows[kind] = {
                "bytes_on_the_wire": int(mat.sum()),
                "model_bottleneck_us": round(model_us, 2),
                "astra_max_cycles": astra,
                "astra_ok": astra is not None,
            }
            print(f"   {kind:<28} bytes {mat.sum():>12,}  model {model_us:>10.1f} us  "
                  f"astra {astra if astra else 'FAILED'}")
            sys.stdout.flush()
    finally:
        restore_token_rank()

    ok = {k: v for k, v in rows.items() if v["astra_ok"]}
    rho = None
    if len(ok) >= 3:
        rho = spearman([v["model_bottleneck_us"] for v in ok.values()],
                       [v["astra_max_cycles"] for v in ok.values()])
    # pairwise sign agreement: does each pair order the same way in both?
    agree = total = 0
    keys = list(ok)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = ok[keys[i]], ok[keys[j]]
            m = a["model_bottleneck_us"] - b["model_bottleneck_us"]
            s = a["astra_max_cycles"] - b["astra_max_cycles"]
            total += 1
            agree += int((m > 0) == (s > 0))
    print(f"\n   Spearman(model, astra) = {rho if rho is None else round(rho, 3)}")
    print(f"   pairwise orderings agreeing = {agree}/{total}")

    dest = args.out or Path(f"outputs/astra_sim/placement_order.{args.workload.name}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "workload": str(args.workload), "world_size": W,
        "source_model": args.source_model, "k": args.k,
        "unit": "one MoE layer, custom collective, congestion-unaware analytical backend",
        "spearman": None if rho is None else round(rho, 4),
        "pairwise_agree": agree, "pairwise_total": total, "rows": rows}, indent=1))
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
