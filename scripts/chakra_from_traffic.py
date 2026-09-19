#!/usr/bin/env python3
"""chakra_from_traffic.py — Phase 4a: turn OUR per-pair byte matrix into a
Chakra ET workload that ASTRA-sim can execute.

WHY THIS IS THE SEAM
────────────────────
ASTRA-sim's workload layer states a *collective and its size*; the per-pair split
comes from the system layer, which for a native collective means an algorithm
(ring, halving-doubling) — uniform by construction.  Our traffic is neither
uniform nor symmetric: it is a placement-dependent, uneven rank x rank byte
matrix, and the placement is the thing we are trying to validate.  Feeding
ASTRA-sim a uniform all-to-all (as the committed cross-check does) therefore
validates the *fabric* and nothing about placement.

This generator writes the matrix out explicitly, as per-rank ET files containing
COMM_SEND_NODE / COMM_RECV_NODE pairs with the real byte counts, so ASTRA-sim
executes our traffic instead of a synthetic collective.

WHAT IS VERIFIED AND WHAT IS NOT
────────────────────────────────
Verified here: the ET container format (GlobalMetadata version 0.0.4, then
chunked Node messages), that the Chakra protobuf bindings import and round-trip
under python3.12 + protobuf 6.33.2, and that the generated files decode back to
the exact byte counts.

NOT verified: that ASTRA-sim's **system layer executes explicit send/recv nodes**
without a collective implementation registered.  The documented custom-collective
route (`examples/system/custom_collectives/`) goes through MSCCLang and a separate
`collectiveapi` clone, which is not present in this checkout.  Until that is
settled, this generator produces correct ET files; wiring them through the system
layer is the open step.

Usage
─────
    python3.12 scripts/chakra_from_traffic.py --workload logs/workload/qwen36 \
        --world-size 32 --placement linear --source-model measured_packed_4 \
        --out-dir outputs/astra_sim/et/linear
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "scripts"))

DEFAULT_ASTRA = Path.home() / "astra-sim"


def load_chakra(astra: Path):
    if str(astra) not in sys.path:
        sys.path.insert(0, str(astra))
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (  # noqa: E402
        ALL_TO_ALL, COMM_COLL_NODE, COMM_RECV_NODE, COMM_SEND_NODE, GlobalMetadata, Node,
    )
    from extern.graph_frontend.chakra.src.third_party.utils.protolib import (  # noqa: E402
        encodeMessage,
    )
    return (GlobalMetadata, Node, COMM_SEND_NODE, COMM_RECV_NODE,
            COMM_COLL_NODE, ALL_TO_ALL, encodeMessage)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--placement", default="linear")
    ap.add_argument("--source-model", default="hash")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--astra", type=Path, default=DEFAULT_ASTRA)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--single-layer", action="store_true",
                    help="emit ONE MoE layer's traffic instead of the whole trace. "
                         "The aggregate (6.4 GB for qwen36) is unsurvivable for a "
                         "packet-level simulator; a per-layer pass is the right unit.")
    args = ap.parse_args(argv)

    import src.eval.cost_model as cm  # noqa: E402
    from src.eval.cost_model import (  # noqa: E402
        CostConfig, DispatchMode, FabricConfig, Tier, Topology, hierarchy_for,
        traffic_matrix,
    )
    from src.eval.placement_opt import make_placement  # noqa: E402
    from src.eval.trace_ir import load_workload  # noqa: E402
    from ownership_ocs_envelope import patch_token_rank, restore_token_rank  # noqa: E402

    (GlobalMetadata, Node, COMM_SEND, COMM_RECV,
     COMM_COLL, ALL_TO_ALL, encode_message) = load_chakra(args.astra)

    W = args.world_size
    cm.token_rank_original = cm.token_rank
    t = load_workload(args.workload / "manifest.json", decode_only=True)
    cost = CostConfig(hidden_size=2048)
    base = hierarchy_for(W, "multi_pod")
    fab = FabricConfig(core_oversubscription=4.0, pod_oversubscription=1.0)
    topo = Topology(W, base.gpus_per_node, base.nodes_per_pod, fab, set(),
                    base.rank_to_slot, (Tier.CROSS_POD,))

    try:
        patch_token_rank(args.source_model, args.k)
        pl = make_placement(args.placement, t, W, seed=args.seed)
        # NOTE: only the FIRST MoE layer is sliced.  A per-layer Placement indexes
        # its row by layer position, so handing traffic_matrix a subset containing
        # a single layer is correct only when that layer is index 0.
        src = t.by_layer(int(np.sort(np.unique(t.layers))[0])) if args.single_layer else t
        tm = traffic_matrix(src, pl, topo, DispatchMode.DEDUP_RANK, n_dp=W, seed=args.seed)
    finally:
        restore_token_rank()

    B = cost.hidden_size * cost.dtype_bytes
    mat = (tm.counts * B).astype(np.int64)      # [W, W] bytes, src -> dst
    np.fill_diagonal(mat, 0)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.placement}_{args.source_model}"

    node_id = 0
    per_rank = []
    for r in range(W):
        sends = [(int(mat[r, d]), int(d)) for d in range(W) if mat[r, d] > 0]
        recvs = [(int(mat[s, r]), int(s)) for s in range(W) if mat[s, r] > 0]
        sends.sort(reverse=True)
        recvs.sort(reverse=True)
        # paired, largest first: a deterministic, documented order -- the
        # analytical backend is order-sensitive under FIFO scheduling
        body = []
        for i in range(max(len(sends), len(recvs))):
            if i < len(sends):
                body.append(("send", sends[i]))
            if i < len(recvs):
                body.append(("recv", recvs[i]))

        with open(out / f"{prefix}.{r}.et", "wb") as f:
            encode_message(f, GlobalMetadata(version="0.0.4"))
            for kind, (size, peer) in body:
                n = Node()
                n.id = node_id
                n.name = f"{kind}_{r}_to_{peer}"
                n.type = COMM_SEND if kind == "send" else COMM_RECV
                from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (  # noqa: E402
                    AttributeProto as Attr,
                )
                n.attr.append(Attr(name="is_cpu_op", bool_val=False))
                n.attr.append(Attr(name="comm_size", int64_val=size))
                if kind == "send":
                    n.attr.append(Attr(name="comm_src", int64_val=r))
                    n.attr.append(Attr(name="comm_dst", int64_val=peer))
                else:
                    n.attr.append(Attr(name="comm_src", int64_val=peer))
                    n.attr.append(Attr(name="comm_dst", int64_val=r))
                n.attr.append(Attr(name="comm_tag", int64_val=0))
                encode_message(f, n)
                node_id += 1
        per_rank.append({"rank": r, "n_sends": len(sends), "n_recvs": len(recvs),
                         "egress_bytes": int(sum(s for s, _ in sends)),
                         "ingress_bytes": int(sum(s for s, _ in recvs))})

    egress = np.array([p["egress_bytes"] for p in per_rank], dtype=float)
    manifest = {
        "workload": str(args.workload), "world_size": W,
        "placement": args.placement, "source_model": args.source_model, "k": args.k,
        "bytes_per_pair_total": int(mat.sum()),
        "max_egress_rank": int(egress.argmax()),
        "max_egress_bytes": int(egress.max()),
        "et_prefix": str(out / prefix),
        "per_rank": per_rank,
        "protocol": ("send/recv generated from our per-pair byte matrix; largest-first "
                     "pairing; byte counts are evaluate()'s counts * hidden*dtype"),
    }
    (out / f"{prefix}.manifest.json").write_text(json.dumps(manifest, indent=1))

    # ── the trigger: a workload ET whose collective selects the custom path ──
    # CustomAlgorithm::issue() reads the byte counts from the CUSTOM et and
    # ignores the size in the workload et ("we're using the comm size as hardcoded
    # in the Impl Chakra et ... and ignore the comm.size fed in the workload"),
    # so this file only has to make ASTRA-sim issue an all-to-all.
    wl_prefix = f"{prefix}.wl"
    for r in range(W):
        with open(out / f"{wl_prefix}.{r}.et", "wb") as f:
            encode_message(f, GlobalMetadata(version="0.0.4"))
            from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (  # noqa: E402
                AttributeProto as Attr,
            )
            n = Node()
            n.id = 0
            n.name = f"all_to_all_W{W}"
            n.type = COMM_COLL
            n.attr.append(Attr(name="is_cpu_op", bool_val=False))
            n.attr.append(Attr(name="comm_type", int64_val=ALL_TO_ALL))
            n.attr.append(Attr(name="comm_size", int64_val=1 << 20))   # ignored
            encode_message(f, n)

    # ── the system JSON that routes all-to-all to our ET ────────────────────
    system = {
        "scheduling-policy": "FIFO",
        "preferred-dataset-splits": 1,
        "all-to-all-implementation-custom": [str(out / prefix)],
        "local-mem-bw": 3350,
    }
    (out / f"{prefix}.system.json").write_text(json.dumps(system, indent=2))

    print(f"== chakra ET from traffic: {args.workload.name} W={W} "
          f"placement={args.placement} source={args.source_model}")
    print(f"   bytes on the wire : {mat.sum():,}")
    print(f"   hottest egress    : rank {manifest['max_egress_rank']} "
          f"({manifest['max_egress_bytes']:,} bytes)")
    print(f"   files             : {out}/{prefix}.<rank>.et  ({W} ranks)")
    print(f"-> {out}/{prefix}.manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
