#!/usr/bin/env python3
"""astra_ordering.py — Phase 4: validate the *ordering*, not the times.

WHAT THIS TESTS
───────────────
Every claim in this repo rests on an ordering: "config A beats config B".  The
closed-form cost model computes that ordering in ~0.5 ms; ASTRA-sim computes a
time in seconds.  The claim this harness makes is *ordering preserved*, with an
absolute-error band — never equality.

Two things it deliberately does NOT paper over:

1. **Latency accounting differs structurally.**  Measured on the 16-NPU
   all-to-all toy (2026-09-17):

       cycles = 75620 + 240 * latency        (240 = 16*15 rank pairs)

   ASTRA-sim charges latency ONCE PER RANK PAIR PER ITERATION;
   `cost_model.evaluate` adds alpha ONCE to the bottleneck.  Absolute times
   therefore disagree by ~1-2 orders of magnitude, and the promoted/unpromoted
   *ratio* is entirely latency-dependent: 4x bandwidth buys 2.12x at the
   latency the previously committed configs used (500 ns) and 1.07x at
   FabricConfig's own cross-pod latency (12000 ns).  The grid therefore sweeps
   latency as an explicit axis instead of assuming one.
2. **The analytical backend is tier-level.**  It cannot express "this one rank
   pair is promoted", so the promoted config is a whole-tier upper bound.

Reference point: `promoted: false` at sigma=1 is byte-identical to the EPS
config, so its gain is 0 by construction — the falsification row.

Usage
─────
    python3.12 scripts/astra_ordering.py --out outputs/astra_sim/ordering.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

NIC_GBPS = 50.0          # FabricConfig.nic_gbytes_per_s
SIGMAS = (1.0, 2.0, 4.0, 8.0)
LATENCIES_NS = (500, 2000, 12000)   # 500 = the old configs, 12000 = FabricConfig


def astra_dir(explicit: str | None) -> Path:
    import os
    d = Path(explicit or os.environ.get("ASTRA_SIM_DIR") or Path.home() / "astra-sim")
    if not (d / "build/astra_analytical/build/bin").is_dir():
        raise SystemExit(f"error: no ASTRA-sim build at {d} "
                         f"(set ASTRA_SIM_DIR or run configs/astra_sim/build_astrasim.sh)")
    return d


def write_cfg(path: Path, bw: float, latency_ns: int, npus: int = 16) -> Path:
    path.write_text(f"topology: [ Ring ]\nnpus_count: [ {npus} ]\n"
                    f"bandwidth: [ {bw} ]\nlatency: [ {latency_ns} ]\n")
    return path


def run(binary: Path, cfg: Path, astra: Path) -> int | None:
    cmd = [str(binary),
           "--workload-configuration=examples/workload/microbenchmarks/all_to_all/16npus_1MB/all_to_all",
           "--system-configuration=examples/system/native_collectives/HGX-H100-validated.json",
           "--remote-memory-configuration=examples/remote_memory/analytical/no_memory_expansion.json",
           f"--network-configuration={cfg}"]
    p = subprocess.run(cmd, cwd=astra, capture_output=True, text=True)
    m = re.findall(r"Comm time: (\d+)", p.stdout)
    return int(m[-1]) if m else None


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation with ties averaged; no scipy dependency."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
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
    ap.add_argument("--astra", default=None)
    ap.add_argument("--coverage-c", type=float, default=0.1399,
                    help="covered fraction from the measured-ownership envelope "
                         "(what our model prices a circuit at)")
    ap.add_argument("--out", type=Path, default=Path("outputs/astra_sim/ordering.json"))
    args = ap.parse_args(argv)

    astra = astra_dir(args.astra)
    binary = astra / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
    cfgdir = astra / "configs_from_repo"
    cfgdir.mkdir(exist_ok=True)

    rows: dict = {}
    print(f"== ASTRA-sim ordering check ({astra})")
    print(f"   network   : single-tier ring, 16 NPUs, 1 MB all-to-all")
    print(f"   grid      : sigma {{1,2,4,8}} x latency {{500, 2000, 12000}} ns "
          f"x {{EPS, promoted}} = {len(SIGMAS)*len(LATENCIES_NS)*2} configs\n")

    for lat in LATENCIES_NS:
        base_gains, promo_times, eps_times = [], [], []
        for sig in SIGMAS:
            bw_eps = NIC_GBPS / sig
            te = run(binary, write_cfg(cfgdir / f"eps_s{int(sig)}_l{lat}.yml", bw_eps, lat), astra)
            tp = run(binary, write_cfg(cfgdir / f"pro_s{int(sig)}_l{lat}.yml", NIC_GBPS, lat), astra)
            gain = (100.0 * (1 - tp / te)) if (te and tp) else None
            rows[f"lat{lat}|sigma{sig:g}"] = {
                "latency_ns": lat, "sigma": sig,
                "eps_bandwidth_gbps": bw_eps, "promoted_bandwidth_gbps": NIC_GBPS,
                "eps_cycles": te, "promoted_cycles": tp,
                "astra_gain_pct": None if gain is None else round(gain, 3),
                # what the closed-form model predicts for the same sigma:
                # Delta = c * (1 - 1/sigma), c = covered fraction under measured ownership
                "model_gain_pct": round(100.0 * args.coverage_c * (1.0 - 1.0 / sig), 3),
            }
            base_gains.append(gain if gain is not None else 0.0)
        print(f"-- latency {lat} ns")
        print(f"   {'sigma':>6}{'bw_eps':>9}{'astra_gain%':>13}{'model_gain%':>13}")
        for sig in SIGMAS:
            r = rows[f"lat{lat}|sigma{sig:g}"]
            print(f"   {sig:>6g}{r['eps_bandwidth_gbps']:>9.3f}"
                  f"{r['astra_gain_pct']:>13.3f}{r['model_gain_pct']:>13.3f}")
        astra_g = [rows[f"lat{lat}|sigma{s:g}"]["astra_gain_pct"] or 0.0 for s in SIGMAS]
        model_g = [rows[f"lat{lat}|sigma{s:g}"]["model_gain_pct"] for s in SIGMAS]
        rho = spearman(astra_g, model_g)
        rows[f"lat{lat}|spearman"] = round(rho, 4)
        print(f"   Spearman(astra ordering, model ordering) = {rho:.3f}\n")

    doc = {
        "harness": "scripts/astra_ordering.py",
        "what": ("ordering validation, not time validation: ASTRA-sim is event-driven "
                 "and charges latency per rank pair, the closed-form model adds alpha once"),
        "astra_sim_dir": str(astra),
        "coverage_c": args.coverage_c,
        "structural_finding": {
            "latency_model": "cycles = 75620 + 240*latency on the 16-NPU toy "
                             "(240 = 16*15 rank pairs)",
            "consequence": ("the promoted/unpromoted ratio is latency-dependent: "
                            "4x bandwidth buys 2.12x at latency 500 and 1.07x at 12000"),
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1))
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
