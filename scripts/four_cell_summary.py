#!/usr/bin/env python3
"""four_cell_summary.py — turn the four-cell JSON outputs into markdown tables.

    python3 scripts/four_cell_summary.py outputs/four_cell/*.json

Every number it prints comes from one ``ocs_four_cell.py`` run, so a slide can
carry the command that produced it (the repo's P0 rule) rather than a number
that only exists in a chat log.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_RANK = ["affinity_coordinated_layer", "load_balanced_layer", "load_balanced",
         "linear", "random", "affinity_layer", "affinity_global",
         "promote_aware_layer", "bottleneck_search_layer"]


def _row(name: str, c: dict) -> str:
    if not c.get("applicable", False):
        return f"| `{name}` | — | not applicable | — | — |"
    return (f"| `{name}` | {c['eps_bottleneck_us']:.1f} "
            f"| {c['static_reduction_pct']:+.2f} "
            f"| {c['oracle_reduction_pct']:+.2f} "
            f"| {c.get('n_circuits', 0)} / {c.get('covered_fraction', 0) * 100:.1f}% "
            f"| {c.get('port_saturated_ranks', 0)} |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    lines: list[str] = []
    for f in args.files:
        d = json.load(open(f))
        title = f"{d['model']} (E={d['num_experts']}, K={d['top_k']}, " \
                f"{d['n_moe_layers']} MoE layers, world={d['world_size']})"
        lines.append(f"\n## {title}\n")
        lines.append(f"`{f}` — fit {d['fit']['cells']} cells / eval "
                     f"{d['eval']['cells']} cells, leave-categories-out\n")

        fc = d.get("four_cell")
        if fc:
            t = fc["topology"]
            lines.append(f"Regime **{fc['regime']['name']}** "
                         f"({t['n_pods']} pods, {t['n_nodes']} nodes, "
                         f"σ_core={fc['regime']['core_os']}, "
                         f"σ_pod={fc['regime']['pod_os']}); "
                         f"{fc['ocs']['n_circuits']} circuits, "
                         f"{fc['ocs']['ports_per_rank']} ports/rank\n")
            lines.append("| placement | EPS bottleneck (us) | static OCS gain | "
                         "oracle OCS gain | circuits / promotable covered | "
                         "port-saturated ranks |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            cells = fc["cells"]
            for name in _RANK:
                if name in cells:
                    lines.append(_row(name, cells[name]))
            for name, c in cells.items():
                if name not in _RANK:
                    lines.append(_row(name, c))

            two = fc.get("two_by_two")
            if two:
                lines.append("\n**The 2x2** (A = `linear` EPS, "
                             "B = affinity-coordinated EPS, C = `linear` + OCS, "
                             "D = affinity-coordinated + OCS)\n")
                lines.append("| cell | bottleneck (us) | vs A |")
                lines.append("| --- | --- | --- |")
                base = two["A_eps_no_affinity_us"]
                for k, label in (("A_eps_no_affinity_us", "A  eps, no affinity"),
                                 ("B_eps_affinity_us", "B  eps, affinity"),
                                 ("C_ocs_no_affinity_us", "C  ocs, no affinity"),
                                 ("D_ocs_affinity_us", "D  ocs + affinity")):
                    v = two[k]
                    lines.append(f"| {label} | {v:.1f} "
                                 f"| {100 * (1 - v / base):+.2f}% |")
                lines.append(f"\n- affinity alone: **{two['delta_affinity_pct']:+.2f}%**")
                lines.append(f"- OCS alone: **{two['delta_ocs_pct']:+.2f}%**")
                lines.append(f"- both: **{two['delta_both_pct']:+.2f}%**")
                lines.append(f"- synergy `delta_both - (delta_aff + delta_ocs)` = "
                             f"**{two['synergy_pct']:+.2f}%** -> "
                             f"**{two['verdict']}**")
                lines.append(f"- D is the minimum cell: **{two['D_is_minimum']}** "
                             f"(best = {two['best_cell']})")

            cd = fc.get("co_design")
            if cd:
                lines.append("\n**Co-design (the fourth cell), vs the best "
                             "electrical placement**\n")
                lines.append("| search | on the OCS substrate | on the EPS "
                             "substrate | its own circuit gain |")
                lines.append("| --- | --- | --- | --- |")
                for k, v in cd.items():
                    lines.append(f"| `{k}` | "
                                 f"{v['ocs_improvement_vs_electrical_pct']:+.2f}% | "
                                 f"{v['eps_improvement_vs_electrical_pct']:+.2f}% | "
                                 f"{v['circuit_gain_pct']:+.2f}% |")
                lines.append("\n(`bottleneck_search_layer` is the control: the "
                             "same search with no circuits at all. If it loses on "
                             "the EPS substrate too, a co-design loss is the "
                             "search's fault rather than OCS's.)")

        rg = d.get("regime")
        if rg:
            lines.append("\n**Static OCS gain by regime** (per placement)\n")
            names = sorted({k for r in rg["regimes"].values()
                            for k in r["cells"]})
            lines.append("| regime | " + " | ".join(f"`{n}`" for n in names) + " |")
            lines.append("| --- | " + " | ".join("---" for _ in names) + " |")
            for rname, r in rg["regimes"].items():
                vals = []
                for n in names:
                    c = r["cells"].get(n, {})
                    vals.append(f"{c['static_reduction_pct']:+.2f}%"
                                if c.get("applicable") else "n/a")
                lines.append(f"| {rname} | " + " | ".join(vals) + " |")

            ps = rg.get("port_sweep") or {}
            if ps:
                lines.append("\n**Port / circuit budget envelope** "
                             "(focus regime)\n")
                lines.append("| placement | circuits | ports/rank | static OCS gain "
                             "| oracle gain | value of prediction | covered |")
                lines.append("| --- | --- | --- | --- | --- | --- | --- |")
                for key in sorted(ps):
                    place, cpart, ppart = key.split("|")
                    v = ps[key]
                    if v.get("applicable", True) is False:
                        continue
                    lines.append(f"| `{place}` | {cpart[1:]} | {ppart[1:]} "
                                 f"| {v['static_reduction_pct']:+.2f}% "
                                 f"| {v['oracle_reduction_pct']:+.2f}% "
                                 f"| {v['value_of_prediction_pct']:+.2f}% "
                                 f"| {v['covered_fraction'] * 100:.1f}% |")

            re_ = rg.get("reconfig_envelope") or {}
            if re_:
                lines.append("\n**Reconfiguration class envelope** "
                             "(affinity-coordinated placement)\n")
                lines.append("| class | reconfig | static OCS gain | "
                             "breakeven token passes |")
                lines.append("| --- | --- | --- | --- |")
                for cls, v in re_.items():
                    if v.get("applicable", True) is False:
                        lines.append(f"| {cls} | — | not applicable | — |")
                        continue
                    lines.append(f"| {cls} | {v['reconfig_us']:.0f} us "
                                 f"| {v['static_reduction_pct']:+.2f}% "
                                 f"| {v['breakeven_token_passes']} |")

        dp = d.get("dispatch")
        if dp:
            lines.append("\n**Dispatch semantics** (the verdict must be reported "
                         "per mode; see F10)\n")
            lines.append("| mode | total-byte spread across placements | affinity "
                         "| OCS | both | synergy | verdict |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for mode, e in dp.get("by_mode", {}).items():
                t = e.get("two_by_two")
                if not t:
                    continue
                lines.append(
                    f"| {mode} | {e.get('total_bytes_spread_pct')}% "
                    f"| {t['delta_affinity_pct']:+.2f}% "
                    f"| {t['delta_ocs_pct']:+.2f}% "
                    f"| {t['delta_both_pct']:+.2f}% "
                    f"| {t['synergy_pct']:+.2f}% | {t['verdict']} |")
            lines.append("\n(`total-byte spread` is the placement-to-placement "
                         "spread in dispatch volume: exactly 0 under REPLICATED, "
                         "non-zero under the DEDUP modes — a numeric check of the "
                         "documented dispatch semantics.)")

        tm = d.get("timing")
        if tm:
            rows = tm["models"].get("comm_only", {})
            lines.append("\n**Completion time, communication component only** "
                         f"(prefill {tm['prefill_cells']} cells, "
                         f"decode step {tm['decode_step_cells']} cells)\n")
            lines.append("| placement | EPS TTFT (us) | OCS TTFT (us) | TTFT gain "
                         "| EPS ITL (us) | OCS ITL (us) | ITL gain |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for name in _RANK:
                v = rows.get(name)
                if not v or v.get("applicable", True) is False:
                    continue
                lines.append(f"| `{name}` | {v['eps']['ttft_us']:.1f} "
                             f"| {v['static_ocs']['ttft_us']:.1f} "
                             f"| {v.get('ttft_reduction_pct', 0):+.2f}% "
                             f"| {v['eps']['itl_us']:.1f} "
                             f"| {v['static_ocs']['itl_us']:.1f} "
                             f"| {v.get('itl_reduction_pct', 0):+.2f}% |")

        ml = d.get("milp")
        if ml:
            lines.append("\n**Optimality bounds**\n")
            for lname, rep in ml["layers"].items():
                g = rep["gaps"]
                lines.append(f"layer {lname}: {rep['n_cells']} cells, "
                             f"{rep['n_distinct_sets']} distinct expert sets\n")
                lines.append("| placement | linearised max | exact ingress max | "
                             "vs linear bound | vs union bound |")
                lines.append("| --- | --- | --- | --- | --- |")
                for k, v in rep["heuristics"].items():
                    lb = g.get("linear_vs_bound_pct", {}).get(k)
                    ub = g.get("union_vs_bound_pct", {}).get(k)
                    lines.append(f"| `{k}` | {v['linearized_max']:.0f} "
                                 f"| {v['true_ingress_max']} "
                                 f"| {'—' if lb is None else f'{lb:+.1f}%'} "
                                 f"| {'—' if ub is None else f'{ub:+.1f}%'} |")
                ub = rep.get("union_bound", {})
                lines.append(f"\n- linear-objective bound: "
                             f"**{g.get('linear_bound')}** "
                             f"({rep['linear_lp']['kind']}, "
                             f"{rep['linear_lp']['n_constraints']} constraints)")
                if "linear_mip" in rep:
                    m = rep["linear_mip"]
                    lines.append(f"- linear MIP: bound "
                                 f"{m['bound_used']}, incumbent {m['objective']}, "
                                 f"gap {m['mip_gap']} ({m['wall_s']}s)")
                lines.append(f"- union bound: **{g.get('union_bound_value')}** "
                             f"= max(perfectly-spread ideal "
                             f"{ub.get('perfectly_spread_ideal')}, largest expert "
                             f"reach {ub.get('largest_expert_reach')})")

    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
