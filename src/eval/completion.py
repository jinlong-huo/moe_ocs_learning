"""
completion.py — from a communication bottleneck to the quantities a serving
report actually quotes: TTFT, inter-token latency, and throughput.

Why this layer has to exist
───────────────────────────
``cost_model.evaluate`` returns ``bottleneck_us``: the drain time of **one MoE
all-to-all (dispatch + combine) over a whole trace**.  That is the right unit
for comparing placements and circuit plans, and the wrong unit for answering
"does the user wait less", because a served request is a prefill pass followed
by many single-token decode steps, and a circuit plan is paid for once and
reused across those steps.

The composition here is deliberately minimal and explicitly parameterised:

    TTFT = prefill_comm + compute + reconfig/N
    ITL  = decode_comm  + compute + reconfig/N
    thr  = n_sequences / ITL                     (decode tokens per second)

  * ``prefill_comm``  is ``evaluate`` on every prefill cell — one pass over the
    whole prompt batch.
  * ``decode_comm``   is ``evaluate`` on **one** decode position per sequence —
    the batch's per-step all-to-all.
  * ``compute``       is a *parameter*, not a measurement.  This repo has no
    kernel-level timings, and pretending otherwise would let a communication
    saving masquerade as an end-to-end one.  ``compute_us_per_layer = 0`` (the
    default) reports the communication component alone, which is the honest
    headline; a sensitivity sweep shows how the saving shrinks as compute grows,
    since OCS can only ever remove the communication part.
  * ``reconfig/N``    amortises one circuit reconfiguration over N token passes.
    This is the term that decides whether a circuit plan is feasible at all, so
    it is never folded into the communication cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.eval.cost_model import CostConfig, DispatchMode, Placement, Topology, evaluate
from src.eval.trace_ir import CellTable


@dataclass
class ServingModel:
    """Parameters the trace cannot supply.

    ``compute_us_per_layer``   measured compute per MoE layer, if known.  The
                               default of 0 reports communication only.
    ``reconfig_us``            switch reconfiguration time charged once per plan.
    ``passes_per_reconfig``    N — how many token passes one plan is held for.
                               N=1 is the pessimistic "reconfigure every step"
                               reading; large N is the static-plan reading.
    """

    compute_us_per_layer: float = 0.0
    reconfig_us: float = 0.0
    passes_per_reconfig: int = 1
    include_combine: bool = True

    def compute_us(self, n_layers: int) -> float:
        return self.compute_us_per_layer * n_layers

    def reconfig_per_pass_us(self) -> float:
        return self.reconfig_us / max(1, self.passes_per_reconfig)


def first_decode_step(t: CellTable) -> CellTable:
    """One decode position per sequence — the unit of inter-token latency."""
    dec = t.select(t.phase == 1)
    if dec.n_cells == 0:
        raise ValueError("no decode cells in this workload")
    first = np.full(int(dec.run.max()) + 1, np.iinfo(np.int32).max, dtype=np.int32)
    np.minimum.at(first, dec.run, dec.pos)
    return dec.select(dec.pos == first[dec.run])


def prefill_pass(t: CellTable) -> CellTable:
    return t.select(t.phase == 0)


def comm_us(t: CellTable, placement: Placement, topo: Topology,
            cost: CostConfig | None = None,
            mode: DispatchMode = DispatchMode.DEDUP_RANK, seed: int = 0) -> float:
    """The bottleneck of one all-to-all pass over ``t``."""
    return float(evaluate(t, placement, topo, cost or CostConfig(), mode,
                          seed=seed)["bottleneck_us"])


def n_sequences(t: CellTable) -> int:
    """How many sequences the slice actually contains.

    Not ``t.n_runs``: ``CellTable.select`` keeps the parent's ``runs`` list, so
    ``by_runs``/``by_category`` slices still report the whole workload's run
    count.  Using it inflates batch throughput by the ratio between the slice
    and the full workload.
    """
    return int(np.unique(t.run).size)


def timing_report(prefill_comm: float, decode_comm: float, t: CellTable,
                  model: ServingModel, n_seq: int | None = None) -> dict:
    """Compose the two pass costs into TTFT / ITL / throughput."""
    n_layers = t.n_layers
    n_seq = n_sequences(t) if n_seq is None else n_seq
    comp = model.compute_us(n_layers)
    rc = model.reconfig_per_pass_us()
    ttft = prefill_comm + comp + rc
    itl = decode_comm + comp + rc
    return {
        "ttft_us": round(ttft, 4),
        "itl_us": round(itl, 4),
        "throughput_tok_s": round(n_seq / (itl * 1e-6), 4) if itl > 0 else None,
        "prefill_comm_us": round(prefill_comm, 4),
        "decode_comm_us": round(decode_comm, 4),
        "compute_us": round(comp, 4),
        "reconfig_per_pass_us": round(rc, 4),
        "comm_share_of_itl": round(decode_comm / itl, 6) if itl > 0 else None,
        "n_sequences": n_seq,
        "n_layers": n_layers,
    }


def speedup(base: dict, variant: dict) -> dict:
    """Relative improvement of ``variant`` over ``base`` (positive = faster)."""
    out = {}
    for k, label in (("ttft_us", "ttft"), ("itl_us", "itl"),
                     ("throughput_tok_s", "throughput")):
        b, v = base.get(k), variant.get(k)
        if not b or not v:
            continue
        out[f"{label}_reduction_pct"] = round(100.0 * (1.0 - v / b), 4)
    return out
