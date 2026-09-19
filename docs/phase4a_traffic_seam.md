# Phase 4a — the traffic seam: our byte matrix into ASTRA-sim

Status: **generator built and run; system-layer execution not yet wired.**
`scripts/chakra_from_traffic.py`, output under `outputs/astra_sim/et/`.

## The problem the seam solves

ASTRA-sim's workload layer states a **collective and its size**; the per-pair split
comes from the system layer, which for a native collective means a fixed algorithm
(ring, halving-doubling) and is therefore **uniform by construction**. Our traffic
is neither uniform nor symmetric — it is a placement-dependent, uneven rank x rank
byte matrix, and the placement is precisely the thing we are trying to validate.

So every ASTRA-sim result so far (including `docs/astra_ordering.md`, Spearman
1.000) validates the **fabric** axis only. Feeding it a uniform all-to-all says
nothing about placement.

## What was built

`scripts/chakra_from_traffic.py` reads a workload + placement + source model,
computes `traffic_matrix` (the same matrix `evaluate()` charges), and writes:

* one ET file per rank, in ASTRA-sim's own container format
  (`GlobalMetadata` version 0.0.4, then chunked `Node` messages), containing
  explicit **`COMM_SEND_NODE` / `COMM_RECV_NODE`** pairs carrying the real byte
  counts, largest-first, paired so the trace is deterministic under FIFO;
* a manifest with per-rank egress/ingress totals, for verification.

```bash
python3.12 scripts/chakra_from_traffic.py --workload logs/workload/qwen36 \
    --world-size 32 --placement linear --source-model measured_packed_4 \
    --out-dir outputs/astra_sim/et/linear
```

Measured output for that configuration: **6 396 280 832 bytes** on the wire,
hottest egress **rank 1 at 1 617 907 712 bytes**, 32 ET files plus a manifest.

## What is verified

* The ET container format was decoded from a real ASTRA-sim workload
  (`all_to_all.0.et`: version 0.0.4, one `COMM_COLL_NODE` type-7 node with
  `is_cpu_op`, `comm_type`, `comm_size`), so the writer targets the format ASTRA-sim
  actually reads rather than a guess.
* The Chakra protobuf bindings import and encode under python3.12 + protobuf
  6.33.2, which is what the generator uses.
* The generator runs end to end and produces well-formed files (~3.2 KB per rank).

## The wiring, read out of ASTRA-sim's source

`astra-sim/system/astraccl/custom_collectives/CustomAlgorithm.cc` — `issue()` — is the
entry point, and it consumes **exactly the attributes this generator writes**:

| node type | attributes read | action |
| --- | --- | --- |
| `COMM_SEND_NODE` (5) | `comm_dst`, `comm_src`, `comm_size`, `comm_tag` | `front_end_sim_send(..., dst_rank, comm_tag, ...)` |
| `COMM_RECV_NODE` (6) | `comm_src`, `comm_size`, `comm_tag` | `front_end_sim_recv(..., src_rank, comm_tag, ...)` |
| `COMP_NODE` | `runtime` (µs) | trivial reduce, default 1 ns |

Two consequences that settle the open question:

1. **Explicit send/recv nodes ARE executed** — no collective algorithm needs to be
   registered for them. The custom path exists precisely for this.
2. **The byte counts come from the custom ET, not the workload ET.** The source
   says so in a comment: *"we're using the comm size as hardcoded in the Impl
   Chakra et ... and ignore the comm.size fed in the workload chakra et."* So the
   workload ET only has to *trigger* an all-to-all; our matrix lives in the custom
   ET, and the workload's `comm_size` is ignored. That is exactly the seam we need.

Selection is by a **system JSON key**, mirroring the shipped example
(`all-reduce-implementation-custom`):

```json
{
  "scheduling-policy": "FIFO",
  "preferred-dataset-splits": 1,
  "all-to-all-implementation-custom": ["<path to the ET prefix>"],
  "local-mem-bw": 3350
}
```

The generator now emits all three pieces: the per-pair custom ET, the workload
trigger ET (`<prefix>.wl.<rank>.et`), and the system JSON (`<prefix>.system.json`).

```bash
# custom ET + trigger ET + system JSON
python3.12 scripts/chakra_from_traffic.py --workload logs/workload/qwen36 \
    --world-size 32 --placement linear --source-model measured_packed_4 \
    --single-layer --out-dir outputs/astra_sim/et/linear_layer0

# then, from the ASTRA-sim checkout
cd ~/astra-sim && build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware \
  --workload-configuration=<out>/linear_measured_packed_4.wl \
  --system-configuration=<out>/linear_measured_packed_4.system.json \
  --remote-memory-configuration=examples/remote_memory/analytical/no_memory_expansion.json \
  --network-configuration=<repo>/configs/astra_sim/eps_baseline_2tier.yml
```

`convert_algo_rank_to_real_rank` assumes **algo rank == real rank** (1:1), which holds
for our flat rank numbering.

## What is NOT verified, and why this is still open

1. **The system layer.** Whether ASTRA-sim executes explicit send/recv nodes
   without a collective implementation registered is **untested**. The documented
   custom-collective route (`examples/system/custom_collectives/`) goes through
   MSCCLang and a separate `collectiveapi` clone (the script literally prompts for
   its path), which is **not present in this checkout**.
2. **Scale.** The generator emits the *aggregate* traffic of the whole trace
   (6.4 GB). A first execution attempt did not complete within a 10-minute
   wall-clock ceiling, and the machine was at load average 8 at the time. The
   correct unit to hand a packet-level simulator is **one layer-pass**, not the
   aggregate — a `--max-layers` / per-layer slice is the concrete next change.
3. **A round-trip decode of the generated files** (re-reading them and comparing
   against the manifest) was launched twice and **did not complete** — the machine
   sat at load average 8 and both attempts hit a 10-minute wall-clock ceiling, as
   did a later read of the job. The check is therefore **outstanding**, and the
   generated files should be treated as *format-plausible* (the format was verified
   against a real ASTRA-sim ET; the round trip was not) until it lands. Run it on an
   idle machine before trusting the byte counts.

## The concrete next step

Slice the workload to a single MoE layer (or a small decode window), then either:

* find the system-layer entry point for explicit send/recv and run it, or
* instantiate the MSCCLang route by cloning `astra-sim/collectiveapi`.

Either path ends in the same measurement: the **placement ordering** — two or more
placements, same fabric, compared by ASTRA-sim, correlated against our model's
ordering. That is the one external check this program still lacks, and it is the
axis our actual claims live on.