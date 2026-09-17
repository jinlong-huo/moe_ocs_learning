# Final comment on OptiTrainSim (文献3) — what it changes here, and the better version of its experiment

Status: comment, no code change. Nothing in the evidence chain is modified by
this document. Source: *OptiTrainSim: Multi-Job Distributed AI Training over
Reconfigurable Optical Datacenters*, 2026, 7 pp. (Zotero `RMX936IK`; PDF also at
`~/Downloads/work/inet4ai26-paper7.pdf`). Slide material:
`~/Downloads/PhD/PPT/JL/OptiTrainSim_lit_slides.{md,pptx}`.

**Verdict.** OptiTrainSim is a *simulator and design-space* paper, not a
scheduler paper: it never optimises CCT (the ILP minimises the number of OCS
configurations), it reports no EPS-only baseline, and its headline gains
(−30 % CCT with the ILP; −30/−53/−62 % JCT for 1/2/3/4 OCSes) are all
*scheduling* deltas measured inside one fabric. Read against this repo, it is
complementary rather than contradictory: it isolates the regime in which OCS
scheduling pays, and our evidence chain says our workloads are not in it.

---

## 1. What actually transfers

1. **The TSW / reconfiguration-delay axis is our $\alpha_r$ question in
   disguise.** A time-slice width is just a *reconfiguration frequency*; their
   Fig. 3/4 sweep is the special case of our segment DP in which every segment
   has the same length. `src/eval/harvest_sched.py` answers the question their
   sweep assumes away — *whether to reconfigure at all* — and charges
   $\alpha_r$ per **actual** circuit-set change instead of one reconfiguration
   per slice. Same physical cost, strictly more decision freedom.
2. **Their Eq. 2 is our A6 "circuit budget", published.** A $p$-port OCS
   configuration is a permutation matrix (one-to-one ingress↔egress), so a
   dense demand matrix cannot be served by one configuration —
   $DM \le \sum_k M[k]$ — which is exactly the port-limited β-class behaviour
   `docs/assumptions.md` A6 describes (single outgoing port, serially
   re-pointed) and exactly the constraint the plan builder encodes through
   `max_circuits`. Their Eq. 1 gives us a citation for "why a schedule is a
   *sequence* of circuit sets" that does not depend on our own model.
3. **"In-collective reconfiguration" is the published name for our per-layer
   plan churn.** Their Fig. 3d–f (DP-dominated jobs, non-monotonic in TSW,
   workload-specific optimum) is the closest published analogue of our
   per-layer plan instability. It gives us the mechanism to cite when
   contrasting with our measurement — and our measurement (circuit-plan
   Jaccard ≈ 0.09 across windows, churn = weight-tie noise) is the missing
   *evidence* their paper cannot supply.
4. **Soft preemption at collective boundaries is the runtime primitive our
   schedule proposal lacks.** Their checkpoint signal injected into the
   workload = "rewire only between steps/layers" as an actual system design.
   `docs/proposal_harvest_schedule.md` §4 gate 4 ("schedule replay; no
   execution exists yet") should cite this as prior art for the boundary,
   not invent it.

## 2. Why their positive result and our negative result coexist

Their own ablation localises the gain: the biggest step is 1 → 2 OCSes
(38.1 → 26.8 s, i.e. 30 of the total 62 points), and they attribute it to
**cross-job circuit contention**, not link bandwidth. Our four measured
negatives remove that precondition for MoE inference windows:

| their precondition | our measurement |
| --- | --- |
| concurrent jobs contending for the same OCS circuits | aggregate rank×rank traffic ≈ rank-1 (no pairwise structure) |
| dense, multi-configuration demand (`DM` not coverable by one $M_k$) | at realistic pod sizes (256 GPU/pod) no EP degree these models reach produces cross-pod traffic |
| dynamic demand worth re-planning per period | plan Jaccard ≈ 0.09 across request/layer windows → churn is weight-tie noise |
| OCS in the critical path | static fit-only plan captures ≈ 8.9 % of the critical path; DP certifies $k^* = 0$ under MEMS-class $\alpha_r$ |

So OptiTrainSim tells us *where* OCS scheduling pays (multi-tenant, circuit-
contention-bound, port-limited fabrics with dense demand) and this repo shows
our workloads are not there. That is a stronger pairing than either result
alone: it converts both papers from "OCS good/bad" into a **regime boundary**.

## 3. If there is a better way — the better version of their experiment

Ranked by value per unit of work:

1. **Sweep $k^*(\alpha_r)$, not TSW.** Run the segment DP over *their*
   workloads' step sequences (their traces are synthetic but their step
   structure is published) and report the optimal rewire count as a function
   of $\alpha_r$. Expect: interior optima only for large messages and cheap
   $\alpha_r$; uniform-segment TSW sweeps are recovered as the one-dimensional
   slice through that surface. This is a publishable generalisation of their
   Fig. 3/4 and it is already implemented here
   (`scripts/harvest_schedule_demo.py`, gates in the proposal §4).
2. **Supply the baseline they never ran.** When the simulator is open-sourced
   (promised on acceptance), replay our bit-exact per-layer MoE demand through
   their packet-level OCS fabric with the OCS layer disabled vs enabled. That
   yields the EPS-vs-OCS number their paper lacks *and* tests our aggregate
   model against a packet-level engine — the single most valuable external
   validation available to us today.
3. **Use their axes as the α_r realism test we owe.** Their $R \in [0, 50]$ µs
   and $TSW$ range bracket our switch classes; the honest comparison scale for
   us is a *layer round* (10–100 µs), not a pass. Stating $k^* = 0$ against a
   MEMS-class 1–10 ms $\alpha_r$ is defensible; against a SOA-class µs-class
   switch it needs the DP output, which is already computed.
4. **Adopt soft preemption as the boundary rule, and cite it.** If the
   OCS-feasibility section ever describes a runtime, "release at collective
   boundaries, in-flight collectives complete" is the correct and citable
   policy — better than an unspecified preemption model.

## 4. What to cite them for (and what not to)

- **Cite:** the TSW/R trade-off (Fig. 3, Fig. 5a); the permutation constraint
  and $DM \le \sum_k M[k]$ (Eq. 1–2); "a single configuration cannot cover a
  dense demand matrix ⇒ in-collective reconfiguration" (Fig. 3d–f); soft
  preemption at collective boundaries (§3.2); the observation that #OCSes
  (optical parallelism) is worth more than link bandwidth in a
  contention-bound multi-job fabric (Fig. 6b).
- **Do not cite:** any "OCS is faster" number — there is no EPS-only baseline
  in the paper, and no port-count sweep, so its gains are scheduling deltas
  inside one fabric.
- **Metadata:** the ACM reference block and the Zotero record both have empty
  author/venue fields; the record must be completed before this enters the
  bibliography.

## 5. Limits of this comment

Read at the abstract/section level plus figures and Table 1; not verified by
running their simulator (unreleased). Fig. 4's per-$N$ values are read from the
plotted labels (≈100 µs → 34 µs for $N = 1 \to 8$, non-monotone in between) and
should be re-read from the PDF before being quoted in a paper.
