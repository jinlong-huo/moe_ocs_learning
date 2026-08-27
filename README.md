# MoE + OCS Research Testbed

Simulate **various real MoE models** — their actual expert weights on ranks, their actual captured routing — and then **verify OCS given the recorded affinity**, on an authentic 3-tier electrical fabric (EPS) with the α-β cost model.

```
models (real & various) ──▶ affinity (recorded from captures) ──▶ experts on ranks ──▶ OCS verification
```

---

## ⚠️ Start here: the staged evidence chain (`src/eval/`, `docs/research_assessment.md`)

The pipeline below (`α-β` cost model, pooled affinity, `Phase 1` gate) has been
**superseded** for research claims. A measured re-evaluation on 199 real traces
across two models found that several of its metrics are saturated by
construction and that its OCS cost model cannot show a benefit. Read
**[`docs/research_assessment.md`](docs/research_assessment.md)** before building
on anything in this README.

Headline corrections, all measured:

| finding | consequence |
|---|---|
| Per-layer expert-load vectors are uncorrelated (Pearson r = 0.004–0.011) | Layer-**pooled** expert statistics average independent namespaces and saturate. `load_entropy_norm`, `top5_expert_share`, `layer_diversity_mean_js`, `affinity_strength_offdiag` and pooled `js_divergence` are not usable. |
| The rank×rank all-to-all traffic matrix is **99.9 % rank-1** | Expert co-activation does **not** shape the traffic matrix. The exploitable mechanism is destination coalescing (fan-out), not traffic-matrix topology engineering. |
| Naive per-layer affinity placement is **2× worse than random** on Qwen3.6 | It cuts volume 33 % but triples ingress imbalance. Layers must be balanced **jointly**; `affinity_coordinated_layer` gets **+32.8 %** out-of-sample. |
| `α_ocs = α_eps + T_reconfig`, `β_ocs = β_eps` | Makes a circuit strictly slower than electrical — OCS could never win. Replaced by tier promotion (a circuit removes oversubscription). |
| At realistic pod sizes (256 GPU/pod) no EP degree ≤ 256 produces cross-pod traffic | There is nothing for OCS to promote; `ocs_eval` reports `applicable: False` instead of a number. |
| Bandwidth fields named `_gbps`, documented Gb/s, divided as GB/s | Inter-node tiers were modelled 8× too fast (`src/comm/topology.py`, duplicated in `runtime/placement.py`, `runtime/worker.py`). |

### Reproduce the chain

```bash
# 1. capture routing over a 112-sequence factorial workload suite (MLX; no vLLM needed)
python3 scripts/capture_workload.py \
    --model models/Qwen1.5-MoE-A2.7B-Chat-4bit \
    --out logs/workload/qwen15 --max-tokens 96                       # ~5.5 min

python3 scripts/capture_workload.py \
    --model models/Qwen3.6-35B-A3B-4bit \
    --out logs/workload/qwen36 --max-tokens 64 \
    --per-category 4 --n-repeats 3                                   # ~6.5 min

# 2. Q1..Q5 evidence chain (each stage may legitimately FAIL and say why)
python3 scripts/verify_live_invariance.py \
    --workload logs/workload/qwen36 --world-size 32 \
    --topology single_pod --topology multi_pod --topology realistic

# 3. figures
python3 scripts/make_figures.py --workload logs/workload/qwen36
```

| stage | question | Qwen1.5 (E=60,K=4) | Qwen3.6 (E=256,K=8) |
|---|---|---|---|
| Q1 | routing decoupled from placement/topology? | PASS, gate bit-exact | PASS, gate bit-exact |
| Q2 | does routing carry workload structure? | PASS — 62.5 % category decoding (null 6.4 %) | PASS — **93.75 %** (null 6.0 %) |
| Q3 | does placement change cost of fixed routing? | PASS — 3.7 % spread | PASS — 13.3 % spread |
| Q4 | does affinity beat random / load-balancing OOS? | PASS — +18.8 % | PASS — **+32.8 %** |
| Q5 | can OCS help after reconfiguration? | **FAIL** — no cross-pod traffic at EP=15 | conditional — 8.9 % under a small-pod assumption; plan Jaccard 0.09 |

### New modules

| module | role |
|---|---|
| `src/serving/suite.py` | 112-prompt factorial workload: 12 categories, paraphrase sets (same meaning / different words), lexical controls (same words / different meaning), length ladders, identical-prompt repeats for the **noise floor** |
| `scripts/capture_workload.py` | loads the model once, captures a trace per prompt + a manifest that is the design matrix |
| `src/eval/trace_ir.py` | `CellTable` — the immutable canonical routing IR; per-layer expert namespaces are first-class |
| `src/eval/affinity_graph.py` | 6 affinity definitions + a **load-preserving null** (popular experts co-occur because they are popular) |
| `src/eval/specialization.py` | category decoding and MI with **run-level** permutation nulls (cell-level nulls inflate n by ~1000×) |
| `src/eval/cost_model.py` | GPU→node→pod hierarchy, 3 dispatch modes, per-pair byte matrix, per-rank egress/ingress **bottleneck** |
| `src/eval/placement_opt.py` | 11 placement generators incl. the coordinated per-layer optimiser and a bitset `IngressOracle` |
| `src/eval/ocs_eval.py` | degree-bounded circuit planning, reconfiguration break-even, temporal stability at 3 timescales |

The legacy pipeline documented below is retained for the data-plane
(`src/comm/all_to_all.py` moves real tensors through real Qwen experts) and for
reproducing prior results.

---

## Models

| Model             | Experts | top-k | Role                                                   |
| ----------------- | ------- | ----- | ------------------------------------------------------ |
| Qwen3.6-35B-A3B   | 256     | 8     | primary: canonical traces, exported weights            |
| Qwen1.5-MoE-A2.7B | 60      | 4     | hardware invariance (Phase 2), model control (Phase 3) |
| Hy3               | —      | —    | additional real-model capture                          |
| Whittle (Qwen3.8) | —      | —    | additional real-model capture                          |

**Capture** (vLLM primary, MLX secondary; `--temp 0` = deterministic greedy; every backend validates the trace before saving):

```bash
# vLLM (CUDA)
python scripts/run_vllm.py run --model Qwen/Qwen3.6-35B-A3B \
    --prompt "Explain MoE routing." --max-tokens 128 --temp 0

# vLLM + Metal (Apple Silicon; the script pins VLLM_HOST_IP to loopback itself)
source ~/.venv-vllm-metal/bin/activate
python scripts/run_vllm.py run --model ./models/Qwen3.6-35B-A3B-4bit --max-tokens 256 --temp 0
python scripts/run_vllm.py run --model ./models/Qwen3.8-Whittle-MoE-27B-A17.8B-4bit --max-tokens 256 --temp 0
python scripts/run_vllm.py run --model ./models/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit --max-tokens 256 --temp 0

# MLX (secondary)
.venv/bin/python moe_run.py --model models/Qwen3.6-35B-A3B-4bit --max-tokens 128 --temp 0
.venv/bin/python moe_run.py --model models/Qwen3.8-Whittle-MoE-27B-A17.8B-4bit --max-tokens 128 --temp 0
.venv/bin/python moe_run.py --model models/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit --max-tokens 128 --temp 0

# multi-tenant serving
PY=~/.venv-vllm-metal/bin/python
$PY scripts/vllm_serve.py run --tenants 4 --schedule burst --greedy --max-tokens 32
$PY scripts/vllm_serve.py analyze logs/multi_tenant/run_burst_4t --plot      # contention/TTFT
$PY scripts/vllm_serve.py affinity logs/multi_tenant/run_burst_4t --plot     # prompt families

# additional models
python3 scripts/download_external_models.py --target hy3 --mirror      # or whittle: add models
python3 moe_run.py --model ./models/Hy3-oQ2 --max-tokens 32 --temp 0 --log-dir logs/phase4/hy3
```

*One-time prep for on-rank simulation:* `python3 scripts/export_qwen_experts.py --model models/Qwen3.6-35B-A3B-4bit --output exported_qwen_weights --max-layers 1`

## How a routing trace is captured

![routing capture flow](routing_capture_flow.png)

One pipeline, three phases. Entry points on the left spine, state on the right:

```text
verify_live_invariance.py            # Phase-1 matrix: per model → fresh subprocess
└─ scripts/run_vllm.py run           # single capture (also runnable directly)
   └─ src/data/live_capture.py::capture_live()     # vLLM-metal backend
      ├─ vllm.LLM(...) + locate_model(llm)         # load the MLX model
      ├─ VllmRoutingCapture()                      # _routes = {}
      ├─ install_vllm_metal_hooks(loaded, capture) # tag + patch every MoE gate
      ├─ get_vllm_layout(llm)                      # num_layers / experts / top_k
      └─ llm.chat(SamplingParams(max_tokens=…))    # generation runs
```

1. **Setup (before generation)** — load the model, locate it inside the engine, create an empty `VllmRoutingCapture`, patch every MoE block's gate, read the layout (`num_hidden_layers`, `num_experts`, `top_k`) from config.
2. **Capture (during generation)** — each patched MoE block computes `gate(x) → softmax → top-k`, then calls `capture.log(layer, positions, experts, weights)` — the single record point in `src/data/vllm_capture.py` — which writes `_routes[pos]["layers"][layer] = {"experts", "weights"}` (plus an `_expert_load` histogram). Prefill routes all prompt positions in one pass; decode routes one position per step.
3. **Assemble (after generation)** — `capture.build_trace(prompt_tokens, generated_tokens, tokenizer, **layout)` turns the dict into one `TokenRoute` per position (`token_pos/id/str`, `phase`, `layers: {idx → LayerRoute}`) plus `RunMeta`, wrapped in a validated `RoutingTrace` saved as JSON.

**Reading the numbers:** `len(routes)` = token positions routed (`prompt_len + generated_len`, minus the final token on some backends — don't assume equality with `total_tokens`); `len(route.layers)` = the model's MoE layer count (`num_hidden_layers`, e.g. 40 for Qwen3.6-35B-A3B), a model constant.

## Experts on ranks

Each rank loads the **real SwitchGLU expert weights** of the experts it owns
(`world_size × experts_per_rank` partition of the model), plus the real gate;
replay mode swaps in the captured routing so the testbed replays exactly
what the real model did:

```bash
python3 -m src.launcher --config configs/qwen_ocs_lite.yaml      # 8 experts, fast
python3 -m src.launcher --config configs/qwen_ocs_pipeline.yaml  # 32 experts
python3 -m src.launcher --config configs/ocs_affinity_placement.yaml  # 256 experts, 32 ranks
```

Placement decides *where* experts live — `placement.strategy`: `linear` (default `e//k`), `shuffle`, `affinity` (co-activation clustering), or `permutation`; `placement.rank_locations` pins ranks to physical spots. Routing never reads placement.

## Affinity

Recorded from the captured routing (expert co-activation per token/layer).
Verification gates prove it is trustworthy before OCS uses it — routing is
a pure function of (input, weights): topology, placement, and engine never
change *which expert* a token hits, only the cost.

**Phase 1 is a LIVE one-variable-at-a-time matrix** — the gate runs the
models itself (vLLM, greedy) at call time and captures routing in real
time; no pre-recorded traces. One fixed baseline (one model, one prompt,
one rank-node projection), then change exactly one knob and observe:

| Vary (keep the rest fixed)     | `token → expert` (routing)               | `token → rank` / cost                                                                     |
| ------------------------------ | ------------------------------------------- | -------------------------------------------------------------------------------------------- |
| topology (3 named fabrics)     | **identical** (bit-for-bit, computed) | delays move by tier                                                                          |
| placement (linear vs shuffled) | **identical** (computed)              | **relabeled** — same experts, different owning ranks, each bound to its rank+location |
| prompt                         | **changes** (new affinity graph)      | changes accordingly                                                                          |
| model                          | **changes everywhere**                | changes accordingly                                                                          |

**Topology is varied across three named 3-tier fabrics**, one per link tier,
so the invariance is demonstrated over the exact tiers OCS must span:

| Fabric          | Shape (pods × nodes × rpn) | Max tier   |
| --------------- | ---------------------------- | ---------- |
| `within-rack` | `1 × 1 × world`          | INTRA_NODE |
| `in-pod`      | `1 × N × rpn`            | INTRA_POD  |
| `cross-pod`   | `2 × N × rpn`            | CROSS_POD  |

**Every trace now records its placement manifest** — `expert_to_rank`,
`rank_to_location` (`[pod, node, local_rank]`), and the topology that
produced them — so the token→expert ↔ rank↔location binding is explicit in
the recording, not assumed. `token_expert_identical` is **computed** by
bit-comparing the expert-id keys across all variants (never asserted `True`
by construction).

```bash
# Phase 1: LIVE matrix (needs the vllm-metal env — it runs real inference)
~/.venv-vllm-metal/bin/python scripts/verify_live_invariance.py            # all models in order
~/.venv-vllm-metal/bin/python scripts/verify_live_invariance.py \
    --model models/Qwen3.8-Whittle-MoE-27B-A17.8B-4bit                     # one model
python3 scripts/compare_backend_traces.py \
    --a logs/phase2/mlx/routing.json \
    --b logs/phase2/run_uniform_1t/traces/tenant-000.json   # Phase 2: MLX vs vLLM-metal
python3 scripts/compare_model_affinity.py \
    --small logs/phase2/mlx/routing.json \
    --large logs/phase3/large/routing.json                  # Phase 3: 60e vs 256e models
```

The live gate also refreshes the canonical replay trace
(`data/routing_traces/routing.json` + a model-stamped copy) with the
baseline capture and its **placement manifest** (expert→rank +
rank→`[pod, node, local_rank]` + topology). `run_vllm.py run` likewise
writes model-stamped files (`logs/routing_vllm_<model>.json`) with a
default linear/flat placement, so captures from different models never
overwrite each other and every trace is self-documenting about placement.

| Phase | Varied                                       | Result                                                                                                    | Verdict                                     |
| ----- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 1     | topology (3 fabrics) + expert/rank placement | token→expert bit-identical (computed); token→rank relabels (bound to rank+location); cost moves by tier | topology/placement-independent ✓           |
| 2     | engine (MLX vs vLLM-metal)                   | prefill overlap 0.933, JS 1.1e-4, corr 0.998, hit-rate 1.0                                                | hardware-independent (up to noise floor) ✓ |
| 3     | model (60e vs 256e)                          | top-5 share 0.101→0.043, layer-JS 0.123→0.677                                                           | presets are model-specific ✓               |

So **affinity = f(input, weights)** and can safely drive OCS configuration.
Full ledger (claim → gate → report → open items): [docs/assumptions.md](docs/assumptions.md).

## OCS verification — the α-β time model

All communication cost follows the classic network α-β model, `T(n) = α + β·n`
(α = fixed latency µs, β = inverse bandwidth µs/byte). **EPS** pays the
3-tier fabric α/β:

| Tier       | Link            | α (latency) | β (1/BW) |
| ---------- | --------------- | ------------ | --------- |
| INTRA_NODE | NVLink/NVSwitch | ~1 µs       | 900 GB/s  |
| INTRA_POD  | InfiniBand NDR  | ~3 µs       | 400 Gb/s  |
| CROSS_POD  | core fabric     | ~10 µs      | 200 Gb/s  |

**OCS adds the fixed reconfiguration delay to α** on a cold circuit —
exactly the "certain delay on top of EPS" the literature uses:

```
α_ocs = α_eps + T_reconfig      β_ocs = β_eps
```

under a **per-rank circuit budget** (`max_circuits` = ports/wavelengths,
FIFO port reassignment when exhausted — a port-limited switch serially
re-points per destination, the authentic OCS cost on all-to-all traffic).
Two parameterizations of `T_reconfig` (SOA/ring ns–µs,
[Sirius](https://dlnext.acm.org/doi/epdf/10.1145/3387514.3406221);
3D-MEMS tens of µs + damping):

| model | switch                       | `T_reconfig` | budget        |
| ----- | ---------------------------- | -------------- | ------------- |
| alpha | SOA/ring (fast, WSS fan-out) | 1 µs          | world_size−1 |
| beta  | 3D-MEMS (port-limited)       | 50 µs         | 1             |

```bash
python3 -m src.launcher --config configs/qwen_eps_baseline.yaml   # EPS
python3 -m src.launcher --config configs/ocs_alpha_model.yaml     # OCS alpha
python3 -m src.launcher --config configs/ocs_beta_model.yaml      # OCS beta
python3 scripts/compare_ocs_models.py    # all three on the same fabric + table
```

Reference run (real Qwen, 2×2×1 fabric): EPS 22.1 ms comm; alpha +3.4 ms
(3 µs reconfig total); beta +2.3 ms (**3000 µs reconfig**, 60 switches, 59
port reassignments — the single MEMS port re-points per target for both
dispatch and gather).

**Scheduling modes** (how/when the reconfig is paid):

| Mode             | Reconfig                                          | Use case                |
| ---------------- | ------------------------------------------------- | ----------------------- |
| `ocs_pipeline` | inline, before each scatter                       | runtime adaptability    |
| `ocs_dbo`      | hidden behind previous batch's compute            | mask reconfig latency   |
| `ocs_preset`   | **none during inference** (plan pre-loaded) | affinity → pre-config  |
| `ocs_online`   | adaptive, from live co-activation (decay 0.99)    | inference self-learning |

```bash
bash scripts/run_preset_pipeline.sh data/routing_traces/routing.json   # trace → plan → EPS/OCS/preset → compare
```

**Placement payoff** (affinity applied): `placement.strategy: affinity`
raises intra-rank affinity 0.026 → 0.102; `placement.rank_locations`
(plan-centrality packing) cuts the top-16 plan's cross-pod pairs 5 → 2.

## Project structure

```
src/
├── model/      Qwen expert loader + gate, captured-routing replay router
├── comm/       all-to-all, transport, 3-tier topology, timeline export
├── ocs/        α-β circuit model, affinity placement, preconfig, online controller
├── runtime/    worker, schedulers, placement tables, process groups
├── data/       routing schema, MLX/vLLM capture, interventions
├── serving/    multi-tenant vLLM serving capture (engine, workload, affinity, analyze)
├── eval/       affinity consistency metrics
└── utils/      timer, logging, seed
configs/        qwen replay/OCS configs, EPS baseline, alpha/beta, affinity placement
scripts/        capture, verification gates (Phase 1–3), EPS-vs-OCS comparison, presets
docs/           assumption ledger, research discussions, architecture alignment
data/           reference routing traces (vLLM-captured) + fine-tuning dataset
logs/           Phase 1–4 verification reports + captures
models/         MLX model weights
adapters/       LoRA adapters
```

## Key design decisions

- **Real and various models** — Qwen3.6-35B (primary), Qwen1.5-MoE-A2.7B, Hy3, Whittle; real weights on ranks, real captured routing; `RoutingTrace` is the single source of truth across all backends
- **Placement is a cost-side variable** — routing never reads placement; only dispatch and the topology delay model do
- **OCS = α-β with a fixed α-adder** — `α_ocs = α_eps + T_reconfig`, circuit-budget constrained, honestly paid on the cold path
- **Metadata packing** — `local_expert_id` + `original_index` ride as float columns through `all_to_all_single` (zero extra comm rounds); async scatter/gather in NCCL CUDA-stream pattern

## License

MIT
