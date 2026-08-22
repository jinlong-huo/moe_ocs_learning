# MoE + OCS Research Testbed

Simulate **various real MoE models** — their actual expert weights on ranks, their actual captured routing — and then **verify OCS given the recorded affinity**, on an authentic 3-tier electrical fabric (EPS) with the field-standard α-β cost model.

```
models (real & various) ──▶ affinity (recorded from captures) ──▶ experts on ranks ──▶ OCS verification
```

## Models

| Model | Experts | top-k | Role |
| ----- | ------- | ----- | ---- |
| Qwen3.6-35B-A3B | 256 | 8 | primary: canonical traces, exported weights |
| Qwen1.5-MoE-A2.7B | 60 | 4 | hardware invariance (Phase 2), model control (Phase 3) |
| Hy3 | — | — | additional real-model capture |
| Whittle (Qwen3.8) | — | — | additional real-model capture |

**Capture** (vLLM primary, MLX secondary; `--temp 0` = deterministic greedy; every backend validates the trace before saving):

```bash
python scripts/run_vllm.py run --model Qwen/Qwen3.6-35B-A3B \
    --prompt "Explain MoE routing." --max-tokens 128 --temp 0          # vLLM (CUDA)

source ~/.venv-vllm-metal/bin/activate                                 # vLLM + Metal (Apple Silicon;
python scripts/run_vllm.py run \                                       #   script pins VLLM_HOST_IP itself)
    --model ./models/Qwen3.6-35B-A3B-4bit --max-tokens 256 --temp 0

.venv/bin/python moe_run.py --model models/Qwen3.6-35B-A3B-4bit --max-tokens 128 --temp 0   # MLX

PY=~/.venv-vllm-metal/bin/python                                       # multi-tenant serving
$PY scripts/vllm_serve.py run --tenants 4 --schedule burst --greedy --max-tokens 32
$PY scripts/vllm_serve.py analyze logs/multi_tenant/run_burst_4t --plot      # contention/TTFT
$PY scripts/vllm_serve.py affinity logs/multi_tenant/run_burst_4t --plot     # prompt families

python3 scripts/download_external_models.py --target hy3 --mirror      # or whittle: add models
python3 moe_run.py --model ./models/Hy3-oQ2 --max-tokens 32 --temp 0 --log-dir logs/phase4/hy3
```

*One-time prep for on-rank simulation:* `python3 scripts/export_qwen_experts.py --model models/Qwen3.6-35B-A3B-4bit --output exported_qwen_weights --max-layers 1`

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

Placement decides *where* experts live — `placement.strategy`: `linear`
(default `e//k`), `shuffle`, `affinity` (co-activation clustering), or
`permutation`; `placement.rank_locations` pins ranks to physical spots.
Routing never reads placement.

## Affinity

Recorded from the captured routing (expert co-activation per token/layer).
Verification gates prove it is trustworthy before OCS uses it — routing is
a pure function of (input, weights); topology, placement, and engine never
change *which* expert a token hits, only the cost:

```bash
python3 scripts/verify_ocs_invariance.py        # Phase 1: topology + placement invariance + payoff
python3 scripts/compare_backend_traces.py \
    --a logs/phase2/mlx/routing.json \
    --b logs/phase2/run_uniform_1t/traces/tenant-000.json   # Phase 2: MLX vs vLLM-metal
python3 scripts/compare_model_affinity.py \
    --small logs/phase2/mlx/routing.json \
    --large logs/phase3/large/routing.json                  # Phase 3: 60e vs 256e models
```

| Phase | Varied | Result | Verdict |
| ----- | ------ | ------ | ------- |
| 1 | topology config + expert/rank placement | affinity + plan bit-identical; only cost moves | topology/placement-independent ✓ |
| 2 | engine (MLX vs vLLM-metal) | prefill overlap 0.933, JS 1.1e-4, corr 0.998, hit-rate 1.0 | hardware-independent (up to noise floor) ✓ |
| 3 | model (60e vs 256e) | top-5 share 0.101→0.043, layer-JS 0.123→0.677 | presets are model-specific ✓ |

So **affinity = f(input, weights)** and can safely drive OCS configuration.
Full ledger (claim → gate → report → open items): [docs/assumptions.md](docs/assumptions.md).

## OCS verification — the α-β time model

All communication cost follows the classic network α-β model, `T(n) = α + β·n`
(α = fixed latency µs, β = inverse bandwidth µs/byte). **EPS** pays the
3-tier fabric α/β:

| Tier | Link | α (latency) | β (1/BW) |
| ---- | ---- | ----------- | -------- |
| INTRA_NODE | NVLink/NVSwitch | ~1 µs | 900 GB/s |
| INTRA_POD | InfiniBand NDR | ~3 µs | 400 Gb/s |
| CROSS_POD | core fabric | ~10 µs | 200 Gb/s |

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

| model | switch | `T_reconfig` | budget |
| ----- | ------ | ------------ | ------ |
| alpha | SOA/ring (fast, WSS fan-out) | 1 µs | world_size−1 |
| beta | 3D-MEMS (port-limited) | 50 µs | 1 |

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

| Mode | Reconfig | Use case |
| ---- | -------- | -------- |
| `ocs_pipeline` | inline, before each scatter | runtime adaptability |
| `ocs_dbo` | hidden behind previous batch's compute | mask reconfig latency |
| `ocs_preset` | **none during inference** (plan pre-loaded) | affinity → pre-config |
| `ocs_online` | adaptive, from live co-activation (decay 0.99) | inference self-learning |

```bash
bash scripts/run_preset_pipeline.sh data/routing_traces/routing.json   # trace → plan → EPS/OCS/preset → compare
```

**Placement payoff** (affinity applied): `placement.strategy: affinity`
raises intra-rank affinity 0.026 → 0.115; `placement.rank_locations`
(plan-centrality packing) cuts the top-16 plan's cross-pod pairs 3 → 0.

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
