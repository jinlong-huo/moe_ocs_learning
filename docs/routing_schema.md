# Routing Schema — Capture Pipeline and Data Structures

This document maps how a routing decision made inside a real MoE forward pass
becomes a validated, self-describing trace on disk, and then the intermediate
representation every downstream stage consumes.

The one-line dataflow:

```
model (real) ─▶ capture patch (per backend) ─▶ RoutingTrace JSON ─▶ validate
            ─▶ workload manifest ─▶ CellTable IR ─▶ affinity / cost / placement / OCS
```

---

## 1. Capture pipeline

Four capture paths exist because the hook point must live wherever the
*batching owner* lives. All four run the same routing math
(`gate(x) → softmax → top-k → (ids, weights)`) and emit the same canonical
`RoutingTrace`; only the token-position attribution differs.

```mermaid
flowchart TB
    subgraph MODEL["MoE model (fixed weights, e.g. Qwen-MoE 4-bit)"]
        L["decoder layers: layer.mlp — dense MLP *or* sparse MoE block"]
    end

    subgraph MLXPATH["MLX direct — scripts/capture_workload.py, scripts/moe_run.py"]
        M1["patch MLX MoE block class __call__<br/>(re-implements forward: logged == executed,<br/>top-k sorted so experts[0] = argmax)"]
        M1P["positions: shared mutable state dict<br/>(seq_pos / phase) owned by OUR generation loop"]
    end

    subgraph VLLMTORCH["vLLM torch — src/data/vllm_capture.py"]
        V1["patch each MoE block's gate.forward (per instance)<br/>+ decoder-layer forward publishes positions<br/>also supports router steering (force / bias / ablate)"]
        V1P["positions: the engine's own positions tensor,<br/>read through a contextvar"]
    end

    subgraph VLLMMETAL["vLLM-metal — install_vllm_metal_hooks"]
        T1["patch MLX MoE block __call__<br/>(engine is vLLM, model is MLX)"]
        T1P["positions: monotonic _METAL_OFFSET counter<br/>⚠ assumes single request, in-order,<br/>no prefix-cache reuse"]
    end

    subgraph MTENGINE["vllm-metal engine, multi-tenant — src/serving/capture.py"]
        E1["patch MLX MoE block __call__ (log-then-delegate)<br/>+ Metal-runner wrappers (_start_paged_forward, ...)"]
        E1P["positions: exact per-forward token map _TOKEN_MAP<br/>(decode-first ordering, spec-decode widening)<br/>— the only exact attribution under continuous batching"]
    end

    L --> M1 & V1 & T1 & E1
    M1P -.-> M1
    V1P -.-> V1
    T1P -.-> T1
    E1P -.-> E1

    M1 & V1 & T1 & E1 --> RT["RoutingTrace (canonical JSON)<br/>src/data/routing_schema.py"]
    RT --> VAL["RoutingTrace.validate()<br/>bounds: expert &lt; num_experts, top-k arity,<br/>weight range, position monotonicity, placement consistency"]
    VAL -->|pass| MAN["workload manifest.json — the design matrix<br/>(uid / category / group / role / variant + trace paths)<br/>scripts/capture_workload.py"]
    VAL -->|fail| DEAD["ValueError — poisoned trace never reaches analysis"]
    MAN --> CT["CellTable IR — src/eval/trace_ir.py<br/>(run, layer, pos, tok, phase, experts [N,K], weights [N,K])"]
    CT --> CONSUMERS["downstream consumers (§3)"]
```

**Known calibrations / limitations of the capture layer**

* **Marginal-expert noise**: on the Metal backend the k-th (marginal) expert
  flips on near-ties under identical inputs (~5–6 % of cells, quantized-GEMM
  noise). Every comparison metric therefore also has a top-(k−1) variant
  (`k_compare`), and weight-aware metrics price these flips by gate mass
  rather than as total misses.
* **Order within top-k**: `argpartition` is unordered. Only
  `capture_workload.py` sorts the top-k by descending score, so only its
  traces guarantee `experts[:,0]` is the argmax (the invariant documented on
  `CellTable`).
* **Log-vs-execute divergence (deferred fix)**: `src/serving/capture.py`
  logs a *recomputed* gate and then delegates to the original `__call__`,
  which computes the gate a second time. Under the Metal GEMM's
  non-bit-reproducibility the logged decision can in principle diverge from
  the executed one on near-ties. The `capture_workload.py` patch style
  (re-implement the forward) avoids this at the cost of duplicating model
  code.
* **MoE-block identification** is duck-typing, not counting: a decoder layer
  is MoE iff its `mlp` exposes `switch_mlp` (plus `gate`,
  `shared_expert(_gate)` / `shared_mlp`). Dense layers carry a plain MLP.

---

## 2. Data structures

```mermaid
classDiagram
    direction LR

    class RunMeta {
        +str model_id
        +str model_type
        +int num_layers
        +int num_moe_layers
        +int num_experts
        +int top_k
        +int prompt_len
        +int generated_len
        +int total_tokens
        +str backend
        +str run_id
        recorded once per inference run;
        every validate() bound is
        expressed against these fields
    }

    class LayerRoute {
        +list~int~ experts
        +list~float~ weights
        raw top-k softmax mass
        (renormalized only if the
        block sets norm_topk_prob)
    }

    class TokenRoute {
        +int token_pos
        +int token_id
        +str token_str
        +str phase
        +dict~str,LayerRoute~ layers
    }

    class RoutingTrace {
        +RunMeta meta
        +list~int~ prompt_tokens
        +list~int~ generated_tokens
        +list~TokenRoute~ routes
        +list~list~float~~ guide_affinity?
        +dict placement?
        +save() / load()
        +validate()
        +expert_load() / per_layer_expert_load()
        +rank_communication_matrix()  %% legacy, deferred
        +per_layer_rank_targets()      %% legacy, deferred
    }

    class PlacementManifest {
        expert_to_rank
        rank_to_location
        world_size
        topology
        cost-side projection only;
        validated, never fed back
        into routing
    }

    class RunInfo {
        +int run_idx
        +str uid
        +str category / group / role / variant
        +str model_id
        +int prompt_len / generated_len / total_tokens
    }

    class CellTable {
        +int32[] run / layer / pos / tok / phase
        +int32[,] experts  [n_cells, K]
        +float32[,] weights [n_cells, K]
        +list~RunInfo~ runs
        +int num_experts / top_k
        +layers
        +select() / by_runs() / by_role() / by_layer()
        +expert_load() / per_layer_load()
        +load_balance() / layer_signature()
    }

    RoutingTrace *-- RunMeta
    RoutingTrace *-- TokenRoute
    TokenRoute *-- LayerRoute
    RoutingTrace o-- PlacementManifest
    CellTable o-- RunInfo
    RoutingTrace ..> CellTable : load_workload() / load_single_trace()<br/>(trace_ir.py)
```

**Indexing rule**: routes are indexed by *absolute token position* (not
forward-pass step), so cross-token and cross-layer analysis is trivial and
batched captures remain order-independent.

**meta vs manifest** (three distinct objects):

| object | scope | purpose |
| --- | --- | --- |
| `RunMeta` | one inference run | self-describing trace; the bounds `validate()` checks against; downstream asserts (`compare_backend_traces.py`) rely on it |
| workload `manifest.json` | whole capture session | the experimental *design matrix* (labels + trace paths) — makes every downstream number reproducible without re-running inference; consumed by `load_workload()` |
| placement manifest (`trace.placement`) | one trace | the *cost-side projection* (expert→rank, rank→location, topology). Routing is the measurement, placement is a re-projectable hypothesis: one trace can be re-costed on many topologies |

---

## 3. Downstream consumers

```mermaid
flowchart TB
    CT["CellTable IR"]

    CT --> AG["affinity_graph.py — expert↔expert affinity within a workload<br/>cooccurrence / conditional / pmi / jaccard / cosine / weighted<br/>+ load-preserving null model & structure test<br/>+ graph-to-graph similarity, signature similarity"]
    CT --> AC["affinity_consistency.py — train vs inference routing correlation"]
    CT --> CM["cost_model.py — fan-out, dedup ingress skew<br/>(the quantities that govern all-to-all completion)"]
    CT --> PO["placement_opt.py — expert placement optimisation<br/>(per-layer, jointly balanced — see README result)"]
    CT --> PC["preconfig.py — OCS circuit plans from co-activation"]

    RT2["RoutingTrace (pair of traces)"]
    RT2 --> PA["serving/affinity.py pairwise_metrics — trace↔trace similarity<br/>set-level: topk_overlap, jaccard, plan_hit_rate, JS, co-act corr<br/>weight-aware: mass intersection, EMD, Bhattacharyya,<br/>matched-cell weight MAE / cosine<br/>+ repeat_noise_floor() calibrated z-scores"]
    PA --> INV["compare_backend_traces.py — cross-backend<br/>routing-invariance contract (MLX vs vLLM-metal)"]
    PA --> SESS["affinity_report() — multi-tenant session analysis"]

    LEGACY["LEGACY (deferred): RoutingTrace.rank_communication_matrix,<br/>per_layer_rank_targets — dead code; if revived must drop the<br/>diagonal, count unordered pairs once, use the placement manifest<br/>instead of expert//experts_per_rank, and weight by gate mass"]
    CT -.-> LEGACY
```

**Why the legacy methods are deferred**: `rank_communication_matrix` /
`per_layer_rank_targets` are never called; their semantics predate the
placement work (contiguous-placement assumption, ordered-pair double
counting, intra-rank diagonal counted as "communication"), and the cost
analysis that the project actually publishes flows through
`cost_model.py`'s fan-out and dedup-ingress-skew instead. Reviving them has
no effect on the affinity work and is intentionally left for later.

**The shared-expert path is not routing.** `y = Σ score_k·Expert_k(x) +
sigmoid(shared_expert_gate(x)) · shared_expert(x)` — the shared expert is an
always-on dense MLP, gated per token. It never enters the routing trace; it
matters only for cost accounting (per-token local compute that never leaves
the rank).
