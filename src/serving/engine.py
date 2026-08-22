"""
engine.py — Multi-tenant serving loop on the vllm-metal engine.

vllm-metal runs the real vLLM V1 scheduler with an MLX compute backend on
Apple Silicon.  The async vLLM engine requires a multiprocess core, which
the Metal backend cannot initialize in a child process, so this module
drives the *in-process* engine the way a serving dispatcher would:

    VLLM_ENABLE_V1_MULTIPROCESSING=0  →  engine core lives in this process
    LLMEngine.add_request(..., arrival_time=...)  →  requests enter the queue
    LLMEngine.step()                             →  scheduler batches a step

Requests arrive on a generated schedule (Poisson / periodic / burst /
uniform); every engine step processes whatever mix of tenants the V1
scheduler picked, so decode steps genuinely contend for the same experts
and KV cache — the multi-tenant case that a single-stream trace cannot
show.

Sequential mode (``--mode sequential``) runs the identical workload with
no overlap and serves as the zero-contention baseline for TTFT / ITL
comparison.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

# ── Environment (must be set before vLLM is imported) ──────────────

def configure_environment() -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_METAL_DECODE_PIPELINE", "0")
    os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")
    # macOS loopback: pin the host IP — the hostname may resolve to an
    # unreachable LAN IP (e.g. 10.23.0.1) and crash PyTorch's TCPStore.
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "VLLM_LOGGING_LEVEL" not in os.environ:
        os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"


@dataclass
class ServeConfig:
    """Options for one serving session."""

    model: str
    max_model_len: int = 4096
    enforce_eager: bool = True
    trust_remote_code: bool = True
    max_tokens: int = 128
    temperature: float = 0.6
    seed: int = 0
    num_tenants: int = 4
    schedule: str = "poisson"
    rate: float = 1.0
    mode: str = "concurrent"  # "concurrent" | "sequential"
    prompts_file: str | Path | None = None
    no_chat: bool = False
    system_prompt: str = "You are a helpful assistant."
    output_dir: str | Path = "logs/multi_tenant"
    # Experiment-design controls
    family: str = "mixed"  # "identical" | "similar" | "mixed"
    base_prompt: str | None = None
    slot_step: int = 1
    greedy: bool = False
    seed_mode: str = "same"  # "same" | "distinct"
    prefix_caching: bool | None = None  # None → engine default


# ═══════════════════════════════════════════════════════════════════
# Engine bootstrap + layout
# ═══════════════════════════════════════════════════════════════════

def build_llm(cfg: ServeConfig):
    """Load the model through vLLM + vllm-metal (in-process engine core)."""
    configure_environment()

    from vllm import LLM
    import inspect

    kwargs = dict(
        model=cfg.model,
        max_model_len=cfg.max_model_len,
        enforce_eager=cfg.enforce_eager,
        trust_remote_code=cfg.trust_remote_code,
        seed=cfg.seed,
    )
    try:
        sig_params = inspect.signature(LLM.__init__).parameters
        for opt in ("disable_log_stats", "enable_log_requests"):
            if opt in sig_params:
                kwargs[opt] = True if opt == "disable_log_stats" else False
    except (TypeError, ValueError):
        pass
    # ``enable_prefix_caching`` is not an explicit LLM.__init__ param, but
    # LLM forwards **kwargs to EngineArgs, which accepts it.
    if cfg.prefix_caching is not None:
        kwargs["enable_prefix_caching"] = cfg.prefix_caching

    print(f"[engine] Loading {cfg.model} via vllm-metal "
          f"(VLLM_ENABLE_V1_MULTIPROCESSING=0)")
    t0 = time.time()
    llm = LLM(**kwargs)
    print(f"[engine] Loaded in {time.time() - t0:.1f}s")
    return llm


def get_vllm_layout(llm) -> dict:
    """Extract model metadata for the routing traces."""
    hf_cfg = llm.llm_engine.vllm_config.model_config.hf_config
    text_cfg = getattr(hf_cfg, "text_config", None)

    def _cfg(attr, default=0):
        v = getattr(hf_cfg, attr, None)
        if v in (None, 0) and text_cfg is not None:
            v = getattr(text_cfg, attr, None)
        return v if v is not None else default

    num_experts = int(_cfg("num_experts") or _cfg("num_local_experts") or 0)
    top_k = int(_cfg("num_experts_per_tok") or _cfg("top_k") or 0)
    return {
        "model_type": _cfg("model_type", "unknown"),
        "num_layers": int(_cfg("num_hidden_layers")),
        "num_experts": num_experts,
        "top_k": top_k,
    }


def _get_eos_ids(llm) -> list[int] | None:
    hf_cfg = llm.llm_engine.vllm_config.model_config.hf_config
    eos = getattr(hf_cfg, "eos_token_id", None)
    if eos is None:
        return None
    if isinstance(eos, (list, tuple)):
        ids = [int(e) for e in eos if e is not None]
        return ids or None
    return [int(eos)]


def _build_sampling_params(cfg: ServeConfig, llm, extra_seed: int = 0):
    from vllm import SamplingParams

    temperature = 0.0 if cfg.greedy else cfg.temperature
    seed = cfg.seed
    if cfg.seed_mode == "distinct":
        seed = cfg.seed + extra_seed
    kwargs = dict(temperature=temperature, max_tokens=cfg.max_tokens, seed=seed)
    eos = _get_eos_ids(llm)
    if eos:
        kwargs["stop_token_ids"] = eos
    return SamplingParams(**kwargs)


def _render_prompt(cfg: ServeConfig, llm, prompt: str) -> str:
    """Render the chat template (matching the MLX/HF backends) unless disabled."""
    if cfg.no_chat:
        return prompt
    tokenizer = llm.get_tokenizer()
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": prompt},
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass
    return prompt


# ═══════════════════════════════════════════════════════════════════
# Serving loops
# ═══════════════════════════════════════════════════════════════════

@dataclass
class _RequestRec:
    request_id: str
    tenant_idx: int
    prompt: str
    arrival_s: float
    first_token_s: float | None = None
    finish_s: float | None = None
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    token_timestamps_s: list[float] = field(default_factory=list)


def _record_outputs(
    recs: dict[str, _RequestRec],
    outputs: list,
    now_s: float,
) -> None:
    """Ingest one engine step's request outputs into the per-request records."""
    for ro in outputs:
        rec = recs.get(ro.request_id)
        if rec is None:
            continue
        if ro.prompt_token_ids is not None:
            rec.prompt_token_ids = list(ro.prompt_token_ids)
        for out in ro.outputs:
            tokens = list(out.token_ids)
            new = tokens[len(rec.generated_token_ids):]
            if rec.first_token_s is None and tokens:
                rec.first_token_s = now_s
            for _ in new:
                rec.token_timestamps_s.append(now_s)
            rec.generated_token_ids = tokens
        if ro.finished and rec.finish_s is None: # represent the engine is done, either eos hit or max_tokens, got to next session json.
            rec.finish_s = now_s


def _build_session(
    cfg: ServeConfig,
    llm,
    capture,
    layout: dict,
    workload,
    recs: list[_RequestRec],
    t0: float,
    output_dir: Path,
):
    """Build per-tenant traces + the session record, and save both."""
    from src.serving.schema import (
        MultiTenantSession,
        SessionMeta,
        StepRecord,
        TenantSummary,
    )

    capture.finalize_steps()
    tokenizer = llm.get_tokenizer()
    model_id = str(cfg.model)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    tenants: list[TenantSummary] = []
    for rec in recs:
        trace = capture.build_tenant_trace(
            req_id=rec.request_id,
            prompt_tokens=rec.prompt_token_ids,
            generated_tokens=rec.generated_token_ids,
            tokenizer=tokenizer,
            model_id=model_id,
            layout=layout,
            backend="vllm",
        )
        trace_path = traces_dir / f"{rec.request_id}.json"
        trace.save(str(trace_path))

        itl = []
        ts = rec.token_timestamps_s
        for i in range(1, len(ts)):
            itl.append(ts[i] - ts[i - 1])
        slots_changed = (
            workload.slots_changed[rec.tenant_idx]
            if rec.tenant_idx < len(workload.slots_changed)
            else 0
        )
        tenants.append(TenantSummary(
            request_id=rec.request_id,
            tenant_idx=rec.tenant_idx,
            prompt=rec.prompt,
            prompt_len=len(rec.prompt_token_ids),
            generated_len=len(rec.generated_token_ids),
            arrival_s=rec.arrival_s,
            first_token_s=rec.first_token_s if rec.first_token_s is not None else rec.arrival_s,
            finish_s=rec.finish_s if rec.finish_s is not None else rec.arrival_s,
            ttft_s=(
                rec.first_token_s - rec.arrival_s
                if rec.first_token_s is not None
                else 0.0
            ),
            token_timestamps_s=ts,
            itl_s=itl,
            text="".join(
                _decode_token(tokenizer, [tid]) for tid in rec.generated_token_ids
            ),
            trace_path=str(trace_path.relative_to(output_dir)),
            slots_changed=slots_changed,
        ))

    steps = [
        StepRecord(step=s["step"], t_s=s["t_s"] - t0, duration_s=s["duration_s"], tokens=s["tokens"])
        for s in capture.steps
    ]

    meta = SessionMeta(
        model_id=model_id,
        model_type=layout.get("model_type", "unknown"),
        backend="vllm",
        num_layers=int(layout.get("num_layers", 0)),
        num_moe_layers=len(capture.moe_layer_indices),
        num_experts=int(layout.get("num_experts", 0)),
        top_k=int(layout.get("top_k", 0) or capture.top_k or 0),
        schedule=workload.schedule,
        rate=workload.rate,
        num_tenants=cfg.num_tenants,
        max_tokens=cfg.max_tokens,
        mode=cfg.mode,
        seed=cfg.seed,
        family=workload.family,
        temperature=0.0 if cfg.greedy else cfg.temperature,
        seed_mode=cfg.seed_mode,
        prefix_caching=cfg.prefix_caching if cfg.prefix_caching is not None else True,
    )
    session = MultiTenantSession(meta=meta, tenants=tenants, steps=steps)
    session.save(str(output_dir / "session.json"))
    return session


def _decode_token(tokenizer, ids: list[int]) -> str:
    try:
        return str(tokenizer.decode(ids))
    except Exception:
        return ""


def _warm_engine(engine) -> None:
    """Execute the engine's dummy-batch warmup step before any request.

    Keeps the first admitted tenant from paying the cold-start cost, so
    baseline vs concurrent comparisons reflect contention, not warmup.
    """
    if getattr(engine, "should_execute_dummy_batch", False):
        engine.step()


def run_concurrent_session(cfg: ServeConfig, llm, capture, layout, workload) -> Path:
    """Run all tenants through the engine with staggered arrivals."""
    engine = llm.llm_engine
    params_by_tenant = {
        idx: _build_sampling_params(cfg, llm, idx)
        for idx in range(len(workload.prompts))
    }
    prompts = [_render_prompt(cfg, llm, p) for p in workload.prompts]
    _warm_engine(engine)

    order = sorted(range(len(workload.arrivals_s)), key=lambda i: workload.arrivals_s[i])
    arrivals = [workload.arrivals_s[i] for i in order]
    ids = [workload.tenant_ids[i] for i in order]

    recs: dict[str, _RequestRec] = {}
    t0 = time.perf_counter()
    next_i = 0

    print(f"[serve] {cfg.num_tenants} tenants, schedule={cfg.schedule} "
          f"rate={cfg.rate} req/s, max_tokens={cfg.max_tokens}")

    while next_i < len(order) or engine.has_unfinished_requests():
        now = time.perf_counter() - t0

        # Admit all requests whose arrival time has passed.  The in-process
        # engine core is single-threaded, so admission happens at step
        # boundaries; the engine still gets the *true* arrival time for
        # FCFS queue ordering, and TTFT is measured from it.
        
        while next_i < len(order) and arrivals[next_i] <= now:
            req_id = ids[next_i]
            recs[req_id] = _RequestRec(
                request_id=req_id,
                tenant_idx=order[next_i],
                prompt=workload.prompts[order[next_i]],
                arrival_s=arrivals[next_i],  # true user arrival, not admission
            )
            backdated = time.time() - (now - arrivals[next_i])
            engine.add_request(
                req_id,
                prompts[order[next_i]],
                params_by_tenant[order[next_i]],
                arrival_time=backdated,
            )
            print(f"[serve] +{req_id} @ t={arrivals[next_i]:.3f}s "
                  f"(admitted {now:.3f}s, prompt {workload.prompts[order[next_i]][:50]!r})")
            next_i += 1
            now = time.perf_counter() - t0

        if engine.has_unfinished_requests():
            outputs = engine.step()
            _record_outputs(recs, outputs, now_s=time.perf_counter() - t0)
        else:
            # Idle gap between arrivals — wait for the next one.
            if next_i < len(order):
                wait = arrivals[next_i] - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(min(wait, 0.01))

    output_dir = Path(cfg.output_dir) / f"run_{workload.schedule}_{cfg.num_tenants}t"
    _build_session(cfg, llm, capture, layout, workload, list(recs.values()), t0, output_dir)
    return output_dir


def run_sequential_baseline(cfg: ServeConfig, llm, capture, layout, workload) -> Path:
    """Same workload, one tenant at a time — the zero-contention baseline."""
    engine = llm.llm_engine
    params_by_tenant = {
        idx: _build_sampling_params(cfg, llm, idx)
        for idx in range(len(workload.prompts))
    }
    prompts = [_render_prompt(cfg, llm, p) for p in workload.prompts]
    _warm_engine(engine)

    order = sorted(range(len(workload.arrivals_s)), key=lambda i: workload.arrivals_s[i])
    arrivals = [workload.arrivals_s[i] for i in order]

    recs: dict[str, _RequestRec] = {}
    t0 = time.perf_counter()

    print(f"[baseline] {cfg.num_tenants} tenants sequentially (no overlap)")

    for slot, idx in enumerate(order):
        wait_until = arrivals[slot]
        now = time.perf_counter() - t0
        if wait_until > now:
            time.sleep(wait_until - now)

        req_id = workload.tenant_ids[idx]
        recs[req_id] = _RequestRec(
            request_id=req_id,
            tenant_idx=idx,
            prompt=workload.prompts[idx],
            arrival_s=time.perf_counter() - t0,
        )
        engine.add_request(
            req_id, prompts[idx], params_by_tenant[idx], arrival_time=time.time()
        )
        print(f"[baseline] +{req_id} @ t={recs[req_id].arrival_s:.3f}s")

        while recs[req_id].finish_s is None:
            outputs = engine.step()
            _record_outputs(recs, outputs, time.perf_counter() - t0)

    output_dir = Path(cfg.output_dir) / f"run_{workload.schedule}_{cfg.num_tenants}t_sequential"
    _build_session(cfg, llm, capture, layout, workload, list(recs.values()), t0, output_dir)
    return output_dir
