"""Live vLLM inference with routing capture — the realtime capture library.

Runs REAL model inference through vLLM (vllm-metal on Apple Silicon,
CUDA elsewhere) in-process and returns the captured routing as a validated
``RoutingTrace`` — no pre-recorded trace files involved. This is the input
for the live verification gate (``scripts/verify_live_invariance.py``):
one model + one prompt + one inference run, captured at call time.

Environment: the in-process engine core (``VLLM_ENABLE_V1_MULTIPROCESSING=0``)
and the loopback host IP (``VLLM_HOST_IP=127.0.0.1``) are set before vLLM is
imported — the macOS hostname can resolve to an unreachable LAN IP and crash
PyTorch's TCPStore.
"""
from __future__ import annotations

import os
from typing import Optional

# Must run before vLLM is imported anywhere in this process.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
# vllm-metal engine knobs (same as the multi-tenant serving harness):
# decode pipeline off + stable request ids avoid the V1 scheduler/runner
# "Scheduled cached request(s) have no RequestState" desync on some models.
os.environ.setdefault("VLLM_METAL_DECODE_PIPELINE", "0")
os.environ.setdefault("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "1")


def capture_live(
    model: str,
    prompt: str,
    max_tokens: int = 32,
    temp: float = 0.0,
    seed: int = 0,
    system_prompt: str = "You are a helpful assistant.",
    no_chat: bool = False,
    enforce_eager: bool = False,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    trust_remote_code: bool = False,
    steering=None,
    verbose: bool = True,
):
    """Run one live inference and return the captured, validated RoutingTrace.

    Args:
        model: model path (HF id or local MLX-format dir for vllm-metal).
        prompt: user prompt (chat template applied unless ``no_chat``).
        max_tokens: generation length.
        temp: sampling temperature; 0 = greedy (deterministic traces).
        seed: sampling seed.
        steering: optional VllmSteering for interventions.
        verbose: print load/capture progress.

    Returns:
        ``src.data.routing_schema.RoutingTrace`` (validated).
    """
    import torch

    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise RuntimeError(
            "vllm is not importable in this interpreter — run with the "
            "vllm-metal environment, e.g.: ~/.venv-vllm-metal/bin/python "
            f"(or install vllm). Original error: {e}"
        ) from e
    from src.data.routing_schema import RoutingTrace
    from src.data.vllm_capture import (
        VllmRoutingCapture, install_vllm_hooks, install_vllm_metal_hooks,
        get_vllm_layout, locate_model,
    )

    llm_kwargs = dict(
        model=model,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
        # Traces must record the FULL routing — the engine must never serve
        # cached prefill routing (our capture rule: recompute, greedy,
        # prefix-cache off). Also avoids the vLLM V1 cached-request scheduler
        # desync seen with some models on the metal backend.
        enable_prefix_caching=False,
    )
    if max_model_len:
        llm_kwargs["max_model_len"] = max_model_len

    if verbose:
        print(f"[live] Loading model: {model}")
    llm = LLM(**llm_kwargs)

    loaded = locate_model(llm)
    if loaded is None:
        raise RuntimeError("Could not locate the loaded model for hook installation")

    capture = VllmRoutingCapture()
    is_metal = (
        not isinstance(loaded, torch.nn.Module)
        and loaded.__class__.__module__.startswith("mlx_lm")
    )
    if is_metal:
        install_vllm_metal_hooks(loaded, capture, steering)
    else:
        install_vllm_hooks(loaded, capture, steering)

    layout = get_vllm_layout(llm, capture)
    if verbose:
        print(f"[live] layout: model_type={layout['model_type']}, "
              f"layers={layout['num_layers']}, experts={layout['num_experts']}, "
              f"top_k={layout['top_k']}")

    tokenizer = llm.get_tokenizer()
    sampling_kwargs = dict(temperature=temp, max_tokens=max_tokens, seed=seed)
    hf_cfg = llm.llm_engine.vllm_config.model_config.hf_config
    eos = getattr(hf_cfg, "eos_token_id", None)
    if eos is not None:
        eos_ids = [int(e) for e in eos] if isinstance(eos, (list, tuple)) else [int(eos)]
        eos_ids = [e for e in eos_ids if e is not None]
        if eos_ids:
            sampling_kwargs["stop_token_ids"] = eos_ids
    sampling_params = SamplingParams(**sampling_kwargs)

    messages = None
    if not no_chat:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    if no_chat:
        outputs = llm.generate([prompt], sampling_params=sampling_params)
    else:
        try:
            outputs = llm.chat([messages], sampling_params=sampling_params)
        except Exception as e:
            if verbose:
                print(f"[live] chat template failed ({e}); falling back to plain prompt")
            outputs = llm.generate([prompt], sampling_params=sampling_params)

    out = outputs[0]
    trace = capture.build_trace(
        prompt_tokens=list(out.prompt_token_ids),
        generated_tokens=list(out.outputs[0].token_ids),
        tokenizer=tokenizer,
        model_id=model,
        backend="vllm",
        **layout,
    )
    trace.validate()
    if verbose:
        print(f"[live] captured: {trace.total_routing_events()} routing events, "
              f"{len(trace.routes)} token positions, "
              f"{trace.meta.num_moe_layers} MoE layers, "
              f"{trace.meta.num_experts} experts, top_k={trace.meta.top_k}")
    return trace
