"""
schema.py — Multi-tenant serving session data structures.

A ``MultiTenantSession`` ties together everything a concurrent vLLM
serving run produces:

  * one canonical ``RoutingTrace`` per tenant (schema-compatible with the
    single-stream HF/MLX backends), and
  * session-level timing + engine-step composition, so bandwidth
    contention and per-tenant delay are measurable.

With a single tenant, per-token delay is just the device speed — there is
no contention. With N tenants sharing the engine, the per-step
``tokens`` map (which tenant's tokens were co-computed in each engine
step) plus per-request TTFT / ITL series quantify the contention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class SessionMeta:
    """Static description of one multi-tenant serving run."""

    model_id: str
    model_type: str
    backend: str  # "vllm" (vllm-metal on Apple Silicon)
    num_layers: int
    num_moe_layers: int
    num_experts: int
    top_k: int
    schedule: str  # "poisson" | "periodic" | "burst" | "uniform"
    rate: float  # request arrival rate (req/s), if applicable
    num_tenants: int
    max_tokens: int
    mode: str  # "concurrent" | "sequential" (baseline, no overlap)
    seed: int
    family: str = "mixed"  # "identical" | "similar" | "mixed"
    temperature: float = 0.6
    seed_mode: str = "same"  # "same" | "distinct" (per-tenant sampling seeds)
    prefix_caching: bool = True
    run_id: str = field(default_factory=lambda: __import__("uuid").uuid4().hex[:12])


@dataclass
class TenantSummary:
    """Timing + identity of one tenant request in the session."""

    request_id: str
    tenant_idx: int
    prompt: str
    prompt_len: int
    generated_len: int
    arrival_s: float  # session clock, when the request entered the engine
    first_token_s: float  # session clock, first generated token observed
    finish_s: float  # session clock, request marked finished
    ttft_s: float
    token_timestamps_s: list[float]  # one per generated token
    itl_s: list[float]  # inter-token latencies (len = generated_len - 1)
    text: str
    trace_path: str  # relative path to the per-tenant RoutingTrace JSON
    slots_changed: int = 0  # prompt edit distance vs family base (similar family)
    prefix_cache_hit: float = 0.0  # 1.0 = whole prompt served from KV cache
    prefill_routes_logged: int = 0  # prompt positions actually recomputed

    @property
    def tpot_s(self) -> float:
        """Mean time-per-output-token (ITL average)."""
        if not self.itl_s:
            return 0.0
        return sum(self.itl_s) / len(self.itl_s)

    @property
    def output_throughput_tok_s(self) -> float:
        if self.finish_s <= self.first_token_s:
            return 0.0
        return self.generated_len / (self.finish_s - self.first_token_s)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StepRecord:
    """One engine forward step and its tenant composition.

    ``tokens`` maps tenant request_id → number of tokens that tenant had
    computed in this step (prefill tokens count too). Steps with ≥2
    tenants are the contention windows.
    """

    step: int
    t_s: float  # session clock at forward submission
    duration_s: float  # wall time until the next forward was submitted
    tokens: dict[str, int]  # request_id → token count


@dataclass
class MultiTenantSession:
    """Complete record of one multi-tenant serving run."""

    meta: SessionMeta
    tenants: list[TenantSummary] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)

    # ── serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "meta": asdict(self.meta),
            "tenants": [t.to_dict() for t in self.tenants],
            "steps": [asdict(s) for s in self.steps],
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "MultiTenantSession":
        with open(path) as f:
            raw = json.load(fp=f)
        meta_raw = raw["meta"]
        meta = SessionMeta(
            model_id=meta_raw["model_id"],
            model_type=meta_raw.get("model_type", "unknown"),
            backend=meta_raw.get("backend", "vllm"),
            num_layers=meta_raw.get("num_layers", 0),
            num_moe_layers=meta_raw.get("num_moe_layers", 0),
            num_experts=meta_raw.get("num_experts", 0),
            top_k=meta_raw.get("top_k", 0),
            schedule=meta_raw.get("schedule", "poisson"),
            rate=meta_raw.get("rate", 0.0),
            num_tenants=meta_raw.get("num_tenants", 0),
            max_tokens=meta_raw.get("max_tokens", 0),
            mode=meta_raw.get("mode", "concurrent"),
            seed=meta_raw.get("seed", 0),
            family=meta_raw.get("family", "mixed"),
            temperature=meta_raw.get("temperature", 0.6),
            seed_mode=meta_raw.get("seed_mode", "same"),
            prefix_caching=meta_raw.get("prefix_caching", True),
            run_id=meta_raw.get("run_id", ""),
        )
        tenants = [
            TenantSummary(
                request_id=t["request_id"],
                tenant_idx=t["tenant_idx"],
                prompt=t["prompt"],
                prompt_len=t["prompt_len"],
                generated_len=t["generated_len"],
                arrival_s=t["arrival_s"],
                first_token_s=t["first_token_s"],
                finish_s=t["finish_s"],
                ttft_s=t["ttft_s"],
                token_timestamps_s=t["token_timestamps_s"],
                itl_s=t["itl_s"],
                text=t.get("text", ""),
                trace_path=t.get("trace_path", ""),
                slots_changed=t.get("slots_changed", 0),
            )
            for t in raw.get("tenants", [])
        ]
        steps = [StepRecord(**s) for s in raw.get("steps", [])]
        return cls(meta=meta, tenants=tenants, steps=steps)

    # ── convenience ────────────────────────────────────────────────

    def by_request_id(self) -> dict[str, TenantSummary]:
        return {t.request_id: t for t in self.tenants}

    def concurrency_windows(self) -> list[dict]:
        """Per-step concurrency: overlapping tenants sorted by time."""
        out = []
        for s in self.steps:
            out.append({
                "step": s.step,
                "t_s": s.t_s,
                "duration_s": s.duration_s,
                "num_tenants": len(s.tokens),
                "tokens": dict(s.tokens),
            })
        return out

    def peak_concurrency(self) -> int:
        return max((len(s.tokens) for s in self.steps), default=0)

    def total_time_s(self) -> float:
        if not self.tenants:
            return 0.0
        return max(t.finish_s for t in self.tenants) - min(
            t.arrival_s for t in self.tenants
        )

    def mean_ttft_s(self) -> float:
        if not self.tenants:
            return 0.0
        return sum(t.ttft_s for t in self.tenants) / len(self.tenants)

    def mean_tpot_s(self) -> float:
        tpots = [t.tpot_s for t in self.tenants if t.tpot_s > 0]
        if not tpots:
            return 0.0
        return sum(tpots) / len(tpots)

    def aggregate_throughput_tok_s(self) -> float:
        if not self.tenants:
            return 0.0
        total_gen = sum(t.generated_len for t in self.tenants)
        span = max(t.finish_s for t in self.tenants) - min(
            t.arrival_s for t in self.tenants
        )
        return total_gen / span if span > 0 else 0.0
