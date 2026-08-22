"""
workload.py — Multi-tenant workload generation.

Arrival schedules model real serving traffic: Poisson arrivals (memoryless,
the standard serving benchmark), periodic (fixed rate), burst (a clump of
requests followed by silence), and uniform (everyone at once).

Prompts come from a JSONL pool (one prompt per line, keys ``prompt`` /
``text``) or a small built-in pool, so every tenant runs a distinct query
like real users.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

_SCHEDULES = ("poisson", "periodic", "burst", "uniform")


@dataclass
class Workload:
    """One generated request workload for a serving session."""

    arrivals_s: list[float]  # session-clock arrival time per tenant
    prompts: list[str]  # one prompt per tenant
    schedule: str
    rate: float
    seed: int
    tenant_ids: list[str] = field(default_factory=list)
    family: str = "mixed"  # "identical" | "similar" | "mixed"
    slots_changed: list[int] = field(default_factory=list)  # per tenant (family=similar)

    def __post_init__(self):
        if not self.tenant_ids:
            self.tenant_ids = [f"tenant-{i:03d}" for i in range(len(self.arrivals_s))]
        assert len(self.arrivals_s) == len(self.prompts) == len(self.tenant_ids)


def generate_arrivals(n: int, rate: float, schedule: str, seed: int) -> list[float]:
    """Return arrival offsets (seconds) for ``n`` requests.

    Parameters
    ----------
    n : int
        Number of requests.
    rate : float
        Mean arrival rate in requests/second.
    schedule : str
        One of ``poisson`` | ``periodic`` | ``burst`` | ``uniform``.
    seed : int
        RNG seed (reproducible traffic).
    """
    if schedule not in _SCHEDULES:
        raise ValueError(f"Unknown schedule {schedule!r}; use one of {_SCHEDULES}")
    rng = random.Random(seed)
    mean_gap = 1.0 / rate if rate > 0 else 0.0

    if schedule == "uniform":
        return [0.0] * n

    if schedule == "periodic":
        return [i * mean_gap for i in range(n)]

    if schedule == "burst":
        # One tight clump at t=0, then silence — worst-case contention.
        return [i * min(mean_gap, 0.05) for i in range(n)]

    # Poisson: exponential inter-arrival gaps.
    arrivals: list[float] = []
    t = 0.0
    for _ in range(n):
        t += rng.expovariate(rate) if rate > 0 else 0.0
        arrivals.append(t)
    return arrivals


def load_prompt_pool(path: str | Path | None, seed: int = 0) -> list[str]:
    """Load prompts from a JSONL file (or the built-in pool)."""
    if path is not None:
        pool: list[str] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("prompt") or rec.get("text") or rec.get("instruction")
                if text:
                    pool.append(str(text))
        if pool:
            return pool
        raise ValueError(f"No usable prompts found in {path}")

    return _DEFAULT_PROMPTS


_DEFAULT_PROMPTS = [
    "Explain why Mixture of Experts models need routing, in one paragraph.",
    "What is the difference between expert parallelism and data parallelism?",
    "Describe how top-k gating selects experts for a token.",
    "Why do MoE models suffer from load imbalance across experts?",
    "Summarize the role of the shared expert in Qwen MoE architectures.",
    "What is the advantage of sparse activation in large language models?",
    "Explain the concept of token-to-expert affinity in MoE inference.",
    "How does an all-to-all dispatch work in expert-parallel inference?",
    "What is the purpose of a router network in a sparsely gated model?",
    "Describe the trade-off between more experts and higher inference latency.",
    "How does vLLM batch decode requests from different tenants together?",
    "What does time-to-first-token measure in LLM serving?",
    "Explain the difference between prefill and decode phases in inference.",
    "Why does concurrent serving reduce hardware utilization gaps?",
    "What is expert offloading and when is it used in MoE serving?",
    "How do serving engines schedule prefill work around decode work?",
    "Describe KV-cache paging and its benefit for multi-tenant serving.",
    "What causes contention between concurrent inference requests?",
    "Explain why throughput and per-request latency trade off in serving.",
    "What is a good arrival-rate model for user requests to an LLM API?",
]


def build_workload(
    num_tenants: int,
    schedule: str,
    rate: float,
    seed: int,
    prompts_file: str | Path | None = None,
    family: str = "mixed",
    base_prompt: str | None = None,
    slot_step: int = 1,
) -> Workload:
    """Build a full workload: arrivals + prompts + tenant ids.

    Parameters
    ----------
    family : str
        Prompt family for the tenants:
          * ``identical`` — every tenant sends the same prompt (reference
            distribution; with greedy sampling all traces must match).
          * ``similar``   — tenant 0 is the base prompt; tenant i varies an
            increasing number of template slots, giving a controlled
            prompt-similarity gradient.
          * ``mixed``     — independent random prompts from the pool.
    base_prompt : str | None
        Base prompt override.  May contain ``{slot}`` placeholders; slots
        are filled from the built-in vocabulary.
    slot_step : int
        For ``similar``: extra slots changed per tenant index step.
    """
    if family == "identical":
        prompts, slots_changed = _identical_family(
            num_tenants, seed, prompts_file, base_prompt
        )
    elif family == "similar":
        prompts, slots_changed = _similar_family(
            num_tenants, seed, prompts_file, base_prompt, slot_step
        )
    else:
        pool = load_prompt_pool(prompts_file, seed)
        rng = random.Random(seed + 1)
        prompts = [rng.choice(pool) for _ in range(num_tenants)]
        slots_changed = []

    arrivals = generate_arrivals(num_tenants, rate, schedule, seed)
    return Workload(
        arrivals_s=arrivals,
        prompts=prompts,
        schedule=schedule,
        rate=rate,
        seed=seed,
        family=family,
        slots_changed=slots_changed,
    )


# ═══════════════════════════════════════════════════════════════════
# Prompt families (controlled input similarity)
# ═══════════════════════════════════════════════════════════════════

# Template families: (template, slot → vocabulary).  Slots are the
# controlled independent variable — changing k slots moves the prompt a
# known edit distance away from the base.
_FAMILY_TEMPLATES: list[tuple[str, dict[str, list[str]]]] = [
    (
        "Explain why {topic} matters in {domain}, in one paragraph.",
        {
            "topic": [
                "Mixture of Experts routing",
                "expert parallelism",
                "sparse activation",
                "token-to-expert gating",
                "router load balancing",
            ],
            "domain": [
                "large language model serving",
                "distributed inference systems",
                "datacenter AI infrastructure",
                "real-time model deployment",
            ],
        },
    ),
    (
        "Describe the role of {concept} in {domain} systems.",
        {
            "concept": [
                "load balancing",
                "KV-cache paging",
                "circuit switching",
                "router networks",
                "expert placement",
            ],
            "domain": [
                "MoE inference",
                "LLM serving",
                "sparse model",
                "multi-tenant",
            ],
        },
    ),
    (
        "How does {tech} reduce {cost} in {domain} deployments?",
        {
            "tech": [
                "expert affinity scheduling",
                "shared KV caching",
                "optical circuit switching",
                "expert offloading",
            ],
            "cost": [
                "communication overhead",
                "compute waste",
                "inference latency",
                "bandwidth pressure",
            ],
            "domain": [
                "multi-tenant MoE",
                "large-scale LLM",
                "GPU-cluster",
                "edge MoE",
            ],
        },
    ),
]


def _resolve_template(template: str, vocab: dict[str, list[str]], rng: random.Random,
                      slot_choices: dict[str, int] | None = None) -> tuple[str, dict[str, int]]:
    """Fill a template's slots.  ``slot_choices`` pins slot→vocab index."""
    import re

    slot_choices = dict(slot_choices or {})
    for slot in re.findall(r"\{(\w+)\}", template):
        if slot not in slot_choices:
            slot_choices[slot] = rng.randrange(len(vocab[slot]))
    filled = template
    for slot, idx in slot_choices.items():
        filled = filled.replace("{" + slot + "}", vocab[slot][idx])
    return filled, slot_choices


def _pick_base(prompts_file, base_prompt, seed) -> tuple[str, dict[str, list[str]]]:
    """Choose the family base template (+ vocab) for the run."""
    if base_prompt:
        # Custom base: slots (if any) draw from every family vocabulary.
        merged: dict[str, list[str]] = {}
        for _, vocab in _FAMILY_TEMPLATES:
            for slot, entries in vocab.items():
                merged.setdefault(slot, []).extend(entries)
        return base_prompt, merged
    # Built-in families, deterministically selected.
    rng = random.Random(seed + 100)
    template, vocab = _FAMILY_TEMPLATES[rng.randrange(len(_FAMILY_TEMPLATES))]
    return template, vocab


def _identical_family(n: int, seed: int, prompts_file, base_prompt):
    template, vocab = _pick_base(prompts_file, base_prompt, seed)
    rng = random.Random(seed + 200)
    if "{" in template:
        base, _ = _resolve_template(template, vocab, rng)
    else:
        base = template
    return [base] * n, [0] * n


def _similar_family(n: int, seed: int, prompts_file, base_prompt, slot_step: int):
    template, vocab = _pick_base(prompts_file, base_prompt, seed)
    rng = random.Random(seed + 300)
    slots = sorted({m for m in __import__("re").findall(r"\{(\w+)\}", template)})
    base, base_choices = _resolve_template(template, vocab, rng)

    prompts = [base]
    slots_changed = [0]
    for i in range(1, n):
        changes = min(len(slots), ((i - 1) // slot_step) + 1) if slots else 0
        if changes == 0:
            prompts.append(base)
            slots_changed.append(0)
            continue
        choices = dict(base_choices)
        for slot in slots[:changes]:
            # Pick a different vocab entry than the base choice.
            others = [j for j in range(len(vocab[slot])) if j != choices[slot]]
            choices[slot] = rng.choice(others)
        filled, _ = _resolve_template(template, vocab, rng, choices)
        prompts.append(filled)
        slots_changed.append(changes)
    return prompts, slots_changed
