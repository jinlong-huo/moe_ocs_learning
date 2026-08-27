#!/usr/bin/env python3
"""capture_workload.py — batch routing capture over the categorised suite.

Loads the MoE model ONCE and captures a canonical ``RoutingTrace`` per prompt,
plus a manifest carrying the experimental labels (category / group / role /
variant).  The manifest is what makes every downstream number reproducible
without re-running inference: the traces are the raw data, the manifest is the
design matrix.

Two backends:
  mlx   — direct mlx_lm forward with the class-level MoE gate patch (default;
          works on Apple Silicon with no vLLM install)
  vllm  — delegate to scripts/run_vllm.py per prompt (subprocess isolation)

Usage
-----
    # full suite, small fast MoE (60 experts, top-4)
    python3 scripts/capture_workload.py \
        --model models/Qwen1.5-MoE-A2.7B-Chat-4bit \
        --out logs/workload/qwen15 --max-tokens 64

    # flagship scale (256 experts, top-8), subsampled suite
    python3 scripts/capture_workload.py \
        --model models/Qwen3.6-35B-A3B-4bit \
        --out logs/workload/qwen36 --max-tokens 48 --per-category 3

    # smoke
    python3 scripts/capture_workload.py --limit 4 --max-tokens 8 \
        --out logs/workload/smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.serving.suite import build_suite, suite_summary  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# MLX backend — model loaded once, capture swapped per prompt
# ═══════════════════════════════════════════════════════════════════════

class _MlxRunner:
    """Holds one loaded MLX MoE model and captures routing per prompt.

    The MoE gate patch is installed on the block CLASS (MLX resolves dunders
    on the class), so the patched ``__call__`` cannot close over a
    per-prompt capture object.  It therefore reads a mutable one-slot holder
    which this runner rebinds before each prompt.  That keeps exactly one
    patch installed for the whole session — no re-patching, no stale
    ``_layer_idx`` tags, no accumulated hook layers.
    """

    def __init__(self, model_path: str, no_chat: bool = False,
                 system_prompt: str = "You are a helpful assistant."):
        import mlx.core as mx
        from mlx_lm import load

        self.mx = mx
        self.model, self.tokenizer = load(model_path)
        self.model_path = model_path
        self.no_chat = no_chat
        self.system_prompt = system_prompt

        self._slot: dict = {"capture": None, "state": None}
        self.meta = self._extract_meta()
        self._install()

    # ── introspection ────────────────────────────────────────────────
    def _layers(self):
        m = self.model
        for attr in ("layers",):
            if hasattr(m, attr):
                return getattr(m, attr)
        for holder in ("model", "language_model"):
            sub = getattr(m, holder, None)
            if sub is not None and hasattr(sub, "layers"):
                return sub.layers
        raise AttributeError("cannot locate decoder layers")

    def _moe_blocks(self):
        out = []
        for idx, layer in enumerate(self._layers()):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "switch_mlp"):
                out.append((idx, mlp))
        return out

    def _extract_meta(self) -> dict:
        blocks = self._moe_blocks()
        num_experts = top_k = 0
        if blocks:
            b = blocks[0][1]
            num_experts = int(getattr(b, "num_experts", 0) or 0)
            top_k = int(getattr(b, "top_k", 0) or 0)
        cfg = getattr(self.model, "config", None)
        model_type = "unknown"
        if cfg is not None:
            if not num_experts:
                num_experts = int(getattr(cfg, "num_experts", 0) or 0)
            if not top_k:
                top_k = int(getattr(cfg, "num_experts_per_tok", 0) or 0)
            model_type = getattr(cfg, "model_type", model_type)
        if not num_experts:
            # last resort: read config.json off disk
            try:
                c = json.load(open(Path(self.model_path) / "config.json"))
                num_experts = int(c.get("num_experts") or c.get("num_local_experts") or 0)
                top_k = top_k or int(c.get("num_experts_per_tok") or 0)
                model_type = c.get("model_type", model_type)
            except Exception:
                pass
        return {
            "model_id": Path(self.model_path).name,
            "model_type": model_type,
            "num_layers": len(self._layers()),
            "num_experts": num_experts,
            "top_k": top_k,
            "moe_layer_indices": [i for i, _ in blocks],
        }

    # ── the gate patch ───────────────────────────────────────────────
    def _install(self) -> int:
        mx = self.mx
        slot = self._slot
        blocks = self._moe_blocks()
        if not blocks:
            raise RuntimeError("no MoE blocks found — is this a dense model?")

        moe_cls = type(blocks[0][1])
        for idx, mlp in blocks:
            mlp._layer_idx = idx
        orig_call = moe_cls.__call__

        def patched(self_block, x):
            cap = slot["capture"]
            gates = self_block.gate(x)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = self_block.top_k
            # argpartition gives an UNORDERED top-k; sort by score descending so
            # experts[0] is the argmax and "rank within top-k" is meaningful.
            inds = mx.stop_gradient(
                mx.argpartition(-gates, kth=k - 1, axis=-1)[..., :k]
            )
            scores = mx.take_along_axis(gates, inds, axis=-1)
            order = mx.argsort(-scores, axis=-1)
            inds = mx.take_along_axis(inds, order, axis=-1)
            scores = mx.take_along_axis(scores, order, axis=-1)
            if getattr(self_block, "norm_topk_prob", False):
                scores = scores / scores.sum(axis=-1, keepdims=True)

            if cap is not None:
                e = inds.tolist()
                w = scores.tolist()
                if e and isinstance(e[0][0], list):   # [B, T, K] -> strip batch
                    e, w = e[0], w[0]
                cap.log(layer_id=self_block._layer_idx,
                        batch_token_experts=e, batch_token_weights=w)

            # Reproduce the block's own compute path with OUR (sorted) inds so
            # the logged decision is exactly the decision executed.
            if getattr(self_block, "use_shared_mlp", False):
                y = self_block.switch_mlp(x, inds)
                y = (y * scores[..., None].astype(mx.float32)).sum(axis=-2).astype(y.dtype)
                return y + self_block.shared_mlp(x)
            y = self_block.switch_mlp(x, inds)
            y = (y * scores[..., None]).sum(axis=-2)
            gate_fn = getattr(self_block, "shared_expert_gate", None)
            if gate_fn is not None:
                y = y + mx.sigmoid(gate_fn(x)) * self_block.shared_expert(x)
            return y

        moe_cls.__call__ = patched
        self._moe_cls, self._orig_call = moe_cls, orig_call
        return len(blocks)

    def restore(self):
        if getattr(self, "_moe_cls", None) is not None:
            self._moe_cls.__call__ = self._orig_call

    # ── one prompt ───────────────────────────────────────────────────
    def run(self, prompt: str, max_tokens: int, temp: float = 0.0):
        from mlx_lm.models.cache import make_prompt_cache
        from src.data.mlx_capture import RoutingCapture

        mx = self.mx
        state = {"seq_pos": 0, "phase": "prefill"}
        capture = RoutingCapture(state)
        self._slot["capture"] = capture
        self._slot["state"] = state

        tok = self.tokenizer
        use_chat = (not self.no_chat and hasattr(tok, "apply_chat_template")
                    and tok.chat_template is not None)
        if use_chat:
            msgs = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
            prompt_tokens = tok.encode(text)
        else:
            prompt_tokens = tok.encode(prompt)

        cache = make_prompt_cache(self.model)
        arr = mx.array([prompt_tokens])

        state["phase"] = "prefill"
        logits = self.model(arr, cache=cache)[:, -1, :]
        state["seq_pos"] += len(prompt_tokens)

        generated: list[int] = []
        for _ in range(max_tokens):
            state["phase"] = "decode"
            if temp <= 0:
                nid = int(mx.argmax(logits, axis=-1).item())
            else:
                nid = int(mx.random.categorical(logits / temp).item())
            generated.append(nid)
            if nid == tok.eos_token_id:
                break
            logits = self.model(mx.array([[nid]]), cache=cache)[:, -1, :]
            state["seq_pos"] += 1

        trace = capture.build_trace(
            prompt_tokens=prompt_tokens, generated_tokens=generated,
            tokenizer=tok, model_id=self.meta["model_id"],
            model_type=self.meta["model_type"],
            num_layers=self.meta["num_layers"],
            num_experts=self.meta["num_experts"],
            top_k=self.meta["top_k"], backend="mlx",
        )
        self._slot["capture"] = None
        return trace, tok.decode(generated)


# ═══════════════════════════════════════════════════════════════════════
# vLLM backend — one subprocess per prompt (isolation, slower)
# ═══════════════════════════════════════════════════════════════════════

def _capture_vllm(model: str, prompt: str, max_tokens: int, temp: float,
                  out_path: Path):
    import subprocess
    from src.data.routing_schema import RoutingTrace
    cmd = [sys.executable, str(_repo_root / "scripts" / "run_vllm.py"), "run",
           "--model", model, "--prompt", prompt,
           "--max-tokens", str(max_tokens), "--temp", str(temp),
           "--output", str(out_path)]
    p = subprocess.run(cmd, cwd=_repo_root, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stdout + p.stderr)[-1200:])
    return RoutingTrace.load(out_path), ""


# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="Batch routing capture over the suite")
    ap.add_argument("--model", default="models/Qwen1.5-MoE-A2.7B-Chat-4bit")
    ap.add_argument("--backend", choices=["mlx", "vllm"], default="mlx")
    ap.add_argument("--out", default="logs/workload/default")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="0 = greedy; keep 0 so repeats isolate NUMERICAL noise only")
    ap.add_argument("--per-category", type=int, default=None,
                    help="subsample each semantic category (seeded)")
    ap.add_argument("--n-repeats", type=int, default=4,
                    help="identical-prompt repeats -> measurement noise floor")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only")
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip prompts whose trace file already exists")
    args = ap.parse_args()

    specs = build_suite(per_category=args.per_category,
                        n_repeats=args.n_repeats, seed=args.seed)
    if args.limit:
        specs = specs[: args.limit]

    out_dir = Path(args.out)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    print(f"[capture] model={args.model} backend={args.backend}")
    print(f"[capture] suite: {json.dumps(suite_summary(specs))}")
    print(f"[capture] out={out_dir}")

    runner = None
    if args.backend == "mlx":
        t0 = time.time()
        runner = _MlxRunner(args.model, no_chat=args.no_chat)
        print(f"[capture] loaded in {time.time()-t0:.1f}s | meta="
              f"{ {k: v for k, v in runner.meta.items() if k != 'moe_layer_indices'} }")
        print(f"[capture] MoE layers: {len(runner.meta['moe_layer_indices'])}")

    records, failures = [], []
    t_start = time.time()
    for i, spec in enumerate(specs):
        tp = traces_dir / f"{spec.uid}.json"
        if args.resume and tp.exists():
            records.append({**spec.to_dict(), "trace": f"traces/{spec.uid}.json",
                            "skipped": True})
            continue
        try:
            t0 = time.time()
            if runner is not None:
                trace, text = runner.run(spec.prompt, args.max_tokens, args.temp)
            else:
                trace, text = _capture_vllm(args.model, spec.prompt,
                                            args.max_tokens, args.temp, tp)
            trace.validate()
            trace.save(str(tp))
            dt = time.time() - t0
            records.append({
                **spec.to_dict(),
                "trace": f"traces/{spec.uid}.json",
                "prompt_len": trace.meta.prompt_len,
                "generated_len": trace.meta.generated_len,
                "total_tokens": trace.meta.total_tokens,
                "n_cells": trace.total_routing_events(),
                "capture_s": round(dt, 3),
                "completion_head": text[:80],
            })
            done = i + 1
            eta = (time.time() - t_start) / done * (len(specs) - done)
            print(f"[{done:3d}/{len(specs)}] {spec.uid:<34s} "
                  f"tok={trace.meta.total_tokens:4d} cells={trace.total_routing_events():6d} "
                  f"{dt:5.1f}s  eta={eta/60:5.1f}m")
        except Exception as e:
            print(f"[{i+1:3d}/{len(specs)}] {spec.uid:<34s} FAILED: {str(e)[:160]}")
            failures.append({"uid": spec.uid, "error": str(e)[:600]})

    if runner is not None:
        runner.restore()

    meta = runner.meta if runner is not None else {}
    manifest = {
        "model": args.model,
        "backend": args.backend,
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "seed": args.seed,
        "model_meta": meta,
        "suite_summary": suite_summary(specs),
        "n_captured": len(records),
        "wall_s": round(time.time() - t_start, 1),
        "records": records,
        "failures": failures,
    }
    mp = out_dir / "manifest.json"
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[capture] {len(records)} traces, {len(failures)} failures "
          f"in {manifest['wall_s']}s -> {mp}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
