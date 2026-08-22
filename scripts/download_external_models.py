#!/usr/bin/env python3
"""Download external MoE models for routing-capture experiments.

Two candidates compatible with this repo's pipeline (MLX format + Qwen-style
MoE routing capture):

  * whittle — logic65/Qwen3.8-Whittle-MoE-27B-A17.8B
      model_type qwen3_5_moe (same family as Qwen3.6-35B-A3B-4bit):
      64 experts, top-16.  BF16 safetensors only (no public MLX conversion),
      so after download convert locally with mlx_lm.convert:
          .venv/bin/mlx_lm.convert --hf-path <dir> --mlx-path <dir>-4bit -q --q-bits 4
  * hy3 — mlx-community/Hy3-oQ2
      MLX format, 2-bit, ready to load.  Architecture is hy_v3 (Tencent
      Hunyuan MoE, 192 experts, top-8) — NOT Qwen: moe_run.py's routing hook
      (mlp.switch_mlp) needs adaptation before capture works.

Skipped (checked, not compatible):
  * DavidAU Mistral-MOE-4X7B / Llama-3.2-8X3B-MOE GGUF — GGUF-only
    mergekit franken-MoE, GGUF metadata reports plain llama arch.
  * amd/Instella-MoE-16B-A3B — repo doesn't exist under that name;
    actual repos are deepseek_v3-arch BF16, no MLX conversion.

Usage:
    .venv/bin/python scripts/download_external_models.py --target whittle
    .venv/bin/python scripts/download_external_models.py --target hy3
    .venv/bin/python scripts/download_external_models.py --target all
    .venv/bin/python scripts/download_external_models.py --target whittle --dry-run
    .venv/bin/python scripts/download_external_models.py --target whittle --mirror
"""
from __future__ import annotations

import argparse
import os
import sys

TARGETS = {
    "whittle": {
        "repo": "logic65/Qwen3.8-Whittle-MoE-27B-A17.8B",
        "allow_patterns": [
            "*.safetensors",
            "*.json",
            "*.jinja",
            "LICENSE",
        ],
        "note": "BF16 ~54 GB; convert to MLX 4-bit after download "
                "(mlx_lm.convert -q --q-bits 4)",
    },
    "hy3": {
        "repo": "mlx-community/Hy3-oQ2",
        "allow_patterns": [
            "*.safetensors",
            "*.json",
            "*.jinja",
        ],
        "note": "MLX 2-bit, ready to load; hy_v3 arch needs hook adaptation",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Download external MoE models")
    ap.add_argument("--target", required=True, choices=sorted(TARGETS) + ["all"])
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--dry-run", action="store_true",
                    help="List repo files + sizes without downloading")
    ap.add_argument("--mirror", action="store_true",
                    help="Use HF mirror (https://hf-mirror.com) via HF_ENDPOINT")
    args = ap.parse_args()

    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[mirror] HF_ENDPOINT=https://hf-mirror.com")

    from huggingface_hub import HfApi

    api = HfApi()
    picks = sorted(TARGETS) if args.target == "all" else [args.target]

    for name in picks:
        spec = TARGETS[name]
        repo = spec["repo"]
        print(f"\n[{name}] {repo}")
        print(f"[{name}] note: {spec['note']}")

        info = api.model_info(repo, files_metadata=True)
        total = 0.0
        for sib in info.siblings:
            f = sib.rfilename
            matched = any(
                f == p or f.endswith(p.removeprefix("*"))
                for p in spec["allow_patterns"]
            )
            if matched:
                size_gb = (sib.size or 0) / 1e9
                total += size_gb
                print(f"    {f:<50s} {size_gb:8.2f} GB")
        print(f"[{name}] total selected ≈ {total:.2f} GB")

        if args.dry_run:
            continue

        from huggingface_hub import snapshot_download

        out = snapshot_download(
            repo_id=repo,
            allow_patterns=spec["allow_patterns"],
            local_dir=f"{args.out_dir}/{repo.split('/')[-1]}",
            resume_download=True,
        )
        print(f"[{name}] downloaded → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
