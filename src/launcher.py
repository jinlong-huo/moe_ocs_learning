"""Multi-process launcher: spawn N workers with torch.multiprocessing.

Usage:
  python -m src.launcher --config configs/synthetic_moe.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml
import torch.multiprocessing as mp


def load_config(config_path: str) -> dict:
    """Load a YAML config file with recursive extends: support.

    Resolves extends: chains transitively — if A extends B and B extends C,
    the result is C merged with B merged with A.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve extends recursively
    if "extends" in cfg:
        base_name = cfg.pop("extends")
        base_dir = os.path.dirname(os.path.abspath(config_path))
        base_path = os.path.join(base_dir, f"{base_name}.yaml")
        if os.path.exists(base_path):
            # Recursively resolve the base (handles chained extends)
            base_cfg = load_config(base_path)
            # Merge: base values, overridden by child
            merged = _deep_merge(base_cfg, cfg)
            return merged
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def launch(config: dict, trace_dir: str = "outputs/traces") -> None:
    """Spawn world_size workers, each as a separate process."""
    # Must import after setting start method
    from src.runtime.worker import worker

    world_size = config.get("world_size", 2)

    # Set start method
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass 
    
    processes = []  # check if this one is authentic, i.e, the mp can simulate the package tranverse then.
    for rank in range(world_size):
        p = mp.Process(
            target=worker,
            args=(rank, world_size, config, trace_dir),
            name=f"rank-{rank}",
        )
        p.start()
        processes.append(p)

    # Wait for all workers to finish
    for p in processes:
        p.join()

    # Check exit codes
    for p in processes:
        if p.exitcode != 0:
            print(f"[launcher] ERROR: {p.name} exited with code {p.exitcode}", file=sys.stderr)
            sys.exit(1)

    print(f"[launcher] All {world_size} workers finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE Communication Research Launcher")
    parser.add_argument("--config", type=str, default="configs/synthetic_moe.yaml",
                       help="Path to YAML config file")
    parser.add_argument("--trace-dir", type=str, default="outputs/traces",
                       help="Directory for Chrome Trace output")
    args = parser.parse_args()

    config = load_config(args.config)
    launch(config, trace_dir=args.trace_dir)
