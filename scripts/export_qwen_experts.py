#!/usr/bin/env python3
"""
export_qwen_experts.py — Extract Qwen MoE expert weights for the OCS simulator.

Loads a Qwen MoE model via MLX (with optional LoRA adapter), extracts
individual expert weights and gate/router weights from every MoE layer,
and saves them as PyTorch-compatible .pt files organized by layer and
expert ID.

Output structure:
    exported_qwen_weights/
        layer_0/
            expert_0.pt   — {gate_proj, up_proj, down_proj} tensors
            expert_1.pt
            ...
            expert_255.pt
            gate.pt       — router gate weights
            shared_expert.pt
        layer_1/
            ...
        meta.json          — model architecture metadata

Usage:
    python3 scripts/export_qwen_experts.py \
        --model ./models/Qwen3.6-35B-A3B-4bit \
        --adapter adapters/qwen3.6-moe-lora \
        --output exported_qwen_weights
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("WARNING: mlx / mlx_lm not installed. Falling back to safetensors parser.")
    print("Install with: pip install mlx mlx-lm")


def _get_layers(model):
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    raise AttributeError("Cannot find decoder layers on model")


def _mx_to_numpy(arr) -> np.ndarray:
    """Convert mlx array to numpy, handling bfloat16."""
    return np.array(arr.astype(mx.float32) if arr.dtype == mx.bfloat16 else arr)


def _extract_quantized(mx_module, param_name: str = "weight"):
    """Extract quantized weight, scales, biases from MLX module.

    Handles both regular nn.Linear and LoRA-fused LoRALinear/LoRASwitchLinear.
    After fuse(), LoRA modules still have the LoRA wrapper but the merged
    weight is in parameters()['linear'].
    """
    if hasattr(mx_module, param_name):
        w = getattr(mx_module, param_name)
        s = getattr(mx_module, "scales", None)
        b = getattr(mx_module, "biases", None)
    elif hasattr(mx_module, "parameters"):
        params = mx_module.parameters()
        if "linear" in params and isinstance(params["linear"], dict):
            lin = params["linear"]
            w = lin.get(param_name, None)
            s = lin.get("scales", None)
            b = lin.get("biases", None)
            if w is None:
                raise AttributeError(f"Cannot find weight in module {type(mx_module).__name__}")
        else:
            raise AttributeError(f"Cannot extract weights from {type(mx_module).__name__}")
    else:
        raise AttributeError(f"Unknown module type: {type(mx_module).__name__}")
    return w, s, b


def _try_dequantize(weight, scales, biases) -> np.ndarray:
    """Dequantize a 4-bit MLX quantized tensor to float32."""
    if scales is None or biases is None:
        return _mx_to_numpy(weight)

    packed_cols = weight.shape[-1]
    num_groups = scales.shape[-1]
    group_size = (packed_cols * 8) // num_groups

    try:
        deq = mx.dequantize(weight, scales, biases, group_size=group_size, bits=4)
        return _mx_to_numpy(deq)
    except Exception:
        return _mx_to_numpy(weight)


def _merge_lora(dequantized: np.ndarray, mx_module) -> np.ndarray:
    """Merge LoRA into dequantized weight if LoRA params exist on module."""
    params = mx_module.parameters() if hasattr(mx_module, "parameters") else {}
    lora_a = params.get("lora_a", None)
    lora_b = params.get("lora_b", None)
    if lora_a is None or lora_b is None:
        return dequantized
    scale = getattr(mx_module, "scale", 1.0)

    a = _mx_to_numpy(lora_a)  # [in_features, rank]
    b = _mx_to_numpy(lora_b)  # [rank, out_features]

    lora_delta = scale * (b.T @ a.T)
    subset_cols = lora_delta.shape[1]
    result = dequantized.copy()
    result[:, :subset_cols] += lora_delta
    return result


def _fuse_all_lora(model) -> None:
    """Fuse all LoRA adapters into base layers for weight extraction."""
    layers = _get_layers(model)
    fused = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        for attr_name in ["gate", "switch_mlp", "shared_expert", "shared_expert_gate"]:
            module = getattr(mlp, attr_name, None)
            if module is None:
                continue
            if attr_name == "switch_mlp":
                for proj_attr in ["gate_proj", "up_proj", "down_proj"]:
                    proj = getattr(module, proj_attr, None)
                    if hasattr(proj, "fuse") and hasattr(proj, "linear"):
                        proj.fuse()
                        fused += 1
            elif hasattr(module, "fuse") and hasattr(module, "linear"):
                if hasattr(module, "linear"):
                    module.fuse()
                    fused += 1
    print(f"[lora] Fused {fused} LoRA adapters into base weights")


def export_with_mlx(model_path: str, adapter_path: str | None,
                    output_dir: str, max_layers: int | None = None) -> dict:
    """Export expert weights using MLX for loading + dequantization."""
    import mlx.nn as nn

    print(f"[mlx] Loading model from {model_path}...")
    model, tokenizer = mlx_load(str(model_path), adapter_path=adapter_path)
    if adapter_path:
        print(f"[mlx] LoRA adapter applied: {adapter_path}")

    # Fuse LoRA — MLX's fuse() merges adapters in-place during forward,
    # but for weight export we merge manually via _merge_lora after dequant
    if adapter_path:
        _fuse_all_lora(model)

    layers = _get_layers(model)
    num_experts = 0
    top_k = 0
    hidden_dim = 0
    intermediate_dim = 0
    moe_layer_indices = []

    # Extract model architecture info (dequantize first expert to get real dims)
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "switch_mlp"):
            continue
        moe_layer_indices.append(layer_idx)
        if num_experts == 0 and hasattr(mlp.switch_mlp, "gate_proj"):
            gp_module = mlp.switch_mlp.gate_proj
            gp_w, gp_s, gp_b = _extract_quantized(gp_module, "weight")
            num_experts = gp_w.shape[0]
            top_k = getattr(mlp, "top_k", 0) or 0

            deq_expert = _try_dequantize(gp_w[0], gp_s[0] if gp_s is not None else None,
                                          gp_b[0] if gp_b is not None else None)
            intermediate_dim = deq_expert.shape[0]
            hidden_dim = deq_expert.shape[1]

    if hidden_dim == 0:
        hidden_dim = 2048
        intermediate_dim = 512

    meta = {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "num_moe_layers": len(moe_layer_indices),
        "moe_layer_indices": moe_layer_indices,
    }
    print(f"[model] {num_experts} experts, hidden={hidden_dim}, "
          f"intermediate={intermediate_dim}, top_k={top_k}, "
          f"{len(moe_layer_indices)} MoE layers")

    layers_to_export = moe_layer_indices
    if max_layers is not None:
        layers_to_export = moe_layer_indices[:max_layers]

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total_exported = 0

    for moe_count, layer_idx in enumerate(layers_to_export):
        layer = layers[layer_idx]
        mlp = layer.mlp
        layer_dir = os.path.join(output_dir, f"layer_{moe_count}")
        os.makedirs(layer_dir, exist_ok=True)

        # Export gate (router) weights — dequantize + merge LoRA
        gate_w_raw, gate_scales, gate_biases = _extract_quantized(mlp.gate, "weight")
        gate_w = _try_dequantize(gate_w_raw, gate_scales, gate_biases)
        if adapter_path:
            gate_w = _merge_lora(gate_w, mlp.gate)
        np.save(os.path.join(layer_dir, "gate.npy"), gate_w)
        print(f"  layer {moe_count} (model layer {layer_idx}): gate {gate_w.shape}")

        # Export shared expert
        if hasattr(mlp, "shared_expert"):
            se_gp_w, se_gp_s, se_gp_b = _extract_quantized(mlp.shared_expert.gate_proj, "weight")
            se_up_w, se_up_s, se_up_b = _extract_quantized(mlp.shared_expert.up_proj, "weight")
            se_dn_w, se_dn_s, se_dn_b = _extract_quantized(mlp.shared_expert.down_proj, "weight")
            se_gate = _try_dequantize(se_gp_w, se_gp_s, se_gp_b)
            se_up = _try_dequantize(se_up_w, se_up_s, se_up_b)
            se_down = _try_dequantize(se_dn_w, se_dn_s, se_dn_b)
            se_gate_w = None
            if hasattr(mlp, "shared_expert_gate"):
                sg_w, sg_s, sg_b = _extract_quantized(mlp.shared_expert_gate, "weight")
                se_gate_w = _try_dequantize(sg_w, sg_s, sg_b)
            np.savez(
                os.path.join(layer_dir, "shared_expert.npz"),
                gate_proj=se_gate, up_proj=se_up, down_proj=se_down,
                shared_expert_gate=(se_gate_w if se_gate_w is not None
                                    else np.zeros((1, hidden_dim))),
            )

        # Export individual expert weights — dequantize per-expert
        switch = mlp.switch_mlp
        gp_w, gp_s, gp_b = _extract_quantized(switch.gate_proj, "weight")
        up_w, up_s, up_b = _extract_quantized(switch.up_proj, "weight")
        dn_w, dn_s, dn_b = _extract_quantized(switch.down_proj, "weight")

        for eid in range(num_experts):
            gp = _try_dequantize(gp_w[eid],
                                 gp_s[eid] if gp_s is not None else None,
                                 gp_b[eid] if gp_b is not None else None)
            up = _try_dequantize(up_w[eid],
                                 up_s[eid] if up_s is not None else None,
                                 up_b[eid] if up_b is not None else None)
            dn = _try_dequantize(dn_w[eid],
                                 dn_s[eid] if dn_s is not None else None,
                                 dn_b[eid] if dn_b is not None else None)
            np.savez(
                os.path.join(layer_dir, f"expert_{eid}.npz"),
                gate_proj=gp, up_proj=up, down_proj=dn,
            )
            total_exported += 1

        print(f"  layer {moe_count}: exported {num_experts} experts "
              f"(gate_proj [{intermediate_dim},{hidden_dim}], "
              f"up_proj [{intermediate_dim},{hidden_dim}], "
              f"down_proj [{hidden_dim},{intermediate_dim}])")

    print(f"\n[done] Exported {total_exported} experts across "
          f"{len(layers_to_export)} layers to {output_dir}/")
    return meta


def export_from_safetensors(model_path: str, output_dir: str,
                            max_layers: int | None = None) -> dict:
    """Export expert weights directly from safetensors (no MLX dependency)."""
    import torch
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No safetensors index at {index_path}")

    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Group keys by file to minimize file opens
    file_keys: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        file_keys.setdefault(filename, []).append(key)

    prefix = "language_model.model.layers."

    # Collect MoE layer indices from weight names
    moe_layers: dict[int, dict[str, str]] = {}
    for key in weight_map:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        parts = rest.split(".")
        if not parts[0].isdigit():
            continue
        layer_idx = int(parts[0])
        if "mlp.switch_mlp" in key or "mlp.gate" in key or "mlp.shared_expert" in key:
            if layer_idx not in moe_layers:
                moe_layers[layer_idx] = {}

    if not moe_layers:
        raise ValueError("No MoE layers found in model")

    layer_indices = sorted(moe_layers.keys())
    print(f"[model] Found {len(layer_indices)} MoE layers: {layer_indices[:5]}...")

    # Load all weights by file
    all_tensors: dict[str, torch.Tensor] = {}
    for filename, keys in file_keys.items():
        filepath = os.path.join(model_path, filename)
        with safe_open(filepath, framework="pt") as f:
            for key in keys:
                if any(f"layers.{lid}." in key for lid in
                       [str(li) for li in (layer_indices[:max_layers]
                                           if max_layers else layer_indices)]):
                    all_tensors[key] = f.get_tensor(key)

    # Determine architecture
    first_layer = layer_indices[0]
    gate_key = f"{prefix}{first_layer}.mlp.gate.weight"
    gate_shape = all_tensors[gate_key].shape
    num_experts = gate_shape[0]
    hidden_dim = gate_shape[1]

    switch_key = f"{prefix}{first_layer}.mlp.switch_mlp.gate_proj.weight"
    switch_shape = all_tensors[switch_key].shape
    intermediate_dim = switch_shape[2]

    # Determine top_k (try to infer from model name)
    path_lower = model_path.lower()
    top_k = 8 if "35b" in path_lower or "3.6" in path_lower else 4

    meta = {
        "model_path": model_path,
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "num_moe_layers": len(layer_indices),
        "moe_layer_indices": layer_indices,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    export_layers = layer_indices[:max_layers] if max_layers else layer_indices
    total_exported = 0

    for moe_count, layer_idx in enumerate(export_layers):
        layer_dir = os.path.join(output_dir, f"layer_{moe_count}")
        os.makedirs(layer_dir, exist_ok=True)

        lp = f"{prefix}{layer_idx}.mlp"

        # Gate
        gate_w = all_tensors[f"{lp}.gate.weight"]
        np.save(os.path.join(layer_dir, "gate.npy"), gate_w.numpy())
        print(f"  layer {moe_count} (model layer {layer_idx}): gate {gate_w.shape}")

        # Shared expert
        se_keys = {
            "gate_proj": f"{lp}.shared_expert.gate_proj.weight",
            "up_proj": f"{lp}.shared_expert.up_proj.weight",
            "down_proj": f"{lp}.shared_expert.down_proj.weight",
            "shared_expert_gate": f"{lp}.shared_expert_gate.weight",
        }
        se_data = {}
        for name, key in se_keys.items():
            if key in all_tensors:
                se_data[name] = all_tensors[key].numpy()
        if se_data:
            np.savez(os.path.join(layer_dir, "shared_expert.npz"), **se_data)

        # Expert weights
        gate_all = all_tensors[f"{lp}.switch_mlp.gate_proj.weight"]
        up_all = all_tensors[f"{lp}.switch_mlp.up_proj.weight"]
        down_all = all_tensors[f"{lp}.switch_mlp.down_proj.weight"]

        for eid in range(num_experts):
            np.savez(
                os.path.join(layer_dir, f"expert_{eid}.npz"),
                gate_proj=gate_all[eid].numpy(),
                up_proj=up_all[eid].numpy(),
                down_proj=down_all[eid].numpy(),
            )
            total_exported += 1

        print(f"  layer {moe_count}: exported {num_experts} experts")

    print(f"\n[done] Exported {total_exported} experts across "
          f"{len(export_layers)} layers to {output_dir}/")
    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen MoE expert weights for OCS simulator"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to MLX-format model directory"
    )
    parser.add_argument(
        "--adapter", default=None,
        help="Optional path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output", default="exported_qwen_weights",
        help="Output directory for exported weights"
    )
    parser.add_argument(
        "--max-layers", type=int, default=None,
        help="Export only first N MoE layers (default: all)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[error] Model not found: {args.model}")
        return 1

    if HAS_MLX:
        export_with_mlx(args.model, args.adapter, args.output, args.max_layers)
    else:
        export_from_safetensors(args.model, args.output, args.max_layers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
