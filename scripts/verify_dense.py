"""Verify a downloaded MLX dense model loads and generates correctly."""

import argparse
import time
from pathlib import Path

from mlx_lm import load, generate


def main():
    parser = argparse.ArgumentParser(description="Verify a dense MLX model")
    parser.add_argument(
        "--model",
        default="models/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
        help="Path to the downloaded model directory",
    )
    parser.add_argument(
        "--prompt",
        default="Explain yourself, like the architecture moe or dense.",
        help="Test prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate",
    )
    args = parser.parse_args()

    model_path = Path(args.model)

    # ── 1. Load ────────────────────────────────────────────────
    print(f"Loading model from: {model_path}")
    t0 = time.time()
    model, tokenizer = load(str(model_path))
    load_time = time.time() - t0
    print(f"  ✓ Loaded in {load_time:.1f}s")

    # ── 2. Model info ──────────────────────────────────────────
    config = getattr(model, "config", None) or {}
    model_type = config.get("model_type", "unknown")
    hidden_size = config.get("hidden_size", "?")
    num_layers = config.get("num_hidden_layers", "?")
    vocab_size = config.get("vocab_size", "?")

    print(f"\nModel info:")
    print(f"  type:         {model_type}")
    print(f"  hidden_size:  {hidden_size}")
    print(f"  num_layers:   {num_layers}")
    print(f"  vocab_size:   {vocab_size}")

    # Check if this is dense or MoE
    is_moe = (
        config.get("num_experts", 0) > 0
        or config.get("text_config", {}).get("num_experts", 0) > 0
    )
    print(f"  architecture: {'MoE' if is_moe else 'Dense'}")

    # ── 3. Generate ────────────────────────────────────────────
    print(f"\nPrompt: {args.prompt!r}")
    print(f"Generating (max {args.max_tokens} tokens)...\n")

    t0 = time.time()

    # Tokenize
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": args.prompt}]
        prompt_tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    elif callable(getattr(tokenizer, "encode", None)):
        prompt_tokens = tokenizer.encode(args.prompt)
    else:
        prompt_tokens = tokenizer(args.prompt)

    # Generate
    response = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        verbose=True,
    )
    gen_time = time.time() - t0

    tok_count = response.count(tokenizer.eos_token or "<")  # rough
    print(f"\n───\n  Generation time: {gen_time:.1f}s")
    print(f"  Response length:  ~{len(response.split())} words")
    print("\n✓ Model verified successfully!")


if __name__ == "__main__":
    main()
