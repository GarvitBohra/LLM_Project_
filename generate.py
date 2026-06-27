from __future__ import annotations

import argparse
from pathlib import Path

from llm_from_scratch.runtime import generate_text, load_checkpoint


CKPT_PATH = Path("checkpoints/model.pt")


def load_model(checkpoint_path: Path):
    model, tokenizer, _ = load_checkpoint(checkpoint_path)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="hello")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=CKPT_PATH)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            "No checkpoint found. Run `python train.py` first."
        )

    model, tokenizer = load_model(args.checkpoint)
    print(
        generate_text(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    )


if __name__ == "__main__":
    main()
