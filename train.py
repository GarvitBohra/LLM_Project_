from __future__ import annotations

from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader

from llm_from_scratch.data import TextDataset
from llm_from_scratch.model import TinyGPT
from llm_from_scratch.tokenizer import SimpleTokenizer


DATA_PATH = Path("data/tiny.txt")
CKPT_PATH = Path("checkpoints/model.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path, default=CKPT_PATH)
    args = parser.parse_args()

    text = args.data.read_text(encoding="utf-8")
    tokenizer = SimpleTokenizer(text)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = TextDataset(tokenizer.encode(text), block_size=args.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Training on: {device}")

    model.train()
    step = 0
    while step < args.steps:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 25 == 0:
                print(f"step {step:04d} | loss {loss.item():.4f}")

            step += 1
            if step >= args.steps:
                break

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": [tokenizer.itos[i] for i in range(tokenizer.vocab_size)],
            "block_size": args.block_size,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
        },
        args.checkpoint,
    )
    print(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
