from __future__ import annotations

from pathlib import Path

import torch

from llm_from_scratch.model import TinyGPT
from llm_from_scratch.tokenizer import SimpleTokenizer


def load_checkpoint(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab = checkpoint["vocab"]
    tokenizer = SimpleTokenizer.from_vocab(vocab)

    model = TinyGPT(
        vocab_size=len(vocab),
        block_size=checkpoint["block_size"],
        n_layer=checkpoint["n_layer"],
        n_head=checkpoint["n_head"],
        n_embd=checkpoint["n_embd"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, checkpoint


@torch.no_grad()
def generate_text(
    model: TinyGPT,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: int | None = 5,
) -> str:
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(out[0].tolist())
