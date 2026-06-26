from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Slices token IDs into fixed-size input/target pairs."""

    def __init__(self, tokens: list[int], block_size: int):
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

