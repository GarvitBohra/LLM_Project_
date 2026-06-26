from __future__ import annotations

import re


class SimpleTokenizer:
    """A beginner-friendly tokenizer that splits text into words and spaces.

    It keeps spaces as tokens so decoded text preserves the original spacing.
    The tokenizer lowercases input to keep the vocabulary small and easier to learn.
    """

    _pattern = re.compile(r"\w+|[^\w\s]|\s+")

    def __init__(self, text: str):
        tokens = self._tokenize(text)
        vocab = ["<unk>"] + sorted(set(tokens))
        self.stoi = {tok: i for i, tok in enumerate(vocab)}
        self.itos = {i: tok for tok, i in self.stoi.items()}

    @classmethod
    def from_vocab(cls, vocab: list[str]) -> "SimpleTokenizer":
        obj = cls.__new__(cls)
        obj.stoi = {tok: i for i, tok in enumerate(vocab)}
        obj.itos = {i: tok for tok, i in obj.stoi.items()}
        return obj

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return SimpleTokenizer._pattern.findall(text.lower())

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(tok, unk) for tok in self._tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids)
