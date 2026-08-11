"""Tekrarlanabilir veri alma, bölme ve mini-batch altyapısı."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .vocab import build_vocab, encode


@dataclass(frozen=True)
class DatasetSplit:
    train: list[int]
    validation: list[int]
    test: list[int]


class TextDataset:
    def __init__(self, words: Sequence[str]):
        self.words = [word for word in words if word]
        if not self.words:
            raise ValueError("Eğitim verisi boş olamaz.")

    @classmethod
    def from_file(cls, path: str | Path, encoding: str = "utf-8") -> "TextDataset":
        return cls(Path(path).read_text(encoding=encoding).split())

    def numericalize(self, min_count: int = 1, max_vocab: int | None = None):
        vocab, counts = build_vocab(self.words, min_count=min_count, max_vocab=max_vocab)
        return vocab, counts, encode(self.words, {word: i for i, word in enumerate(vocab)})

    @staticmethod
    def split(token_ids: Sequence[int], validation_ratio: float = 0.1, test_ratio: float = 0.15) -> DatasetSplit:
        if not 0 <= validation_ratio < 1 or not 0 <= test_ratio < 1 or validation_ratio + test_ratio >= 1:
            raise ValueError("validation_ratio + test_ratio 1'den küçük olmalıdır.")
        n = len(token_ids)
        if n < 3:
            raise ValueError("Bölme için en az üç token gerekir.")
        test_start = max(1, int(n * (1 - test_ratio)))
        validation_start = max(1, int(test_start * (1 - validation_ratio)))
        return DatasetSplit(list(token_ids[:validation_start]), list(token_ids[validation_start:test_start]), list(token_ids[test_start:]))

    @staticmethod
    def batches(token_ids: Sequence[int], batch_size: int) -> Iterator[list[int]]:
        if batch_size <= 0:
            raise ValueError("batch_size pozitif olmalıdır.")
        for start in range(0, len(token_ids), batch_size):
            yield list(token_ids[start:start + batch_size])
