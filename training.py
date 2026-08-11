"""RIDM için veri, eğitim, doğrulama ve model kaydı orkestrasyonu."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from .backend import ComputeBackend
from .core import RIDM
from .datasets import TextDataset


@dataclass(frozen=True)
class TrainingConfig:
    dim: int = 300
    window: int = 5
    rank: int = 128
    seed: int = 42
    min_count: int = 1
    max_vocab: int | None = None
    validation_ratio: float = 0.1
    test_ratio: float = 0.15
    batch_size: int = 65_536
    backend: str = "auto"
    device: str = "auto"
    threads: int | None = None


@dataclass
class TrainingResult:
    model: RIDM
    validation: dict
    test: dict
    vocabulary_size: int
    backend: dict


class Trainer:
    def __init__(self, config: TrainingConfig = TrainingConfig()):
        self.config = config

    def fit(self, dataset: TextDataset) -> TrainingResult:
        cfg = self.config
        vocab, counts, ids = dataset.numericalize(cfg.min_count, cfg.max_vocab)
        split = dataset.split(ids, cfg.validation_ratio, cfg.test_ratio)
        backend = ComputeBackend(cfg.backend, cfg.device, cfg.threads)
        model = RIDM(vocab, counts=counts, dim=cfg.dim, window=cfg.window, seed=cfg.seed, compute_backend=backend)
        # Her batch kendi bağlamını taşır; pencere sınırındaki kaybı önlemek için
        # önceki W tokenı sonraki batch'in başına eklenir.
        history: deque[int] = deque(maxlen=cfg.window)
        for batch in dataset.batches(split.train, cfg.batch_size):
            model.partial_fit(list(history) + batch)
            history.extend(batch)
        model.finalize(cfg.rank)
        return TrainingResult(model, model.evaluate(split.validation), model.evaluate(split.test), len(vocab), backend.info.to_dict())

    def save(self, result: TrainingResult, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        result.model.save(target / "model.npz")
        metadata = {"config": asdict(self.config), "validation": result.validation, "test": result.test,
                    "vocabulary_size": result.vocabulary_size, "backend": result.backend}
        (target / "training.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
