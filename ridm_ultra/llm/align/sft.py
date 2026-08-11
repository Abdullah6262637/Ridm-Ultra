"""Önceden eğitilmiş checkpoint'ten başlayan, prompt-maskeli supervised fine-tuning (SFT)."""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from ..model.config import ModelConfig
from ..model.transformer import DecoderOnlyTransformer
from .data import SFTDataset, collate_sft_batch


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
        return torch, DataLoader
    except ImportError as exc:
        raise RuntimeError("SFT eğitimi için `pip install .[llm]` ile torch kurulmalıdır.") from exc


@dataclass(frozen=True)
class SFTConfig:
    output_dir: str | Path
    init_checkpoint: str | Path
    batch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 3
    peak_lr: float = 1e-5
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0
    precision: str = "bf16"
    num_workers: int = 0
    log_every: int = 10
    checkpoint_every: int = 200
    seed: int = 42

    def __post_init__(self):
        if self.batch_size <= 0 or self.grad_accum_steps <= 0 or self.epochs <= 0:
            raise ValueError("batch_size, grad_accum_steps ve epochs pozitif olmalıdır.")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio [0, 1) aralığında olmalıdır.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision fp32, fp16 veya bf16 olmalıdır.")


class LinearWarmupCosine:
    """Adım bazlı warmup + cosine decay; SFT'nin küçük, epoch-sayılabilir verisine uygundur."""

    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int, min_lr_ratio: float):
        self.peak_lr, self.warmup_steps = peak_lr, max(1, warmup_steps)
        self.total_steps, self.min_lr_ratio = max(1, total_steps), min_lr_ratio

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps
        progress = min(1.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.peak_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)


class SFTTrainer:
    """Ön-eğitilmiş checkpoint'i yükler ve prompt-maskeli yanıt verisiyle ince ayar yapar."""

    def __init__(self, config: SFTConfig, tokenizer):
        torch, _ = _require_torch()
        self.torch, self.config, self.tokenizer = torch, config, tokenizer
        state = torch.load(config.init_checkpoint, map_location="cpu", weights_only=False)
        self.model_config = ModelConfig(**state["model_config"])
        if self.model_config.vocab_size != tokenizer.vocab_size:
            raise ValueError("Checkpoint vocab_size'ı verilen tokenizer ile uyuşmuyor.")
        self.model = DecoderOnlyTransformer(self.model_config)
        self.model.load_state_dict(state["model"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            (no_decay if parameter.ndim < 2 or name.endswith("norm.weight") else decay).append(parameter)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=config.peak_lr, betas=(config.beta1, config.beta2))
        use_scaler = self.device.type == "cuda" and config.precision == "fp16"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        self.step = 0
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _autocast(self):
        torch = self.torch
        enabled = self.device.type == "cuda" and self.config.precision != "fp32"
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled)

    def save_checkpoint(self, tag: str = "latest") -> Path:
        torch = self.torch
        final = self.output_dir / f"sft-{tag}.pt"
        temporary = final.with_suffix(".tmp")
        torch.save({"format_version": 1, "kind": "sft", "step": self.step, "model": self.model.state_dict(),
                    "model_config": asdict(self.model_config), "sft_config": asdict(self.config)}, temporary)
        os.replace(temporary, final)
        (self.output_dir / "latest.json").write_text(f'{{"checkpoint": "{final.name}", "step": {self.step}}}\n', encoding="utf-8")
        return final

    def _make_loader(self, data_files: Sequence[str | Path], epoch: int):
        torch, DataLoader = _require_torch()
        dataset = SFTDataset(data_files, self.tokenizer, self.model_config.max_seq_len, seed=self.config.seed + epoch)
        return DataLoader(dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers,
                          collate_fn=lambda batch: collate_sft_batch(batch, self.tokenizer.pad_id))

    def fit(self, data_files: Sequence[str | Path]) -> None:
        torch, _ = _require_torch()
        steps_per_epoch = max(1, sum(1 for _ in self._make_loader(data_files, 0)) // self.config.grad_accum_steps)
        total_steps = steps_per_epoch * self.config.epochs
        schedule = LinearWarmupCosine(self.config.peak_lr, int(total_steps * self.config.warmup_ratio),
                                      total_steps, self.config.min_lr_ratio)
        self.model.train()
        started = time.perf_counter()
        for epoch in range(self.config.epochs):
            iterator = iter(self._make_loader(data_files, epoch))
            exhausted = False
            while not exhausted:
                self.optimizer.zero_grad(set_to_none=True)
                total_loss, micro_steps = 0.0, 0
                for _ in range(self.config.grad_accum_steps):
                    try:
                        inputs, labels = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    with self._autocast():
                        output = self.model(inputs, labels)
                        loss = output.loss / self.config.grad_accum_steps
                    self.scaler.scale(loss).backward()
                    total_loss += float(loss.detach())
                    micro_steps += 1
                if micro_steps == 0:
                    break
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                lr = schedule(self.step)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.step += 1
                if self.step % self.config.log_every == 0:
                    elapsed = time.perf_counter() - started
                    print(f"[sft] epoch={epoch} step={self.step}/{total_steps} loss={total_loss:.4f} "
                          f"lr={lr:.3e} elapsed={elapsed:.1f}s", flush=True)
                if self.step % self.config.checkpoint_every == 0:
                    self.save_checkpoint(f"step-{self.step:06d}")
        self.save_checkpoint("latest")
