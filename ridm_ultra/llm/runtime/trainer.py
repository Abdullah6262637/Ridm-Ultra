"""Token-bazlı öğrenme oranı, AMP, DDP ve atomik checkpoint ile ön-eğitim."""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

from ..model.config import ModelConfig, PretrainingConfig


def _require_torch():
    try:
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel
        from torch.utils.data import DataLoader
        return torch, dist, DistributedDataParallel, DataLoader
    except ImportError as exc:
        raise RuntimeError("LLM eğitimi için `pip install .[llm]` ile torch kurulmalıdır.") from exc


class TokenCosineSchedule:
    """Adım değil işlenen token sayısına bağlı warmup + cosine decay."""
    def __init__(self, peak_lr: float, warmup_tokens: int, total_tokens: int, min_lr_ratio: float):
        self.peak_lr, self.warmup_tokens = peak_lr, warmup_tokens
        self.total_tokens, self.min_lr_ratio = total_tokens, min_lr_ratio

    def __call__(self, tokens: int) -> float:
        if tokens < self.warmup_tokens:
            return self.peak_lr * tokens / max(1, self.warmup_tokens)
        progress = min(1.0, (tokens - self.warmup_tokens) / max(1, self.total_tokens - self.warmup_tokens))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.peak_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)


class Pretrainer:
    def __init__(self, model, model_config: ModelConfig, config: PretrainingConfig):
        torch, dist, DDP, _ = _require_torch()
        self.torch, self.dist, self.DDP = torch, dist, DDP
        self.model_config, self.config = model_config, config
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1
        if self.distributed and not dist.is_initialized():
            if not torch.cuda.is_available():
                raise RuntimeError("Çok süreçli LLM eğitimi için CUDA/NCCL gerekir.")
            dist.init_process_group("nccl")
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        torch.manual_seed(config.seed + self.rank)
        random.seed(config.seed + self.rank)
        model = model.to(self.device)
        if config.use_torch_compile and hasattr(torch, "compile"):
            model = torch.compile(model)
        self.model = DDP(model, device_ids=[self.local_rank]) if self.distributed else model
        raw_model = self._raw_model
        decay, no_decay = [], []
        for name, parameter in raw_model.named_parameters():
            if not parameter.requires_grad:
                continue
            (no_decay if parameter.ndim < 2 or name.endswith("norm.weight") else decay).append(parameter)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=config.peak_lr, betas=(config.beta1, config.beta2), fused=self.device.type == "cuda",
        )
        self.schedule = TokenCosineSchedule(config.peak_lr, config.warmup_tokens, config.total_tokens, config.min_lr_ratio)
        use_scaler = self.device.type == "cuda" and config.precision == "fp16"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        self.step, self.tokens_seen = 0, 0
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _raw_model(self):
        return self.model.module if self.distributed else self.model

    @property
    def _checkpoint_model(self):
        # torch.compile'ın _orig_mod ön eki checkpoint formatına sızmasın.
        return getattr(self._raw_model, "_orig_mod", self._raw_model)

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def _autocast(self):
        torch = self.torch
        enabled = self.device.type == "cuda" and self.config.precision != "fp32"
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled)

    def _set_lr(self) -> float:
        lr = self.schedule(self.tokens_seen)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def save_checkpoint(self, tag: str = "latest") -> Path | None:
        if not self.is_primary:
            return None
        torch = self.torch
        final = self.output_dir / f"checkpoint-{tag}.pt"
        temporary = final.with_suffix(".tmp")
        state = {
            "format_version": 1, "step": self.step, "tokens_seen": self.tokens_seen,
            "model": self._checkpoint_model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "model_config": asdict(self.model_config), "training_config": asdict(self.config),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(state, temporary)
        os.replace(temporary, final)  # Yarım checkpoint hiçbir zaman geçerli isim almaz.
        (self.output_dir / "latest.json").write_text(json.dumps({"checkpoint": final.name, "step": self.step,
            "tokens_seen": self.tokens_seen}, indent=2), encoding="utf-8")
        return final

    def resume(self, path: str | Path) -> None:
        state = self.torch.load(path, map_location=self.device, weights_only=False)
        expected = asdict(self.model_config)
        if state.get("model_config") != expected:
            raise ValueError("Checkpoint model yapılandırması aktif modelle uyuşmuyor.")
        self._checkpoint_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step, self.tokens_seen = int(state["step"]), int(state["tokens_seen"])
        self.torch.set_rng_state(state["torch_rng"])
        if self.device.type == "cuda" and state.get("cuda_rng") is not None:
            self.torch.cuda.set_rng_state_all(state["cuda_rng"])

    def fit(self, dataset) -> None:
        torch, _, _, DataLoader = _require_torch()
        loader = DataLoader(dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers,
                            pin_memory=self.device.type == "cuda", persistent_workers=self.config.num_workers > 0)
        iterator = iter(loader)
        self.model.train()
        started = time.perf_counter()
        while self.step < self.config.max_steps and self.tokens_seen < self.config.total_tokens:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            micro_tokens = 0
            for _ in range(self.config.grad_accum_steps):
                try:
                    packed = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    packed = next(iterator)
                packed = packed.to(self.device, non_blocking=True)
                inputs, labels = packed[:, :-1], packed[:, 1:]
                with self._autocast():
                    output = self.model(inputs, labels)
                    loss = output.loss / self.config.grad_accum_steps
                self.scaler.scale(loss).backward()
                total_loss += float(loss.detach())
                micro_tokens += labels.numel()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self._raw_model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.step += 1
            self.tokens_seen += micro_tokens * self.world_size
            lr = self._set_lr()
            if self.step % self.config.log_every == 0 and self.is_primary:
                elapsed = time.perf_counter() - started
                rate = self.tokens_seen / max(elapsed, 1e-9)
                print(f"step={self.step} loss={total_loss:.4f} lr={lr:.3e} tokens={self.tokens_seen} tok/s={rate:.0f}", flush=True)
            if self.step % self.config.checkpoint_every == 0:
                self.save_checkpoint(f"step-{self.step:07d}")
        self.save_checkpoint("latest")
