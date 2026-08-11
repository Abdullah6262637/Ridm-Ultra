"""Doğrudan Tercih Optimizasyonu (DPO): dondurulmuş referans modele göre ödül farkını öğrenir.

Rafailov ve ark. (2023) DPO formülasyonunu izler: ayrı bir ödül modeli
eğitmek yerine, politika ile SFT sonrası dondurulan referans modelin
tercih edilen/edilmeyen yanıtlara verdiği log-olasılık farkları üzerinden
doğrudan optimize eder.
"""
from __future__ import annotations

import copy
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from ..model.config import ModelConfig
from ..model.transformer import DecoderOnlyTransformer
from .data import PreferenceDataset, collate_preference_batch
from .sft import LinearWarmupCosine


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        return torch, F, DataLoader
    except ImportError as exc:
        raise RuntimeError("DPO eğitimi için `pip install .[llm]` ile torch kurulmalıdır.") from exc


@dataclass(frozen=True)
class DPOConfig:
    output_dir: str | Path
    init_checkpoint: str | Path
    beta: float = 0.1
    batch_size: int = 2
    grad_accum_steps: int = 8
    epochs: int = 1
    peak_lr: float = 5e-6
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    precision: str = "bf16"
    num_workers: int = 0
    log_every: int = 10
    checkpoint_every: int = 200
    seed: int = 42

    def __post_init__(self):
        if self.batch_size <= 0 or self.grad_accum_steps <= 0 or self.epochs <= 0:
            raise ValueError("batch_size, grad_accum_steps ve epochs pozitif olmalıdır.")
        if self.beta <= 0:
            raise ValueError("beta pozitif olmalıdır.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision fp32, fp16 veya bf16 olmalıdır.")


def _sequence_logprobs(model, inputs, labels):
    """Etiketi -100 olmayan pozisyonlardaki token log-olasılıklarının dizi başına toplamı."""
    torch, F, _ = _require_torch()
    logits = model(inputs).logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    mask = labels != -100
    safe_labels = labels.clamp(min=0)
    token_logp = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta: float):
    """DPO kaybını ve tercih doğruluğunu (chosen > rejected oranı) döndürür."""
    torch, F, _ = _require_torch()
    policy_diff = policy_chosen - policy_rejected
    ref_diff = ref_chosen - ref_rejected
    logits = beta * (policy_diff - ref_diff)
    loss = -F.logsigmoid(logits).mean()
    accuracy = (logits > 0).float().mean()
    return loss, accuracy


class DPOTrainer:
    """Politika modelini, SFT checkpoint'inden türetilen dondurulmuş referansa göre optimize eder."""

    def __init__(self, config: DPOConfig, tokenizer):
        torch, _, _ = _require_torch()
        self.torch, self.config, self.tokenizer = torch, config, tokenizer
        state = torch.load(config.init_checkpoint, map_location="cpu", weights_only=False)
        self.model_config = ModelConfig(**state["model_config"])
        if self.model_config.vocab_size != tokenizer.vocab_size:
            raise ValueError("Checkpoint vocab_size'ı verilen tokenizer ile uyuşmuyor.")
        self.policy = DecoderOnlyTransformer(self.model_config)
        self.policy.load_state_dict(state["model"])
        self.reference = copy.deepcopy(self.policy)
        self.reference.requires_grad_(False)
        self.reference.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.reference.to(self.device)
        torch.manual_seed(config.seed)
        decay, no_decay = [], []
        for name, parameter in self.policy.named_parameters():
            (no_decay if parameter.ndim < 2 or name.endswith("norm.weight") else decay).append(parameter)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=config.peak_lr)
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
        final = self.output_dir / f"dpo-{tag}.pt"
        temporary = final.with_suffix(".tmp")
        torch.save({"format_version": 1, "kind": "dpo", "step": self.step, "model": self.policy.state_dict(),
                    "model_config": asdict(self.model_config), "dpo_config": asdict(self.config)}, temporary)
        os.replace(temporary, final)
        return final

    def _make_loader(self, data_files: Sequence[str | Path], epoch: int):
        torch, _, DataLoader = _require_torch()
        dataset = PreferenceDataset(data_files, self.tokenizer, self.model_config.max_seq_len, seed=self.config.seed + epoch)
        return DataLoader(dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers,
                          collate_fn=lambda batch: collate_preference_batch(batch, self.tokenizer.pad_id))

    def fit(self, data_files: Sequence[str | Path]) -> None:
        torch, _, _ = _require_torch()
        steps_per_epoch = max(1, sum(1 for _ in self._make_loader(data_files, 0)) // self.config.grad_accum_steps)
        total_steps = steps_per_epoch * self.config.epochs
        schedule = LinearWarmupCosine(self.config.peak_lr, int(total_steps * self.config.warmup_ratio),
                                      total_steps, self.config.min_lr_ratio)
        self.policy.train()
        started = time.perf_counter()
        for epoch in range(self.config.epochs):
            iterator = iter(self._make_loader(data_files, epoch))
            exhausted = False
            while not exhausted:
                self.optimizer.zero_grad(set_to_none=True)
                total_loss, total_accuracy, micro_steps = 0.0, 0.0, 0
                for _ in range(self.config.grad_accum_steps):
                    try:
                        chosen_inputs, chosen_labels, rejected_inputs, rejected_labels = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    chosen_inputs, chosen_labels = chosen_inputs.to(self.device), chosen_labels.to(self.device)
                    rejected_inputs, rejected_labels = rejected_inputs.to(self.device), rejected_labels.to(self.device)
                    with self._autocast():
                        policy_chosen = _sequence_logprobs(self.policy, chosen_inputs, chosen_labels)
                        policy_rejected = _sequence_logprobs(self.policy, rejected_inputs, rejected_labels)
                        with torch.no_grad():
                            ref_chosen = _sequence_logprobs(self.reference, chosen_inputs, chosen_labels)
                            ref_rejected = _sequence_logprobs(self.reference, rejected_inputs, rejected_labels)
                        loss, accuracy = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, self.config.beta)
                        loss = loss / self.config.grad_accum_steps
                    self.scaler.scale(loss).backward()
                    total_loss += float(loss.detach()) * self.config.grad_accum_steps
                    total_accuracy += float(accuracy.detach())
                    micro_steps += 1
                if micro_steps == 0:
                    break
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip_norm)
                lr = schedule(self.step)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.step += 1
                if self.step % self.config.log_every == 0:
                    elapsed = time.perf_counter() - started
                    print(f"[dpo] epoch={epoch} step={self.step}/{total_steps} loss={total_loss / micro_steps:.4f} "
                          f"tercih_dogrulugu={total_accuracy / micro_steps:.3f} lr={lr:.3e} elapsed={elapsed:.1f}s", flush=True)
                if self.step % self.config.checkpoint_every == 0:
                    self.save_checkpoint(f"step-{self.step:06d}")
        self.save_checkpoint("latest")
