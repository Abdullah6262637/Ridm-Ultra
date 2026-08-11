"""GPU/CPU forward-backward smoke testi ve ölçülebilir throughput raporu."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from ..model.config import ModelConfig, estimate_parameter_count
from ..model.transformer import DecoderOnlyTransformer


@dataclass(frozen=True)
class SmokeTestResult:
    device: str
    parameters: int
    steps: int
    sequence_length: int
    batch_size: int
    final_loss: float
    tokens_per_second: float
    peak_memory_bytes: int | None

    def to_dict(self):
        return asdict(self)


def run_smoke_test(config: ModelConfig, *, batch_size: int = 2, steps: int = 3, device: str = "auto") -> SmokeTestResult:
    """Kısa ama gerçek optimizer adımı: NaN, şekil ve donanım yolunu doğrular."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Smoke testi için torch kurulmalıdır.") from exc
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device auto, cpu veya cuda olmalıdır.")
    selected = "cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke testi istendi fakat CUDA bulunamadı.")
    runtime_device = torch.device(selected)
    torch.manual_seed(7)
    model = DecoderOnlyTransformer(config).to(runtime_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if selected == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime_device)
        torch.cuda.synchronize(runtime_device)
    inputs = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len), device=runtime_device)
    labels = torch.roll(inputs, shifts=-1, dims=1)
    started = time.perf_counter()
    final_loss = float("nan")
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs, labels)
        if output.loss is None or not torch.isfinite(output.loss):
            raise FloatingPointError("Smoke testinde sonlu olmayan loss oluştu.")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(output.loss.detach())
    if selected == "cuda":
        torch.cuda.synchronize(runtime_device)
    elapsed = time.perf_counter() - started
    return SmokeTestResult(selected, estimate_parameter_count(config), steps, config.max_seq_len, batch_size, final_loss,
                           batch_size * config.max_seq_len * steps / max(elapsed, 1e-9),
                           torch.cuda.max_memory_allocated(runtime_device) if selected == "cuda" else None)
