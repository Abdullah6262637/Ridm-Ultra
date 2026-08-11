"""Doğrulanmış model ve ön-eğitim yapılandırmaları."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 1024
    hidden_size: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int | None = None
    intermediate_size: int | None = None
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    gradient_checkpointing: bool = False

    def __post_init__(self):
        kv_heads = self.n_kv_heads or self.n_heads
        intermediate = self.intermediate_size or int(8 * self.hidden_size / 3)
        # SwiGLU için Tensor Core dostu hizalama.
        intermediate = ((intermediate + 255) // 256) * 256
        if self.vocab_size <= 0 or self.max_seq_len <= 1 or self.n_layers <= 0:
            raise ValueError("vocab_size, max_seq_len ve n_layers pozitif olmalıdır.")
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size, n_heads'e tam bölünmelidir.")
        if self.n_heads % kv_heads != 0:
            raise ValueError("n_heads, n_kv_heads'e tam bölünmelidir.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout [0, 1) aralığında olmalıdır.")
        object.__setattr__(self, "n_kv_heads", kv_heads)
        object.__setattr__(self, "intermediate_size", intermediate)


@dataclass(frozen=True)
class PretrainingConfig:
    output_dir: str | Path
    batch_size: int = 4
    grad_accum_steps: int = 8
    max_steps: int = 10_000
    warmup_tokens: int = 20_000_000
    total_tokens: int = 1_000_000_000
    peak_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0
    precision: str = "bf16"
    num_workers: int = 2
    log_every: int = 10
    checkpoint_every: int = 500
    seed: int = 42
    use_torch_compile: bool = False

    def __post_init__(self):
        if self.batch_size <= 0 or self.grad_accum_steps <= 0 or self.max_steps <= 0:
            raise ValueError("batch_size, grad_accum_steps ve max_steps pozitif olmalıdır.")
        if self.total_tokens <= 0 or self.warmup_tokens < 0 or self.warmup_tokens >= self.total_tokens:
            raise ValueError("warmup_tokens toplam token sayısından küçük olmalıdır.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision fp32, fp16 veya bf16 olmalıdır.")


def estimate_parameter_count(config: ModelConfig) -> int:
    """Bias'sız mimarinin bağlı embedding dahil tam parametre hesabı.

    Bu formül, GPU bütçesi ayrılmadan önce model boyutunu denetlenebilir hale
    getirir; `tie_embeddings=False` ise çıkış projeksiyonu ayrıca eklenir.
    """
    h, kv = config.hidden_size, config.hidden_size // config.n_heads * config.n_kv_heads
    attention = h * h + 2 * h * kv + h * h
    swiglu = 3 * h * config.intermediate_size
    norms = 2 * h
    embeddings = config.vocab_size * h * (1 if config.tie_embeddings else 2)
    return embeddings + config.n_layers * (attention + swiglu + norms) + h
