"""Veri bütçesiyle eşleştirilmiş başlangıç model profilleri."""
from __future__ import annotations

from .config import ModelConfig


def available_presets() -> tuple[str, ...]:
    return ("smoke-17m", "turkish-50m", "turkish-125m")


def model_preset(name: str, vocab_size: int, max_seq_len: int = 1024, *, gradient_checkpointing: bool = False) -> ModelConfig:
    """Presetler yaklaşık Chinchilla oranıyla veri hedeflerine bağlanır.

    1B token için ``turkish-50m`` hedeflenir. 125M modelin sağlıklı ön
    eğitimi için yaklaşık 2.5B token gerekir; aynı modele 1B token vermek
    teknik olarak mümkündür fakat veri-altı-eğitim (undertraining) oluşturur.
    """
    common = {"vocab_size": vocab_size, "max_seq_len": max_seq_len,
              "gradient_checkpointing": gradient_checkpointing}
    if name == "smoke-17m":
        return ModelConfig(hidden_size=256, n_layers=8, n_heads=8, n_kv_heads=2, intermediate_size=768, **common)
    if name == "turkish-50m":
        return ModelConfig(hidden_size=512, n_layers=12, n_heads=8, n_kv_heads=2, intermediate_size=1536, **common)
    if name == "turkish-125m":
        return ModelConfig(hidden_size=768, n_layers=16, n_heads=12, n_kv_heads=4, intermediate_size=2048, **common)
    raise ValueError(f"Bilinmeyen preset: {name}. Seçenekler: {', '.join(available_presets())}")
