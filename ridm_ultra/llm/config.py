"""Geriye uyumluluk: yapılandırmalar `ridm_ultra.llm.model` altına taşındı."""
from .model.config import ModelConfig, PretrainingConfig, estimate_parameter_count

__all__ = ["ModelConfig", "PretrainingConfig", "estimate_parameter_count"]
