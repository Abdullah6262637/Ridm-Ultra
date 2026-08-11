"""Gerçek decoder-only Transformer LLM eğitim bileşenleri.

Bu paket isteğe bağlı ``torch`` bağımlılığı gerektirir. Temel RIDM paketi,
PyTorch kurulu olmadığında da kullanılmaya devam eder.
"""

try:
    from .data import (
        HuggingFaceTokenizer,
        JSONLDocumentStream,
        PackedCausalDataset,
        QualityPolicy,
        SQLiteDeduplicator,
        prepare_corpus,
        train_byte_bpe,
    )
    from .model import (
        CausalLMOutput,
        DecoderOnlyTransformer,
        ModelConfig,
        PretrainingConfig,
        available_presets,
        estimate_parameter_count,
        model_preset,
    )
    from .runtime import Pretrainer, evaluate_multiple_choice_jsonl, evaluate_perplexity, run_smoke_test
except (RuntimeError, ImportError):
    # PyTorch/CUDA not available — LLM training features disabled, NativeDecoder still works
    pass

from .native_decoder import NativeDecoder

__all__ = [
    "NativeDecoder",
    "HuggingFaceTokenizer",
    "JSONLDocumentStream",
    "PackedCausalDataset",
    "QualityPolicy",
    "SQLiteDeduplicator",
    "prepare_corpus",
    "train_byte_bpe",
    "CausalLMOutput",
    "DecoderOnlyTransformer",
    "ModelConfig",
    "PretrainingConfig",
    "available_presets",
    "estimate_parameter_count",
    "model_preset",
    "Pretrainer",
    "evaluate_multiple_choice_jsonl",
    "evaluate_perplexity",
    "run_smoke_test",
]
