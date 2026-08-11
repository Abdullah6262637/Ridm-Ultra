"""Geriye uyumluluk: değerlendirme `ridm_ultra.llm.runtime` altındadır."""
from .runtime.evaluation import (
    LanguageModelMetrics,
    MultipleChoiceMetrics,
    evaluate_multiple_choice_jsonl,
    evaluate_perplexity,
    score_continuation,
)

__all__ = [
    "LanguageModelMetrics",
    "MultipleChoiceMetrics",
    "evaluate_perplexity",
    "score_continuation",
    "evaluate_multiple_choice_jsonl",
]
