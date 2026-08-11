"""Semantic Router and Expert Switch for RIDM Ultra Chat Engine."""
from __future__ import annotations

import re
from typing import Dict, List

from .interfaces import BaseModelAdapter, BaseRouter
from .types import ChatMessage, ModelTier


class SemanticRouter(BaseRouter):
    """Dynamic router classifying intent and entropy to select ModelTier."""

    # Keywords triggering SOTA reasoning model tier
    REASONING_KEYWORDS = {
        "code", "python", "algorithm", "function", "class", "debug", "refactor",
        "math", "proof", "equation", "calculate", "derivative", "integral",
        "analyze", "architecture", "explain in detail", "step-by-step", "why",
        "benchmark", "compare", "optimum", "complex", "deep"
    }

    # Simple patterns suitable for Fast model tier
    FAST_PATTERNS = [
        r"^(hi|hello|hey|greetings|good morning|good evening|merhaba|selam|selamlar)\b",
        r"^(who are you|what is your name|kimsin|ne yapabilirsin)\b",
        r"^(thanks|thank you|teşekkürler|sağol)\b",
        r"^(bye|goodbye|görüşürüz)\b",
    ]

    def __init__(self, default_tier: ModelTier = ModelTier.BALANCED):
        self.default_tier = default_tier
        self._compiled_fast_patterns = [re.compile(p, re.IGNORECASE) for p in self.FAST_PATTERNS]

    def classify_intent(self, query: str) -> ModelTier:
        query_clean = query.strip().lower()

        # Check fast greeting / simple query patterns
        for pattern in self._compiled_fast_patterns:
            if pattern.search(query_clean):
                return ModelTier.FAST

        # Check reasoning keywords and token length
        words = set(re.findall(r"\w+", query_clean))
        overlap = words.intersection(self.REASONING_KEYWORDS)

        if overlap or len(words) > 40:
            return ModelTier.REASONING

        return self.default_tier

    def route_query(
        self,
        query: str,
        context_messages: List[ChatMessage],
        available_adapters: Dict[ModelTier, BaseModelAdapter],
    ) -> BaseModelAdapter:
        selected_tier = self.classify_intent(query)

        # Fallback hierarchy if exact tier adapter is missing
        if selected_tier in available_adapters:
            return available_adapters[selected_tier]
        if ModelTier.BALANCED in available_adapters:
            return available_adapters[ModelTier.BALANCED]
        if ModelTier.FAST in available_adapters:
            return available_adapters[ModelTier.FAST]
        if ModelTier.REASONING in available_adapters:
            return available_adapters[ModelTier.REASONING]

        # Final fallback to any registered adapter
        if available_adapters:
            return next(iter(available_adapters.values()))

        raise RuntimeError("No model adapters available in router.")
