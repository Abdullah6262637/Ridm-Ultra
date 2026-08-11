"""Model Adapters for RIDM Ultra Chat Engine (100% Native Offline SVD Execution)."""
from __future__ import annotations

import logging
from typing import AsyncGenerator, Dict, List

from ridm_ultra.llm.native_decoder import NativeDecoder

from .interfaces import BaseModelAdapter
from .types import ChatMessage, ChatResponseChunk, MessageRole, ModelTier, TokenUsage

logger = logging.getLogger(__name__)


class ConversationTracker:
    def __init__(self, decay_rate=0.8):
        self.decay_rate = decay_rate
        self.global_state = None

    def update(self, prompt_vec):
        if self.global_state is None:
            self.global_state = prompt_vec.copy()
        else:
            self.global_state = self.decay_rate * self.global_state + (1 - self.decay_rate) * prompt_vec

    def get_state(self):
        return self.global_state

class LocalTransformerAdapter(BaseModelAdapter):
    """100% Native Zero-Backprop SVD Autoregressive Decoder Adapter."""

    def __init__(
        self,
        hybrid_lm=None,
        model_name: str = "ridm-ultra-native-svd",
        tier: ModelTier = ModelTier.FAST,
    ):
        self._hybrid_lm = hybrid_lm
        self._model_name = model_name
        self._tier = tier
        self.decoder = NativeDecoder()
        self._tracker = ConversationTracker()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tier(self) -> ModelTier:
        return self._tier

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[ChatResponseChunk, None]:
        last_msg = messages[-1].content if messages else ""
        raw_user_prompt = last_msg.split("[Retrieved Context]:")[0].strip()

        # 0 & 1: Bypassed TurkishTypoCorrector and Math Evaluator for English Pivot
        user_prompt = raw_user_prompt

        # 2. Stateful Conversation Update
        prompt_vec_raw = self.decoder._get_context_vector(self.decoder._encode_prompt(user_prompt))
        self._tracker.update(prompt_vec_raw)


        has_rag = "[Retrieved Context]:" in last_msg

        # 3. Formulate generation prompt and stream via Omni-RAG Synthesizer
        # We completely bypass the NativeDecoder's word-by-word generation (which causes word-salad)
        # and instead rely on Extractive RAG generation for EVERYTHING.

        rag_body = ""
        if has_rag:
            rag_body = last_msg.split("[Retrieved Context]:")[-1].strip()

        from ridm_ultra.llm.graph_decoder import GraphDecoder
        graph_synth = GraphDecoder(self.decoder)

        token_count = 0
        prompt_tok_len = len(user_prompt.split())

        # Detect greetings/chitchat first, even if RAG context was spuriously retrieved
        greeting_response = self._handle_conversational(user_prompt)
        if greeting_response:
            for word in greeting_response.split():
                token_count += 1
                yield ChatResponseChunk(
                    delta=word + " ",
                    finish_reason=None,
                    model_name=self.model_name,
                )
            yield ChatResponseChunk(
                delta="",
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=prompt_tok_len, completion_tokens=token_count, total_tokens=prompt_tok_len + token_count),
                model_name=self.model_name,
            )
            return

        async for word_chunk in graph_synth.generate_stream(user_prompt, rag_body):
            token_count += 1
            yield ChatResponseChunk(
                delta=word_chunk,
                finish_reason=None,
                model_name=self.model_name,
            )

        yield ChatResponseChunk(
            delta="",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=prompt_tok_len, completion_tokens=token_count, total_tokens=prompt_tok_len + token_count),
            model_name=self.model_name,
        )
        return

    def _handle_conversational(self, user_prompt: str) -> str:
        """Handle greetings, chitchat, and meta-questions without RAG context.
        Returns a response string, or empty string if this is not conversational."""
        prompt_lower = user_prompt.lower().strip().rstrip("!?.,")
        
        # Greeting patterns (EN + TR)
        greetings = {
            "hello", "hi", "hey", "howdy", "greetings", "good morning",
            "good afternoon", "good evening", "good night", "whats up",
            "what's up", "sup", "yo",
            "merhaba", "selam", "selamlar", "naber", "nasilsin",
            "nasılsın", "günaydın", "iyi akşamlar", "iyi geceler",
        }
        
        # Identity / meta questions
        identity_patterns = [
            "who are you", "what are you", "what is your name",
            "sen kimsin", "adın ne", "sen nesin", "ne yapabilirsin",
            "what can you do", "tell me about yourself",
        ]
        
        # Farewell patterns
        farewells = {
            "bye", "goodbye", "see you", "later", "take care",
            "hoşçakal", "görüşürüz", "bay bay", "güle güle",
        }
        
        # Check greetings
        for g in greetings:
            if g in prompt_lower or prompt_lower == g:
                return ("Hello! I'm RIDM Ultra, a geometric intelligence engine powered by "
                        "SVD embeddings and closed-form computation. I can answer questions about "
                        "science, history, technology, geography, and much more. Ask me anything!")
        
        # Check identity questions
        for pat in identity_patterns:
            if pat in prompt_lower:
                return ("I am RIDM Ultra v6, a next-generation AI system built on Singular Value "
                        "Decomposition (SVD) and zero-backpropagation architecture. Unlike traditional "
                        "LLMs, I use geometric vector spaces, graph-based sentence synthesis, and "
                        "Monte Carlo hypothesis generation to answer your questions. "
                        "I work 100% offline with no cloud dependency.")
        
        # Check farewells
        for f in farewells:
            if f in prompt_lower:
                return "Goodbye! Feel free to come back anytime. I'm always here to help."
        
        # Check simple thank-you
        if any(w in prompt_lower for w in ["thank", "thanks", "teşekkür", "sağol", "eyvallah"]):
            return "You're welcome! Let me know if you have any other questions."
        
        # Not a conversational message — return empty to proceed with graph_decoder
        return ""

    async def generate(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatMessage:
        full_content = []
        finish_usage = None
        async for chunk in self.generate_stream(messages, temperature=temperature, max_tokens=max_tokens):
            full_content.append(chunk.delta)
            if chunk.usage:
                finish_usage = chunk.usage

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="".join(full_content),
            metadata={"model": self.model_name, "usage": finish_usage.__dict__ if finish_usage else {}},
        )


class AdapterFactory:
    """Dependency Injection Factory for Model Adapters."""

    def __init__(self):
        self._registry: Dict[ModelTier, BaseModelAdapter] = {}

    def register(self, tier: ModelTier, adapter: BaseModelAdapter) -> None:
        self._registry[tier] = adapter

    def get(self, tier: ModelTier) -> BaseModelAdapter:
        if tier not in self._registry:
            raise KeyError(f"No adapter registered for ModelTier '{tier.value}'.")
        return self._registry[tier]

    def all_adapters(self) -> Dict[ModelTier, BaseModelAdapter]:
        return dict(self._registry)
