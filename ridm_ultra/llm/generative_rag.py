"""Abstractive Generative RAG Synthesizer for RIDM Ultra.

Replaces the Extractive RAG approach with a Zero-Backprop Beam Search
generator that uses Context Vocabulary Masking.
"""

import asyncio
import re
from typing import AsyncGenerator


class GenerativeRAGSynthesizer:
    """Uses Omni-Directional Beam Search and Dynamic RAG Vocabulary Boosting."""

    def __init__(self, decoder):
        """Requires a NativeDecoder instance."""
        self.decoder = decoder

    async def generate_stream(self, query: str, rag_context: str) -> AsyncGenerator[str, None]:
        # 1. Clean context
        clean_context = rag_context.replace("[Retrieved Context]:", "").strip()

        # 2. Extract vocabulary from the RAG context to build a dynamic mask
        rag_words = set(re.findall(r'[a-zA-Z]+', clean_context.lower()))

        rag_vocab_ids = set()
        for w in rag_words:
            if w in self.decoder.word2idx:
                rag_vocab_ids.add(self.decoder.word2idx[w])

        # Also add query words to the mask so the model can answer directly
        query_words = set(re.findall(r'[a-zA-Z]+', query.lower()))
        for w in query_words:
            if w in self.decoder.word2idx:
                rag_vocab_ids.add(self.decoder.word2idx[w])

        # Inject English Stop Words and Grammar Connectors
        stop_words = {"the", "and", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "on", "with", "as", "by", "at", "it", "this", "that", "from", "or", "be", "has", "have", "had", "not", "but", "what", "which", "who", "whom"}
        for w in stop_words:
            if w in self.decoder.word2idx:
                rag_vocab_ids.add(self.decoder.word2idx[w])

        if not rag_vocab_ids:
            # Fallback if context is totally empty
            fallback_msg = "I could not find sufficient information to generate an answer."
            for word in fallback_msg.split():
                yield word + " "
                await asyncio.sleep(0.01)
            return

        # 3. Create the prompt for the generator
        # Combine context and query so the Global Anchor Vector represents the full knowledge
        combined_prompt = f"{clean_context} {query}"

        # 4. Stream generation using Beam Search
        # We limit max_new_tokens to prevent grammatical decay
        yield "Based on my analysis, "

        async for token in self.decoder.beam_search_decode(
            prompt=combined_prompt,
            max_new_tokens=30,
            beam_width=5,
            temperature=0.6,
            rag_vocab_ids=rag_vocab_ids,
            use_ngrams=True
        ):
            yield token

        yield "."
