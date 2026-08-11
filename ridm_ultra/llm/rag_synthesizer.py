"""Extractive-Generative RAG Synthesizer for RIDM Ultra.

Fills the gap of Zero-Backprop NLG by extracting the most relevant sentences 
from the RAG context and smoothly wrapping them in dynamic grammar templates.
"""
import asyncio
import random
import re
import time
from typing import AsyncGenerator, List, Tuple

import numpy as np


class RAGSynthesizer:
    """Extracts top RAG sentences and bridges them with PCFG narrative templates."""

    OPENING_TEMPLATES = [
        "Based on the extracted data from my database:",
        "According to the retrieved documents:",
        "My analysis indicates that:",
        "Here is the information I found in my system:",
        "Based on the contextual findings, I can say that:",
        "Here is the summary extracted from my RAG indexes:"
    ]

    CLOSING_TEMPLATES = [
        "I hope this information is helpful.",
        "Do you need any further details?",
        "Would you like me to analyze this topic more deeply?",
        "Do you have any other questions based on these results?"
    ]

    def __init__(self, decoder):
        """Requires a NativeDecoder instance for cosine similarity measurement."""
        self.decoder = decoder

    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple heuristic to split text into sentences."""
        # Split on . ! ? followed by space and capital letter, but we can just split roughly
        raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw_sentences if len(s.split()) > 2]

    def _score_sentences(self, query: str, sentences: List[str]) -> List[Tuple[float, str]]:
        """Scores each sentence against the query using SVD context vectors or lexical fallback."""
        query_vec = self.decoder._get_context_vector(self.decoder._encode_prompt(query))
        q_norm = np.linalg.norm(query_vec)
        q_words = set(re.findall(r'\w+', query.lower()))

        scored = []
        for s in sentences:
            s_vec = self.decoder._get_context_vector(self.decoder._encode_prompt(s))
            s_norm = np.linalg.norm(s_vec)

            # Lexical overlap (Jaccard similarity) fallback for English Pivot
            s_words = set(re.findall(r'\w+', s.lower()))
            overlap = len(q_words.intersection(s_words))
            jaccard = overlap / (len(q_words.union(s_words)) + 1e-9)

            if s_norm == 0 or q_norm == 0:
                scored.append((jaccard, s))
                continue

            cos_sim = np.dot(query_vec, s_vec) / (q_norm * s_norm + 1e-9)

            # If SVD vectors are basically identical (due to unknown English words mapping to <UNK>)
            if cos_sim > 0.99 or cos_sim == 0.0:
                cos_sim = float(jaccard)

            scored.append((float(cos_sim), s))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    async def generate_stream(self, query: str, rag_context: str) -> AsyncGenerator[str, None]:
        """Streams a coherent response bridging the extracted RAG sentences."""
        seed = int(time.time() * 1000) + sum(ord(c) for c in query)
        rng = random.Random(seed)

        # 1. Yield an opening template
        opening = rng.choice(self.OPENING_TEMPLATES)
        for word in opening.split():
            yield word + " "
            await asyncio.sleep(0.01)

        yield "\n\n"

        # 2. Extract and score sentences from RAG Context
        # Clean the context first
        clean_context = rag_context.replace("[Retrieved Context]:", "").strip()
        sentences = self._split_into_sentences(clean_context)

        if not sentences:
            if not clean_context:
                fallback_msg = "Unfortunately, I couldn't find an exact match in my local 2-Billion-token database regarding this specific topic. I can try searching again with a different context or keyword."
                for word in fallback_msg.split():
                    yield word + " "
                    await asyncio.sleep(0.01)
            else:
                # Fallback if no sentences could be parsed but text exists
                for word in clean_context.split():
                    yield word + " "
                    await asyncio.sleep(0.01)
        else:
            scored_sentences = self._score_sentences(query, sentences)

            # Take top 2 best-matching sentences
            top_k = 2
            best_sentences = [s for score, s in scored_sentences[:top_k]]

            # Maintain their original order from the document for flow, or just yield them
            # Actually, just joining them works best
            for s in best_sentences:
                for word in s.split():
                    yield word + " "
                    await asyncio.sleep(0.01)
                yield " "

        yield "\n\n"

        # 3. Yield a closing template
        closing = rng.choice(self.CLOSING_TEMPLATES)
        for word in closing.split():
            yield word + " "
            await asyncio.sleep(0.01)
