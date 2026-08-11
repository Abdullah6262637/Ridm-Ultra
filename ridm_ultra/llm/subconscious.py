import asyncio
import random
import threading
import time
from typing import Optional

import numpy as np


class SubconsciousThinker:
    """
    Background daemon that continuously connects distant concepts in the SQLite DB
    using Multi-Hop retrieval and caches the 'thought paths' as synthetic memories.
    """

    def __init__(self, native_decoder, retriever, graph_decoder, dream_mode: bool = False):
        self.native = native_decoder
        self.retriever = retriever
        self.graph = graph_decoder
        self.dream_mode = dream_mode

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.sleep_interval = 2.0 # seconds between thoughts

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[*] SubconsciousThinker: Daemon started (Dream Mode: {self.dream_mode}).")

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[*] SubconsciousThinker: Daemon stopped.")

    def _run_loop(self):
        # We need an asyncio event loop for the async generators if we use them
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self.is_running:
            try:
                if self.dream_mode:
                    loop.run_until_complete(self._dream_cycle())
                else:
                    loop.run_until_complete(self._think_cycle())
            except Exception as e:
                import logging
                logging.error(f"[!] SubconsciousThinker error: {e}")
            time.sleep(self.sleep_interval)

        loop.close()

    async def _think_cycle(self):
        # 1. Pick a random word from vocab
        if not self.native.vocab:
            return

        word_idx = random.randint(0, len(self.native.vocab) - 1)
        word = self.native.vocab[word_idx]

        # We need a meaningful word
        if len(word) < 4 or word in self.native.vocab: # In a real implementation check stop_ids
            pass

        # 2. Try to find a conceptual bridge (Multi-Hop)
        # Using deep_retrieve to simulate a random thought exploration
        query = f"what is the relationship of {word} to other concepts"
        results = self.retriever.deep_retrieve(query, hops=2, top_k=2)

        if not results:
            return



        # 3. Generate a logical connection using the graph decoder
        synthetic_thought = results[0][0] # Best context bridge

        # 4. Ingest back into memory (Self-Taught Learning)
        if hasattr(self.retriever, "ingest_synthetic_thought"):
            self.retriever.ingest_synthetic_thought(synthetic_thought)

    async def _dream_cycle(self):
        """
        Phase 4: Subconscious Dream State (Anti-Vector Generation)
        Multiplies the concept vector by -1 to find surreal mathematical opposites.
        """
        if not self.native.vocab or self.retriever.dense.doc_count == 0:
            return

        word_idx = random.randint(0, len(self.native.vocab) - 1)
        word = self.native.vocab[word_idx]

        # 1. Get original concept vector
        q_vec = self.retriever.ridm.context_vector_for([word])

        # 2. Invert it to create an Anti-Vector
        anti_vec = -q_vec
        anti_vec = anti_vec / (np.linalg.norm(anti_vec) + 1e-8)

        # 3. Find the most mathematically distant (opposite) concepts
        dense_scores = self.retriever.dense.get_scores(anti_vec)
        if len(dense_scores) == 0:
            return

        best_doc_idx = np.argmax(dense_scores)
        if dense_scores[best_doc_idx] > 0.1: # Threshold for interesting dream
            surreal_dream = self.retriever.documents[best_doc_idx]

            # Save the dream as a special synthetic thought
            if hasattr(self.retriever, "ingest_synthetic_thought"):
                self.retriever.ingest_synthetic_thought(f"[DREAM about ANTI-{word.upper()}]: {surreal_dream}")
