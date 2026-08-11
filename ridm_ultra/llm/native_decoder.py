"""Native Zero-Backprop Text Generator Decoder with N-Gram Grammar Constraints for RIDM Ultra."""
import asyncio
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Set, Tuple

import numpy as np

from attention import ClosedFormAttention
from ridm_ultra.backend import ComputeBackend

logger = logging.getLogger(__name__)

# Named constants for tuning parameters (previously magic numbers)
NGRAM_BOOST_WEIGHT = 18.0
RAG_BOOST_WEIGHT = 40.0
REPETITION_PENALTY = 6.0
INVALID_TOKEN_PENALTY = 1000.0
BEAM_REPETITION_PENALTY = 15.0
BEAM_NGRAM_BOOST_WEIGHT = 12.0
BEAM_ANCHOR_BLEND_RATIO = 0.85


class NativeDecoder:
    """Zero-Backprop Autoregressive Decoder leveraging C++ SVD projection, N-Gram Grammar Constraints & Logits Sampling."""

    _cached_bigram_next: Optional[Dict[Tuple[int, int], Counter]] = None
    _cached_tr_valid_words: Optional[Set[str]] = None

    def __init__(self, embeddings_path: Optional[str] = None, metadata_path: Optional[str] = None):
        self.backend = ComputeBackend(device="cpu")
        base_dir = Path("artifacts")
        self.embeddings_path = Path(embeddings_path) if embeddings_path else base_dir / "ridm_fineweb_embeddings.npz"
        self.metadata_path = Path(metadata_path) if metadata_path else base_dir / "ridm_fineweb_metadata.json"

        # Load English 3-Grams for abstractive generative RAG
        self.ngram_path = base_dir / "english_ngram_3gram.json"
        # Legacy Turkish word list path (kept for backward-compatible mask building)
        self.tr_words_path = base_dir / "turkish_valid_words.json"

        self.word_emb: Optional[np.ndarray] = None
        self.vocab: List[str] = []
        self.word2idx: Dict[str, int] = {}
        self.dim: int = 64
        self._is_loaded: bool = False
        self.bigram_next: Dict[Tuple[int, int], Counter] = {}
        self.stop_ids: Set[int] = set()
        self._en_chars = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Load trained SVD embeddings, Turkish vocabulary mask, and 3-Gram transition table."""
        vocab_json_path = Path("artifacts/ridm_fineweb_vocab.json")
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"NativeDecoder: embeddings artifact not found at {self.embeddings_path}."
            )

        npz = np.load(self.embeddings_path)
        self.word_emb = npz["word_emb"].astype(np.float32)

        if "vocab" in npz:
            self.vocab = list(npz["vocab"])
        elif vocab_json_path.exists():
            self.vocab = json.loads(vocab_json_path.read_text(encoding="utf-8"))
        elif self.metadata_path.exists():
            meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.vocab = meta.get("vocab", [])

        if not self.vocab:
            raise ValueError(f"NativeDecoder: no vocabulary found alongside {self.embeddings_path}.")

        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.dim = self.word_emb.shape[1]

        # English Stop words mapping
        stop_words = {"the", "and", "a", "an", "is", "are", "was", "were", "to", "of", "in", "for", "on", "with", "as", "by", "at", "it", "this", "that", "from", "or", "be"}
        self.stop_ids = {self.word2idx[w] for w in stop_words if w in self.word2idx}

        # SVO Language (English) doesn't need end-of-sentence verb boosting
        self.verb_ids = set()

        # Load 3-Gram transition table with class caching
        if NativeDecoder._cached_bigram_next is None:
            bigram_dict = {}
            if self.ngram_path.exists():
                raw_ngram = json.loads(self.ngram_path.read_text(encoding="utf-8"))
                for key, counts in raw_ngram.items():
                    w1_str, w2_str = key.split("_")
                    w1_id, w2_id = int(w1_str), int(w2_str)
                    bigram_dict[(w1_id, w2_id)] = Counter({int(k): v for k, v in counts.items()})
            NativeDecoder._cached_bigram_next = bigram_dict

        self.bigram_next = NativeDecoder._cached_bigram_next

        self._build_valid_mask()
        self.word_emb_norms = np.linalg.norm(self.word_emb, axis=1).astype(np.float32)

        # Initialize ClosedFormAttention for deep context extraction
        self.attention = ClosedFormAttention(dim=self.dim, n_heads=4)

        self._is_loaded = True
        logger.info(f"NativeDecoder: Loaded FineWeb SVD embeddings ({self.word_emb.shape[0]} words, dim={self.dim}) & {len(self.bigram_next)} 3-gram contexts.")

    def _build_valid_mask(self):
        NativeDecoder._cached_tr_valid_words = None
        if self.tr_words_path.exists():
            NativeDecoder._cached_tr_valid_words = set(json.loads(self.tr_words_path.read_text(encoding="utf-8")))
        else:
            NativeDecoder._cached_tr_valid_words = set()

        mask = []
        for w in self.vocab:
            wl = w.lower()
            clean_w = re.sub(r'[^a-zA-Z]', '', wl)
            if not clean_w:
                mask.append(False)
                continue
            if len(clean_w) == 1 and clean_w not in ('a', 'i'):
                mask.append(False)
                continue
            mask.append(True)
        self.valid_mask = np.array(mask, dtype=bool)

    def _encode_prompt(self, prompt: str) -> List[int]:
        en_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        prompt_clean = "".join(c for c in prompt if c in en_chars)
        words = prompt_clean.strip().split()
        unk_idx = self.word2idx.get("<UNK>", 0)
        return [self.word2idx.get(w, unk_idx) for w in words]

    def _get_context_vector(self, token_ids: List[int], raw_words: Optional[List[str]] = None) -> np.ndarray:
        if not token_ids and not raw_words:
            return np.zeros(self.dim, dtype=np.float32)

        if token_ids:
            vecs = self.word_emb[token_ids]
        elif raw_words:
            from subword import SubwordHasher
            hasher = SubwordHasher(dim=self.dim)
            vecs = np.array([hasher.vector(w) for w in raw_words], dtype=np.float32)
        else:
            return np.zeros(self.dim, dtype=np.float32)

        # Apply ClosedFormAttention instead of basic exponential decay
        ctx_vec, pos_weights = self.attention.forward(vecs)
        return (ctx_vec / (np.linalg.norm(ctx_vec) + 1e-8)).astype(np.float32)

    def analogical_inference(self, w1: str, w2: str, w3: Optional[str] = None) -> np.ndarray:
        """
        Performs SVD Vector Arithmetic to derive a new query vector.
        If w3 is given: w1 - w2 + w3 (e.g., King - Man + Woman = Queen)
        If w3 is None: w1 + w2 (additive combination)
        """
        unk_idx = self.word2idx.get("<UNK>", 0)
        id1 = self.word2idx.get(w1.lower(), unk_idx)
        id2 = self.word2idx.get(w2.lower(), unk_idx)

        v1 = self.word_emb[id1]
        v2 = self.word_emb[id2]

        if w3:
            id3 = self.word2idx.get(w3.lower(), unk_idx)
            v3 = self.word_emb[id3]
            out = v1 - v2 + v3
        else:
            out = v1 + v2

        return (out / (np.linalg.norm(out) + 1e-8)).astype(np.float32)

    def get_orthogonal_muse(self, target_vec: np.ndarray, muse_vec: np.ndarray, lambda_weight: float = 0.3) -> np.ndarray:
        """
        Phase 4: Geometric Creativity Engine
        Extracts the orthogonal (perpendicular) component of the muse_vec relative to target_vec.
        Adds this orthogonal component to target_vec to creatively "bend" the manifold.
        """
        p = target_vec / (np.linalg.norm(target_vec) + 1e-8)
        m = muse_vec / (np.linalg.norm(muse_vec) + 1e-8)

        # Projection of m onto p
        proj_p_m = np.dot(m, p) * p

        # Orthogonal component
        orthogonal_muse = m - proj_p_m

        # Bend the target
        creative_vec = p + (lambda_weight * orthogonal_muse)
        return (creative_vec / (np.linalg.norm(creative_vec) + 1e-8)).astype(np.float32)

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 25,
        temperature: float = 0.7,
        top_k: int = 8,
        delay_sec: float = 0.015,
    ) -> AsyncGenerator[str, None]:
        prompt_ids = self._encode_prompt(prompt)
        current_ids = list(prompt_ids) if prompt_ids else [0]

        rag_boost_ids = set()
        if "[Retrieved Context]:" in prompt:
            rag_text = prompt.split("[Retrieved Context]:")[-1]
            rag_words = [w.strip(".,!?\"'") for w in rag_text.split() if w.strip(".,!?\"'")]
            for rw in rag_words:
                if rw in self.word2idx:
                    rag_boost_ids.add(self.word2idx[rw])

        ctx_vec = self._get_context_vector(current_ids)
        rng = np.random.RandomState(int(time.time() * 1000) % (2**31 - 1))
        en_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

        for step_i in range(max_new_tokens):
            t0 = time.perf_counter()
            # 1. Compute parallel Cosine Similarity logits using C++ SIMD kernel
            row_norms = getattr(self, "word_emb_norms", None)
            if hasattr(self.backend, "cosine_matvec"):
                logits = self.backend.cosine_matvec(self.word_emb, ctx_vec, row_norms=row_norms)
            else:
                logits = self.backend.matvec(self.word_emb, ctx_vec)

            # 2. Assemble N-gram boost dict
            ngram_boost = {}
            if len(current_ids) >= 2:
                ctx_pair = (current_ids[-2], current_ids[-1])
                next_counts = self.bigram_next.get(ctx_pair)
                if next_counts:
                    total = float(sum(next_counts.values()))
                    for nxt_id, count in next_counts.items():
                        ngram_boost[nxt_id] = NGRAM_BOOST_WEIGHT * (count / total)

            if rag_boost_ids:
                for r_id in rag_boost_ids:
                    ngram_boost[r_id] = ngram_boost.get(r_id, 0.0) + RAG_BOOST_WEIGHT

            # 3. Penalty IDs
            penalty_ids = list(set(current_ids[-8:]))
            if current_ids[-1] in self.stop_ids:
                penalty_ids.extend(list(self.stop_ids))

            # 3.5. Syntax-guided POS Boost (SOV Language Rule: verbs at the end)
            if step_i >= max_new_tokens * 0.7:
                for v_id in self.verb_ids:
                    ngram_boost[v_id] = ngram_boost.get(v_id, 0.0) + 12.0 * (step_i / max_new_tokens)

            # 4. Sample next token using C++ constrained sampling
            if hasattr(self.backend, "sample_logits_constrained"):
                next_idx = self.backend.sample_logits_constrained(
                    logits, valid_mask=self.valid_mask, ngram_boost=ngram_boost,
                    penalty_ids=penalty_ids, penalty_val=6.0,
                    temperature=temperature, top_k=top_k, rng=rng
                )
            else:
                logits[~self.valid_mask] -= INVALID_TOKEN_PENALTY
                for nid, bval in ngram_boost.items():
                    if nid < len(logits):
                        logits[nid] += bval
                for pid in penalty_ids:
                    if pid < len(logits):
                        logits[pid] -= REPETITION_PENALTY
            raw_word = self.vocab[next_idx] if next_idx < len(self.vocab) else ""
            clean_word = "".join(c for c in raw_word if c in en_chars)

            if not clean_word:
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            best_cos = float(logits[next_idx]) if 0 <= next_idx < len(logits) else 0.0
            logger.debug(f"Step {step_i+1}: Generated '{clean_word}' (score={best_cos:.2f}) in {elapsed_ms:.2f}ms")

            yield clean_word + " "

            # 7. Update sliding context vector
            current_ids.append(next_idx)
            ctx_vec = self._get_context_vector(current_ids)

            await asyncio.sleep(delay_sec)

    async def beam_search_decode(
        self,
        prompt: str,
        max_new_tokens: int = 40,
        beam_width: int = 10,
        temperature: float = 0.7,
        delay_sec: float = 0.015,
        rag_vocab_ids: Optional[Set[int]] = None,
        use_ngrams: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Omni-Directional Beam-Search implementation for N-Gram + SVD Logits with Global Context Anchor."""
        prompt_ids = self._encode_prompt(prompt)
        initial_ids = list(prompt_ids) if prompt_ids else [0]

        ctx_vec_init = self._get_context_vector(initial_ids)
        # Global Anchor: This vector remembers the original prompt intent to prevent word-salad drift
        anchor_vec = np.copy(ctx_vec_init)

        # Beams: list of (cumulative_log_prob, current_sequence, current_ctx_vec)
        beams = [(0.0, initial_ids, ctx_vec_init)]

        en_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

        for step_i in range(max_new_tokens):
            t0 = time.perf_counter()
            all_candidates = []

            for score, seq, c_vec in beams:
                # GLOBAL CONTEXT ANCHOR BLENDING: 85% Sliding Context + 15% Original Prompt
                # Allows the thought to move forward while maintaining slight adherence to the topic
                blended_ctx = (c_vec * BEAM_ANCHOR_BLEND_RATIO) + (anchor_vec * (1.0 - BEAM_ANCHOR_BLEND_RATIO))
                blended_ctx = blended_ctx / (np.linalg.norm(blended_ctx) + 1e-8)

                row_norms = getattr(self, "word_emb_norms", None)
                if hasattr(self.backend, "cosine_matvec"):
                    logits = self.backend.cosine_matvec(self.word_emb, blended_ctx, row_norms=row_norms)
                else:
                    logits = self.backend.matvec(self.word_emb, blended_ctx)

                # Apply Valid Mask
                logits[~self.valid_mask] -= INVALID_TOKEN_PENALTY

                # Dynamic RAG Vocabulary Masking (Strict)
                if rag_vocab_ids:
                    rag_mask = np.zeros_like(logits, dtype=bool)
                    for r_id in rag_vocab_ids:
                        if r_id < len(rag_mask):
                            rag_mask[r_id] = True
                    logits[~rag_mask] -= INVALID_TOKEN_PENALTY

                # N-Gram Boost (Syntax Constraint Enforcer)
                if use_ngrams and len(seq) >= 2:
                    ctx_pair = (seq[-2], seq[-1])
                    next_counts = self.bigram_next.get(ctx_pair)
                    if next_counts:
                        total = float(sum(next_counts.values()))
                        for nxt_id, count in next_counts.items():
                            if nxt_id < len(logits):
                                logits[nxt_id] += BEAM_NGRAM_BOOST_WEIGHT * (count / total)

                # Penalty (Prevent looping)
                penalty_ids = list(set(seq[-20:]))
                if seq[-1] in self.stop_ids:
                    penalty_ids.extend(list(self.stop_ids))
                for pid in penalty_ids:
                    if pid < len(logits):
                        logits[pid] -= BEAM_REPETITION_PENALTY

                # POS Boost (SOV Rule) - Only push for verbs at the very end to close the sentence
                if step_i >= max_new_tokens - 3:
                    for v_id in self.verb_ids:
                        if v_id < len(logits):
                            logits[v_id] += 5.0

                # Temperature scaling
                logits = logits / max(temperature, 0.1)

                # Get Top K (beam_width) for this beam
                # Convert logits to log probabilities to maintain scale
                max_logit = np.max(logits)
                exp_logits = np.exp(logits - max_logit)
                probs = exp_logits / np.sum(exp_logits)
                log_probs = np.log(probs + 1e-10)

                top_indices = np.argsort(log_probs)[-beam_width:]

                for idx in top_indices:
                    candidate_score = score + float(log_probs[idx])
                    candidate_seq = seq + [idx]
                    candidate_ctx = self._get_context_vector(candidate_seq)
                    all_candidates.append((candidate_score, candidate_seq, candidate_ctx))

            # Keep top beam_width overall
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:beam_width]

            # Optional debug print: print(f"[BEAM SEARCH] Step {step_i+1} completed")

        best_seq = beams[0][1][len(initial_ids):]
        for t_id in best_seq:
            raw_word = self.vocab[t_id] if t_id < len(self.vocab) else ""
            clean_word = "".join(c for c in raw_word if c in en_chars)
            if clean_word:
                yield clean_word + " "
                await asyncio.sleep(delay_sec)
