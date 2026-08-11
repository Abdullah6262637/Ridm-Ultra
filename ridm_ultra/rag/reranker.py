from typing import Any, Dict, List, Tuple

import numpy as np


class RAGCrossEncoderReranker:
    """A neural reranker adapted for RAG retrieval results."""

    def __init__(self, n_features: int = 3, hidden: int = 16, seed: int = 42):
        # We replace random weights with logical static weights to ensure the reranker
        # doesn't shuffle context randomly before fine-tuning.
        self.W1 = np.zeros((hidden, n_features))
        self.W1[0, 0] = 1.0  # Pass cosine_sim to neuron 0
        self.W1[1, 1] = 1.0  # Pass coverage to neuron 1
        self.W1[2, 2] = 0.1  # Pass len_ratio to neuron 2

        self.b1 = np.zeros(hidden)

        self.W2 = np.zeros(hidden)
        self.W2[0] = 5.0   # Weight for cosine_sim
        self.W2[1] = 3.0   # Weight for coverage
        self.W2[2] = 0.5   # Weight for len_ratio
        self.b2 = -1.0     # Sigmoid bias

        # Default normalization stats
        self.mu = np.zeros(n_features)
        self.sigma = np.ones(n_features)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        h = np.tanh(X @ self.W1.T + self.b1)
        z = h @ self.W2 + self.b2
        p = 1.0 / (1.0 + np.exp(-z))
        return p

    def extract_features(self, query: str, doc: str, ridm) -> np.ndarray:
        """Extracts features: [Dense Cosine Sim, Query Coverage, Length Ratio]"""
        # 1. Dense Cosine
        q_vec = ridm.context_vector_for(query.split())
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)

        d_vec = ridm.context_vector_for(doc.split())
        d_vec = d_vec / (np.linalg.norm(d_vec) + 1e-8)

        cosine_sim = float(q_vec @ d_vec)

        STOPWORDS = {
            "a", "an", "and", "are", "as", "at", "be", "but", "by",
            "for", "if", "in", "into", "is", "it", "no", "not", "of",
            "on", "or", "such", "that", "the", "their", "then", "there", "these",
            "they", "this", "to", "was", "will", "with", "from", "which", "were", "how", "what", "why", "when", "does", "did"
        }

        # 2. Query Coverage (how many unique non-stopword query words in doc)
        q_words = set([w.lower() for w in query.split() if w.lower() not in STOPWORDS])
        d_words = set([w.lower() for w in doc.split() if w.lower() not in STOPWORDS])
        coverage = len(q_words.intersection(d_words)) / max(1, len(q_words))

        # 3. Length Ratio (Doc len vs avg 50 words)
        len_ratio = len(doc.split()) / 50.0

        return np.array([cosine_sim, coverage, len_ratio])

    def rerank(self, query: str, candidates: List[Tuple[str, float, Dict[str, Any]]], ridm, top_n: int = 3) -> List[Tuple[str, float, Dict[str, Any]]]:
        if not candidates:
            return []

        feats = []
        for doc, _, _ in candidates:
            feats.append(self.extract_features(query, doc, ridm))

        X = np.array(feats)
        Xn = (X - self.mu) / self.sigma

        scores = self._forward(Xn)
        order = np.argsort(scores)[::-1]

        reranked = []
        for i in order[:top_n]:
            # Update the score to the neural reranker score
            reranked.append((candidates[i][0], float(scores[i]), candidates[i][2]))

        return reranked
