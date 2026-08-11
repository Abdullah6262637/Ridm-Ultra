import math
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import re

_WORD_PATTERN = re.compile(r'\w+')
_STOP_WORDS = frozenset({"the", "and", "of", "to", "in", "a", "is", "that", "it", "was", "for", "on", "are", "as", "with", "they", "be", "at", "one", "have", "this", "from", "or", "had", "by", "not", "word", "but", "what", "some", "we", "can", "out", "other", "were", "all", "there", "when", "up", "use", "your", "how", "said", "an", "each", "she", "which", "their", "will"})


class BM25SparseIndex:
    """A pure Python implementation of Okapi BM25 for sparse retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0

        # Inverted index: term -> list of (doc_idx, term_freq)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    def _tokenize(self, text: str) -> List[str]:
        # Simple lowercase and alphanumeric splitting with stop words
        return [w for w in _WORD_PATTERN.findall(text.lower()) if len(w) > 2 and w not in _STOP_WORDS]

    def add_documents(self, documents: List[str]):
        """Adds a list of documents to the index."""
        start_idx = self.doc_count
        for i, text in enumerate(documents):
            tokens = self._tokenize(text)
            self.doc_len.append(len(tokens))

            term_counts = defaultdict(int)
            for token in tokens:
                term_counts[token] += 1

            for term, freq in term_counts.items():
                self.inverted_index[term].append((start_idx + i, freq))
                self.doc_freqs[term] += 1

            self.doc_count += 1

        if self.doc_count > 0:
            self.avg_doc_len = sum(self.doc_len) / self.doc_count

    def get_scores(self, query: str) -> np.ndarray:
        """Computes BM25 scores for all documents given a query."""
        scores = np.zeros(self.doc_count, dtype=np.float32)
        if self.doc_count == 0:
            return scores

        query_tokens = self._tokenize(query)
        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            df = self.doc_freqs[term]
            # IDF calculation (BM25 variant)
            idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

            for doc_idx, freq in self.inverted_index[term]:
                doc_len_ratio = self.doc_len[doc_idx] / self.avg_doc_len
                # TF calculation
                tf = (freq * (self.k1 + 1)) / (freq + self.k1 * (1.0 - self.b + self.b * doc_len_ratio))
                scores[doc_idx] += idf * tf

        return scores

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'k1': self.k1, 'b': self.b,
                'doc_freqs': dict(self.doc_freqs),
                'doc_len': self.doc_len,
                'avg_doc_len': self.avg_doc_len,
                'doc_count': self.doc_count,
                'inverted_index': dict(self.inverted_index)
            }, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.k1 = data['k1']
            self.b = data['b']
            self.doc_freqs = defaultdict(int, data['doc_freqs'])
            self.doc_len = data['doc_len']
            self.avg_doc_len = data['avg_doc_len']
            self.doc_count = data['doc_count']
            self.inverted_index = defaultdict(list, data['inverted_index'])

class DenseVectorIndex:
    """Numpy-backed dense vector index using mmap for large-scale offline lookup."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.vectors = None
        self.doc_count = 0

    def add_vectors(self, vectors: np.ndarray):
        """Adds dense vectors (shape: N, dim)"""
        if self.vectors is None:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        self.doc_count = self.vectors.shape[0]

    def get_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """Computes Cosine Similarity scores. Assumes query_vec and stored vectors are normalized."""
        if self.doc_count == 0 or self.vectors is None:
            return np.zeros(0, dtype=np.float32)

        # Standard matrix-vector dot product (cosine similarity since normalized)
        return self.vectors @ query_vec.astype(np.float32)

    def save(self, filepath: str):
        np.save(filepath, self.vectors)

    def load(self, filepath: str, mmap_mode: str = 'r'):
        self.vectors = np.load(filepath, mmap_mode=mmap_mode)
        self.doc_count = self.vectors.shape[0]
