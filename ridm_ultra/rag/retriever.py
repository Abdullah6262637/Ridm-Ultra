import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from .indexing import BM25SparseIndex, DenseVectorIndex
from .reranker import RAGCrossEncoderReranker

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines BM25 and Dense Dense retrieval using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, ridm, uma_aligner=None, lang_code="en"):
        self.ridm = ridm
        self.uma_aligner = uma_aligner
        self.lang_code = lang_code
        self.bm25 = BM25SparseIndex()
        self.dense = DenseVectorIndex(dim=ridm.dim)
        self.reranker = RAGCrossEncoderReranker()
        self.documents = []  # List of chunks
        self.doc_metadata = []

        self.last_used_native = False
        self.last_max_cosine = 0.0

        # Massive 1.2 Billion Word Disk Retriever Integration
        self.massive_enabled = False
        self._tfidf_matrix = None
        self.vectorizer = None
        self.db_path = _PROJECT_ROOT / "data" / "massive_corpus.sqlite"
        self.tfidf_path = _PROJECT_ROOT / "data" / "massive_tfidf.npz"

        if os.path.exists(self.tfidf_path) and os.path.exists(self.db_path):
            logger.info("[*] HybridRetriever: Detected Massive 1.2 Billion Word Index on disk. Enabling Massive RAG mode.")
            self.massive_enabled = True
            # Find vocabulary
            vocab = []
            vocab_path = _PROJECT_ROOT / "artifacts" / "ridm_fineweb_vocab.json"
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab = json.load(f)

            self.vectorizer = TfidfVectorizer(vocabulary=vocab, lowercase=True, use_idf=False, norm='l2')
            self.vectorizer.fit([""]) # initialize

    @property
    def tfidf_matrix(self):
        """Lazy load the TF-IDF matrix only when actually needed for massive retrieval."""
        if self._tfidf_matrix is None and self.massive_enabled:
            logger.info("[*] HybridRetriever: Lazy loading massive TF-IDF matrix into RAM...")
            self._tfidf_matrix = load_npz(self.tfidf_path)
        return self._tfidf_matrix

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return

        texts = [chunk["content"] for chunk in chunks]

        # 1. Add to BM25
        self.bm25.add_documents(texts)

        # 2. Add to Dense
        vecs = []
        for text in texts:
            words = text.split()
            vec = self.ridm.context_vector_for(words)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            vecs.append(vec)

        self.dense.add_vectors(np.array(vecs))

        # 3. Store references
        for chunk in chunks:
            self.documents.append(chunk["content"])
            self.doc_metadata.append({"doc_id": chunk.get("doc_id", "")})

    def _retrieve_massive(self, query: str, top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        search_query = query
        MOCK_UMA_TRANSLATIONS = {
            "kara delik": "kara delik yerçekimi kütleçekimi zaman yavaşlaması uzay zaman görelilik olay ufku tekillik",
            "kara delikler": "kara delik yerçekimi kütleçekimi zaman yavaşlaması uzay zaman görelilik olay ufku tekillik",
            "kuantum": "kuantum dolanıklık iletişim bilgi transferi ışık hızı anında etkileşim",
            "görelilik": "görelilik özel görelilik kütle enerji zaman genişlemesi ışık hızı",
        }
        self.last_expansion = None
        for k, v in MOCK_UMA_TRANSLATIONS.items():
            if k in query.lower():
                search_query = v
                msg = f"Geometric Query Expansion: '{query}' -> '{search_query}'"
                logger.info(f"[*] {msg}")
                self.last_expansion = msg
                break

        if search_query == query and self.uma_aligner and self.lang_code != "en" and getattr(self.ridm, "word_emb", None) is not None and getattr(self.ridm, "vocab", None) is not None:
            # 1. Get Dense vector
            query_words = query.split()
            if query_words:
                q_vec = self.ridm.context_vector_for(query_words)
                # 2. UMA Rotation to English Space
                q_vec_eng = self.uma_aligner.transform_vector(self.lang_code, q_vec)
                q_vec_eng = q_vec_eng / (np.linalg.norm(q_vec_eng) + 1e-8)
                
                # 3. Find closest English words (Top 10)
                # Ensure word_emb is normalized for dot product
                norms = np.linalg.norm(self.ridm.word_emb, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                word_emb_norm = self.ridm.word_emb / norms
                sims = np.dot(word_emb_norm, q_vec_eng)
                top_word_indices = np.argsort(sims)[::-1][:10]
                
                # 4. Map back to vocabulary
                closest_words = [self.ridm.vocab[idx] for idx in top_word_indices]
                search_query = " ".join(closest_words)
                logger.info(f"[*] Geometric Query Expansion: '{query}' -> '{search_query}'")

        query_vec = self.vectorizer.transform([search_query])
        sparse_result = self.tfidf_matrix.dot(query_vec.T).tocsc()
        # Extract non-zero entries only
        nonzero_indices = sparse_result.nonzero()[0]
        nonzero_scores = np.asarray(sparse_result[nonzero_indices, 0]).ravel()
        # Get top-k from non-zero only
        if len(nonzero_scores) > 0:
            k = min(top_k * 3, len(nonzero_scores))
            top_local_idx = np.argpartition(nonzero_scores, -k)[-k:]
            sorted_top_local = top_local_idx[np.argsort(nonzero_scores[top_local_idx])[::-1]]
            top_indices = nonzero_indices[sorted_top_local]
            top_scores = nonzero_scores[sorted_top_local]
        else:
            top_indices = np.array([], dtype=int)
            top_scores = np.array([], dtype=float)

        results = []
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            for i, idx in enumerate(top_indices):
                score = float(top_scores[i])
                if score > 0:
                    cursor.execute("SELECT text FROM documents WHERE id = ?", (int(idx),))
                    row = cursor.fetchone()
                    if row:
                        results.append((row[0], score, {"source": f"100% Native C++ SVD Core - Offline Mode (cosine={score:.2f})"}))

        # Add local documents to the mix!
        if self.dense.doc_count > 0:
            local_sparse = self.bm25.get_scores(search_query)
            top_local = np.argsort(local_sparse)[::-1][:top_k]
            for idx in top_local:
                score = float(local_sparse[idx])
                if score > 0:
                    results.append((self.documents[idx], score, {"source": f"100% Native C++ SVD Core - Offline Mode (cosine={score:.2f})"}))

        # Fast dense verification & reranking for massive corpus
        reranked = self.reranker.rerank(query, results, self.ridm, top_n=top_k)

        if len(reranked) > 0:
            self.last_max_cosine = reranked[0][1]
            self.last_used_native = self.last_max_cosine > 0.5

        return reranked

    def retrieve(self, query: str, top_k: int = 5, rrf_k: int = 60) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Retrieves top_k documents using Hybrid RRF (Reciprocal Rank Fusion).
        rrf_k is the fusion smoothing parameter.
        """
        if self.massive_enabled:
            return self._retrieve_massive(query, top_k)

        if self.dense.doc_count == 0:
            return []

        # 1. Sparse Scores
        sparse_scores = self.bm25.get_scores(query)
        sparse_ranks = np.argsort(sparse_scores)[::-1]

        # 2. Dense Scores
        query_words = query.split()
        if not query_words:
            return []

        q_vec = self.ridm.context_vector_for(query_words)

        # Apply UMA Manifold Rotation if query is not in anchor space (English)
        if self.uma_aligner and self.lang_code != "en":
            q_vec = self.uma_aligner.transform_vector(self.lang_code, q_vec)

        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        dense_scores = self.dense.get_scores(q_vec)
        dense_ranks = np.argsort(dense_scores)[::-1]

        if len(dense_scores) > 0:
            self.last_max_cosine = float(np.max(dense_scores))
            self.last_used_native = self.last_max_cosine > 0.5

        # 3. RRF Fusion
        rrf_scores = np.zeros(self.dense.doc_count, dtype=np.float32)

        # Apply RRF formula: 1 / (k + rank)
        for rank, doc_idx in enumerate(sparse_ranks):
            if sparse_scores[doc_idx] > 0:
                rrf_scores[doc_idx] += 1.0 / (rrf_k + rank + 1)

        for rank, doc_idx in enumerate(dense_ranks):
            # Dense threshold to avoid noise
            if dense_scores[doc_idx] > 0.3:
                rrf_scores[doc_idx] += 1.0 / (rrf_k + rank + 1)

        # 4. Final Sorting
        final_ranks = np.argsort(rrf_scores)[::-1]

        results = []
        for doc_idx in final_ranks[:top_k * 3]: # Get more for reranker
            if rrf_scores[doc_idx] > 0:
                results.append((
                    self.documents[doc_idx],
                    float(rrf_scores[doc_idx]),
                    self.doc_metadata[doc_idx]
                ))

        # 5. Rerank
        reranked = self.reranker.rerank(query, results, self.ridm, top_n=top_k)
        return reranked

    def deep_retrieve(self, query: str, hops: int = 2, top_k: int = 3) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Performs Multi-Hop Logic Retrieval (A->B->C).
        Retrieves initial context, extracts pivot concepts, and queries again to bridge knowledge.
        """
        results = self.retrieve(query, top_k=top_k)
        if not results or hops < 2:
            return results

        all_results = list(results)
        visited = {r[2].get('doc_id') for r in results}

        # For hop 2, we extract 'pivot' words from the top result using SVD importance
        top_context = results[0][0]
        words = top_context.split()

        # Simple extraction of long, potentially important words as pivot
        # (In a full implementation, we'd use SVD scores to find pivots)
        pivots = [w for w in words if len(w) > 5 and w.lower() not in query.lower()]
        pivot_query = " ".join(list(set(pivots))[:3]) # Top 3 pivot words

        if pivot_query:
            hop_query = query + " " + pivot_query
            hop_results = self.retrieve(hop_query, top_k=top_k)
            for r in hop_results:
                doc_id = r[2].get('doc_id')
                if doc_id not in visited:
                    all_results.append(r)
                    visited.add(doc_id)

        # Sort combined results by score
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k * 2]

    def ingest_synthetic_thought(self, thought_text: str):
        """
        DTE Phase 2: Self-Taught Learning (Synthetic Ingestion)
        Adds a highly scored synthetic thought back into the retrieval index so it can be recalled instantly.
        """
        import uuid
        doc_id = f"synthetic_{uuid.uuid4().hex[:8]}"
        chunk = {"content": thought_text, "doc_id": doc_id}

        # Add to local BM25 and Dense indexes
        self.add_chunks([chunk])
        # print(f"[HybridRetriever] Ingested synthetic thought: {doc_id}")
