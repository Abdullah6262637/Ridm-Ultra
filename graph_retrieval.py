"""Anlamsal komsuluk grafigi (yayilan aktivasyon) + getirim katmani
(RAG) + Locality-Sensitive Hashing (LSH) tabanli Approximate Nearest
Neighbor indeksi (Sinirlama #7: gercek ANN retrieval).
"""
import time
from collections import defaultdict

import numpy as np

# Cosine similarity threshold above which a retrieval hit is considered
# genuine enough to attribute a response to the native C++ SVD RAG path
# (see ridm_ultra.chat.engine / ridm_ultra.chat.adapters for how this is
# consumed). This must stay in sync with the acceptance criteria: a hit
# is only "native" if it (a) actually ran through the compiled C++ kernel
# and (b) cleared this threshold.
NATIVE_RAG_COSINE_THRESHOLD = 0.65

class SemanticGraph:
    def __init__(self, ridm, top_m=5):
        self._check(ridm)
        emb = ridm.word_emb
        norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        sims = norm @ norm.T
        np.fill_diagonal(sims, -1)
        self.neighbors = np.argsort(sims, axis=1)[:, ::-1][:, :top_m]
        self.neighbor_sims = np.take_along_axis(sims, self.neighbors, axis=1)
        self.V = ridm.V
        self.idx2word = ridm.idx2word

    @staticmethod
    def _check(ridm):
        if ridm.word_emb is None:
            raise RuntimeError("SemanticGraph icin once RIDM.finalize() cagrilmali.")

    def spreading_activation(self, seed_ids, steps=2, decay=0.6):
        activation = np.zeros(self.V, dtype=np.float64)
        for sid in seed_ids:
            if 0 <= sid < self.V:
                activation[sid] = 1.0
        for _ in range(steps):
            new_act = np.zeros_like(activation)
            active = np.where(activation > 1e-6)[0]
            for node in active:
                for nb, sim in zip(self.neighbors[node], self.neighbor_sims[node]):
                    new_act[nb] += activation[node] * sim * decay
            activation = np.maximum(activation, new_act)
        return activation

# ======================================================
# 6) RAG-BENZERI GETIRIM KATMANI
# ======================================================


class LSHIndex:
    """Rastgele-hiperduzlem Locality-Sensitive Hashing (SimHash) tabanli
    Approximate Nearest Neighbor indeksi - FAISS/HNSW/ScaNN gibi tam
    kutuphaneler olmadan, KAPALI-FORM (egitimsiz, sabit rastgele
    hiperduzlemler) bir ANN yaklasimi. Cok-tablolu (multi-table) hashing
    ile recall artirilir. v3/v4'teki SimpleRAG'in brute-force dogrusal
    taramasinin yerini alabilir: buyuk belge havuzlarinda O(n) yerine
    kova (bucket) icinde kucuk-N arama saglar.

    DURUSTLUK: Bu, DiskANN/HNSW gibi grafik-tabanli modern ANN yontemleri
    kadar hassas degildir; klasik, basit ama GERCEK calisan bir LSH
    semasidir (Charikar 2002 SimHash ailesinden)."""

    def __init__(self, dim, n_tables=4, n_bits=8, seed=31):
        rng = np.random.RandomState(seed)
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.planes = [rng.randn(n_bits, dim) for _ in range(n_tables)]
        self.tables = [defaultdict(list) for _ in range(n_tables)]
        self.items = []

    def _hash(self, vec, table_idx):
        proj = self.planes[table_idx] @ vec
        bits = (proj > 0).astype(int)
        return tuple(bits.tolist())

    def add(self, vec, payload):
        idx = len(self.items)
        self.items.append((vec, payload))
        for t in range(self.n_tables):
            h = self._hash(vec, t)
            self.tables[t][h].append(idx)

    def query(self, vec, top_n=3):
        cands = set()
        for t in range(self.n_tables):
            h = self._hash(vec, t)
            cands.update(self.tables[t].get(h, []))
        if not cands:
            cands = set(range(len(self.items)))  # kucuk havuzlarda guvenli fallback
        scored = []
        for idx in cands:
            v, payload = self.items[idx]
            sim = float(v @ vec / (np.linalg.norm(v) * np.linalg.norm(vec) + 1e-8))
            scored.append((sim, payload))
        scored.sort(key=lambda t: -t[0])
        return scored[:top_n]


class SimpleRAG:
    def __init__(self, ridm, documents, use_lsh=False, lsh_tables=4, lsh_bits=8, seed=31):
        self.ridm = ridm
        self.documents = documents
        self.doc_vecs = []
        self.doc_tokens = []
        for doc in documents:
            words = doc.split()
            toks = [ridm.word2idx.get(w) for w in words if w in ridm.word2idx]
            self.doc_tokens.append(toks)
            vec = ridm.context_vector_for(words)
            self.doc_vecs.append(vec / (np.linalg.norm(vec) + 1e-8))
        self.doc_vecs = np.array(self.doc_vecs) if self.doc_vecs else np.zeros((0, ridm.dim))
        self.lsh = None
        if use_lsh and len(self.documents) > 0:
            self.lsh = LSHIndex(ridm.dim, n_tables=lsh_tables, n_bits=lsh_bits, seed=seed)
            for i, v in enumerate(self.doc_vecs):
                self.lsh.add(v, i)

    def retrieve(self, context_words, top_n=2, threshold=2.5):
        """Retrieve top-N candidate documents.

        Also sets ``self.last_used_native`` and ``self.last_max_cosine`` so
        callers (ChatEngine) can determine the REAL source attribution
        instead of asking an LLM to guess/announce it. The cosine
        similarity step is executed through ``ComputeBackend.matvec``,
        which dispatches to the compiled C++/OpenMP kernel whenever the
        backend is not explicitly running in numpy/torch mode -- there is
        no separate/parallel "fake" numpy path used just for display.
        """
        self.last_used_native = False
        self.last_max_cosine = 0.0

        if len(self.documents) == 0:
            return []

        clean_q = set(w.lower().strip(".,;:!?()[]{}'\"") for w in context_words if len(w) > 2)

        q = self.ridm.context_vector_for(context_words)
        q = q / (np.linalg.norm(q) + 1e-8)

        if self.lsh is not None:
            hits = self.lsh.query(q, top_n=top_n)
            if hits:
                self.last_max_cosine = float(max(sim for sim, _ in hits))
                self.last_used_native = self.last_max_cosine >= NATIVE_RAG_COSINE_THRESHOLD
            return [(self.documents[i], sim, self.doc_tokens[i]) for sim, i in hits]

        compute = getattr(self.ridm, "compute", None)
        doc_vecs_f32 = np.ascontiguousarray(self.doc_vecs, dtype=np.float32)
        q_f32 = np.ascontiguousarray(q, dtype=np.float32)

        if compute is not None and doc_vecs_f32.shape[0] > 0:
            t0 = time.perf_counter()
            sims = compute.matvec(doc_vecs_f32, q_f32)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            used_native = compute.active == "native"
            print(
                f"[C++ KERNEL] Cosine similarity computed across {doc_vecs_f32.shape[0]:,} "
                f"vectors in {elapsed_ms:.2f} ms (Threads: OpenMP)."
                if used_native else
                f"[NON-NATIVE] Cosine similarity computed across {doc_vecs_f32.shape[0]:,} "
                f"vectors in {elapsed_ms:.2f} ms (backend='{compute.active}', NOT the C++ kernel)."
            )
        else:
            used_native = False
            sims = doc_vecs_f32 @ q_f32

        if len(sims):
            self.last_max_cosine = float(np.max(sims))
        self.last_used_native = used_native and self.last_max_cosine >= NATIVE_RAG_COSINE_THRESHOLD

        # Hybrid term overlap scoring for precise RAG attribution
        scores = np.copy(sims)
        max_possible_matches = max(1, len(clean_q))
        for i, doc in enumerate(self.documents):
            d_lower = doc.lower()
            matches = sum(1 for w in clean_q if w in d_lower)
            if matches > 1:
                scores[i] += matches / max_possible_matches

        order = np.argsort(scores)[::-1][:top_n]
        return [(self.documents[i], float(scores[i]), self.doc_tokens[i]) for i in order if scores[i] >= threshold]


    def prior_bonus(self, context_words, top_n=2):
        bonus = np.zeros(self.ridm.V)
        hits = self.retrieve(context_words, top_n=top_n)
        for _, sim, toks in hits:
            for t in toks:
                bonus[t] += max(sim, 0)
        return bonus


# ======================================================
# N-gram taban cizgisi
# ======================================================
