"""RIDM cekirdegi: hibrit rastgele indeksleme + kapali-form (tek-adim) SVD.

Sistemin '4. Bilgi kapasitesi' ve temel next-token tahmin mekanizmasi
burada yasar.
"""
import math

import numpy as np

from .backend import ComputeBackend
from .constants import SENTENCE_BOUNDARY_TOKENS, UNK_TOKEN
from .subword import SubwordHasher


class RIDM:
    def __init__(
        self,
        vocab,
        counts=None,
        dim=300,
        window=5,
        seed=42,
        idf_weighting=True,
        pos_decay=0.3,
        subword_ngrams=3,
        subword_buckets=2 ** 16,
        backend="auto",
        device="auto",
        threads=None,
        compute_backend=None,
    ):
        if len(vocab) == 0:
            raise ValueError("Sozluk (vocab) bos olamaz.")
        if dim <= 0 or window <= 0:
            raise ValueError("dim ve window pozitif olmalidir.")

        self.vocab = list(vocab)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.V = len(self.vocab)
        self.dim = dim
        self.window = window
        self.seed = seed
        self.pos_decay = pos_decay
        self.subword_ngrams = subword_ngrams
        self.compute = compute_backend or ComputeBackend(backend=backend, device=device, threads=threads)

        rng = np.random.RandomState(seed)
        # float32 native/CUDA kernels için bellek bant genişliğini yarıya indirir.
        self.context_vecs = (rng.randn(self.V, dim) / np.sqrt(dim)).astype(np.float32)

        if idf_weighting and counts:
            total = sum(counts.values()) + 1
            self.idf = np.array([math.log(total / (counts.get(w, 0) + 1) + 1.0) for w in self.vocab], dtype=np.float32)
            self.idf = self.idf / (self.idf.mean() + 1e-8)
        else:
            self.idf = np.ones(self.V, dtype=np.float32)

        self.hasher = SubwordHasher(dim, buckets=subword_buckets, n=3, seed=seed + 1) if subword_ngrams else None

        self.M = np.zeros((self.V, dim), dtype=np.float32)
        self.target_counts = np.zeros(self.V, dtype=np.float32)
        self.total_contexts = 0
        self.boundary_ids = {self.word2idx[t] for t in SENTENCE_BOUNDARY_TOKENS if t in self.word2idx}

        self.word_emb = None
        self._Vt_k = None
        self._S_k = None
        self.k = None

    def _weighted_context_sums(self, token_ids):
        token_ids = np.asarray(token_ids, dtype=np.int64)
        n = len(token_ids)
        W = self.window
        if n <= W:
            return None, None
        weighted_cvecs = self.context_vecs[token_ids] * self.idf[token_ids][:, None]
        if self.pos_decay > 0:
            dist_w = np.exp(-self.pos_decay * (np.arange(1, W + 1) - 1)).astype(np.float32)
        else:
            dist_w = np.ones(W, dtype=np.float32)
        context_sums = np.zeros((n - W, self.dim), dtype=np.float32)
        for d in range(1, W + 1):
            context_sums += dist_w[d - 1] * weighted_cvecs[W - d : n - d]
        targets = token_ids[W:n]
        return targets, context_sums

    def partial_fit(self, token_ids):
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if len(token_ids) <= self.window:
            return
        if self.pos_decay > 0:
            distance_weights = np.exp(-self.pos_decay * np.arange(self.window, dtype=np.float32))
        else:
            distance_weights = np.ones(self.window, dtype=np.float32)

        # Cümle Sınırı Farkındalığı (Sentence Boundary Awareness)
        # Boundary split sonrası tüm chunk'lar window'dan kısaysa, split'i
        # devre dışı bırakıp tüm diziyi tek parça işle — aksi hâlde
        # total_contexts sıfır kalır ve model hiç eğitilemez.
        chunks = [token_ids]
        if self.boundary_ids:
            mask = np.isin(token_ids, list(self.boundary_ids))
            boundary_indices = np.where(mask)[0] + 1
            if len(boundary_indices) > 0 and boundary_indices[-1] == len(token_ids):
                boundary_indices = boundary_indices[:-1]
            candidate_chunks = np.split(token_ids, boundary_indices)
            has_viable = any(len(c) > self.window for c in candidate_chunks)
            if has_viable:
                chunks = candidate_chunks

        # C++/OpenMP çekirdeği her cümle için ayrı çalıştırılır
        for chunk in chunks:
            if len(chunk) > self.window:
                self.total_contexts += self.compute.accumulate_contexts(
                    chunk, self.context_vecs, self.idf, self.window, distance_weights,
                    self.M, self.target_counts,
                )

    def streaming_fit(self, data_iterator, rank: int = 128,
                      checkpoint_dir: str = "checkpoints",
                      checkpoint_interval: int = 10,
                      batch_size: int = 100_000) -> None:
        """Process an infinite data stream with incremental SVD — no RAM overflow.
        
        Algorithm:
        1. Process batches via partial_fit
        2. Every checkpoint_interval batches:
           a. If first time: finalize (full SVD)
           b. Else: incremental_update (Brand & Hall)
           c. Save checkpoint
           d. Reset accumulation buffers (free memory)
        """
        import os
        import logging
        logger = logging.getLogger(__name__)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        batch_tokens = []
        all_interval_tokens = []
        batch_count = 0
        
        for token_id in data_iterator:
            batch_tokens.append(token_id)
            if len(batch_tokens) >= batch_size:
                self.partial_fit(batch_tokens)
                all_interval_tokens.extend(batch_tokens)
                batch_tokens = []
                batch_count += 1
                
                if batch_count % checkpoint_interval == 0:
                    if self.word_emb is None:
                        logger.info(f"Checkpoint {batch_count}: Initial finalize (rank={rank})")
                        self.finalize(k=rank)
                    else:
                        logger.info(f"Checkpoint {batch_count}: Incremental update")
                        self.incremental_update(all_interval_tokens)
                        
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{batch_count}.npz")
                    self.save(checkpoint_path)
                    
                    # Reset accumulation buffers
                    self.M.fill(0)
                    self.target_counts.fill(0)
                    all_interval_tokens = []
                    
        # Process remaining tokens
        if batch_tokens:
            self.partial_fit(batch_tokens)
            all_interval_tokens.extend(batch_tokens)
            
        if all_interval_tokens:
            if self.word_emb is None:
                self.finalize(k=rank)
            else:
                self.incremental_update(all_interval_tokens)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_final.npz")
            self.save(checkpoint_path)
            self.M.fill(0)
            self.target_counts.fill(0)

    def export_state(self, filepath: str):
        """Dağıtık eğitim (Cluster Training) için durum matrislerini dışa aktarır."""
        np.savez_compressed(
            filepath,
            M=self.M,
            target_counts=self.target_counts,
            total_contexts=np.array([self.total_contexts], dtype=np.int64)
        )

    def import_and_merge_state(self, filepath: str):
        """Başka bir düğümden (Node) gelen matrisleri mevcut sisteme toplar."""
        data = np.load(filepath)
        M_ext = data['M']
        counts_ext = data['target_counts']
        contexts_ext = data['total_contexts'][0]

        if M_ext.shape != self.M.shape:
            raise ValueError(f"Uyumsuz model boyutu. Beklenen {self.M.shape}, gelen {M_ext.shape}")

        self.M += M_ext
        self.target_counts += counts_ext
        self.total_contexts += int(contexts_ext)


    def finalize(self, k=128):
        if self.total_contexts == 0:
            raise RuntimeError("Model hic egitilmedi. Korpus 'window' uzunlugundan kisa olabilir.")
        k = max(1, min(k, self.dim, self.V))
        safe_counts = np.maximum(self.target_counts, 1)[:, None]
        M_norm = self.M / safe_counts
        U_k, S_k, Vt_k = self.compute.truncated_svd(M_norm.astype(np.float32, copy=False), k)
        self.word_emb = U_k * S_k
        self._Vt_k = Vt_k
        self._S_k = S_k
        self.k = k
        self._M_norm_cache = M_norm

    def incremental_update(self, token_ids):
        if self.word_emb is None or self._S_k is None or self._Vt_k is None:
            self.finalize(k=self.k or 128)
            return

        targets, context_sums = self._weighted_context_sums(token_ids)
        if targets is None:
            return

        S = self._S_k
        Vt = self._Vt_k
        U_k = self.word_emb / np.maximum(self._S_k[None, :], 1e-8)
        
        # Handle rank mismatches gracefully
        if U_k.shape[1] != len(S):
            min_k = min(U_k.shape[1], len(S))
            U_k = U_k[:, :min_k]
            S = S[:min_k]
            Vt = Vt[:min_k, :]

        for row_idx, delta in zip(targets, context_sums):
            a = np.zeros(self.V)
            a[row_idx] = 1.0
            b = delta

            m = U_k.T @ a
            p = a - U_k @ m
            Ra = np.linalg.norm(p)
            P = p / Ra if Ra > 1e-8 else np.zeros_like(p)

            n_ = Vt @ b
            q = b - Vt.T @ n_
            Rb = np.linalg.norm(q)
            Q = q / Rb if Rb > 1e-8 else np.zeros_like(q)

            k = len(S)
            K = np.zeros((k + 1, k + 1))
            K[:k, :k] = np.diag(S)
            K[:, k] += np.concatenate([m, [Ra]])
            K[k, :] += np.concatenate([n_, [Rb]])
            K[k, k] = Ra * Rb if Ra > 1e-8 and Rb > 1e-8 else K[k, k]

            Uk2, Sk2, Vtk2 = np.linalg.svd(K)
            k_new = min(k, len(Sk2))
            Uk2, Sk2, Vtk2 = Uk2[:, :k_new], Sk2[:k_new], Vtk2[:k_new, :]

            U_ext = np.hstack([U_k, P[:, None]])
            Vt_ext = np.vstack([Vt, Q[None, :]])

            U_k = U_ext @ Uk2
            Vt = Vtk2 @ Vt_ext
            S = Sk2

            self.target_counts[row_idx] += 1

        self._S_k = S
        self._Vt_k = Vt
        self.word_emb = U_k * S
        self.k = len(S)
        safe_counts = np.maximum(self.target_counts, 1)[:, None]
        self._M_norm_cache = self.M / safe_counts

    def drift_estimate(self, sample_token_ids, max_samples=200):
        self._check_ready()
        approx = self.word_emb @ self._Vt_k
        idx = sample_token_ids[:max_samples]
        idx = [i for i in idx if 0 <= i < self.V]
        if not idx:
            return float("nan")
        true_rows = self._M_norm_cache[idx] if hasattr(self, "_M_norm_cache") else self.M[idx]
        err = np.linalg.norm(approx[idx] - true_rows, axis=1)
        norm = np.linalg.norm(true_rows, axis=1) + 1e-8
        return float((err / norm).mean())

    def _check_ready(self):
        if self.word_emb is None:
            raise RuntimeError("Once finalize() cagirmalisiniz.")

    def _word_vec(self, word):
        idx = self.word2idx.get(word)
        if idx is not None:
            return self.context_vecs[idx]
        if self.hasher is not None:
            return self.hasher.vector(word)
        return self.context_vecs[self.word2idx[UNK_TOKEN]]

    def context_vector_for(self, words):
        if not words:
            return np.zeros(self.dim)
        return np.sum([self._word_vec(w) for w in words], axis=0, dtype=np.float32)

    def logits_from_ids(self, context_token_ids):
        self._check_ready()
        context_token_ids = [t for t in context_token_ids if 0 <= t < self.V]
        if not context_token_ids:
            return np.zeros(self.V)
        ctx_vec = np.sum(self.context_vecs[context_token_ids], axis=0, dtype=np.float32)
        return self.compute.matvec(self.word_emb, self.compute.matvec(self._Vt_k, ctx_vec))

    def logits_from_words(self, words):
        self._check_ready()
        ctx_vec = self.context_vector_for(words)
        return self.compute.matvec(self.word_emb, self.compute.matvec(self._Vt_k, ctx_vec))

    def predict_topk(self, context_words, k=5):
        scores = self.logits_from_words(context_words)
        order = np.argsort(scores)[::-1][:k]
        return [(self.idx2word[i], float(scores[i])) for i in order]

    def softmax_probs(self, context_token_ids, temperature=1.0):
        raw = self.logits_from_ids(context_token_ids)
        std = raw.std()
        z = (raw - raw.mean()) / (std + 1e-8)
        z = z / max(temperature, 1e-8)
        z = z - np.max(z)
        e = np.exp(z)
        return e / e.sum()

    def evaluate(self, token_ids, max_samples=1000):
        self._check_ready()
        n = len(token_ids)
        if n <= self.window:
            return {"accuracy": float("nan"), "perplexity": float("nan"), "n_samples": 0}
        correct, nll_sum, count = 0, 0.0, 0
        step = max(1, (n - self.window) // max_samples) if max_samples else 1
        for i in range(self.window, n, step):
            ctx = token_ids[i - self.window : i]
            target = token_ids[i]
            probs = self.softmax_probs(ctx)
            correct += int(np.argmax(probs) == target)
            nll_sum += -math.log(max(probs[target], 1e-12))
            count += 1
            if max_samples and count >= max_samples:
                break
        ppl = math.exp(nll_sum / count) if count else float("nan")
        return {"accuracy": correct / count if count else float("nan"), "perplexity": ppl, "n_samples": count}

    def save(self, path):
        self._check_ready()
        np.savez_compressed(
            path, context_vecs=self.context_vecs, M=self.M, word_emb=self.word_emb,
            Vt_k=self._Vt_k, S_k=self._S_k, idf=self.idf, dim=self.dim, window=self.window,
            seed=self.seed, k=self.k, total_contexts=self.total_contexts, target_counts=self.target_counts,
            pos_decay=self.pos_decay,
            vocab=np.array(self.vocab, dtype=object),
        )

    @classmethod
    def load(cls, path, backend="auto", device="auto", threads=None):
        data = np.load(path, allow_pickle=True)
        vocab = list(data["vocab"])
        pos_decay = float(data["pos_decay"]) if "pos_decay" in data.files else 0.3
        model = cls(vocab, dim=int(data["dim"]), window=int(data["window"]), seed=int(data["seed"]),
                    subword_ngrams=0, pos_decay=pos_decay, backend=backend, device=device, threads=threads)
        model.context_vecs = data["context_vecs"].astype(np.float32, copy=False)
        model.M = data["M"].astype(np.float32, copy=False)
        model.word_emb = data["word_emb"].astype(np.float32, copy=False)
        model._Vt_k = data["Vt_k"].astype(np.float32, copy=False)
        model._S_k = data["S_k"].astype(np.float32, copy=False)
        model.idf = data["idf"].astype(np.float32, copy=False)
        model.target_counts = (data["target_counts"].astype(np.float32, copy=False)
                               if "target_counts" in data.files else np.ones(model.V, dtype=np.float32))
        model.k = int(data["k"])
        model.total_contexts = int(data["total_contexts"])
        return model


# ======================================================
# 2) HIERARCHICAL MEMORY - kisa/orta/uzun baglam katmanlari
# ======================================================
