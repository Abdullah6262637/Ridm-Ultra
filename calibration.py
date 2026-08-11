"""Kapali-form (ridge) autoregressive kalibrasyon + GERCEKTEN SGD ile
egitilen Neural Reranker (sistemin backprop iceren tek iki parcasindan biri).
"""
import numpy as np


class AutoregressiveCalibrator:
    """Su ana kadarki alt-sistemlerin (RIDM, n-gram, relation-attention,
    reservoir) URETTIGI HAM SKORLARI, GERCEK bir sonraki-token hedefine
    (autoregressive objective) en iyi UYACAK sekilde KAPALI-FORM (ridge,
    tek adimda en-kucuk-kareler) kalibre eder. GPT'nin cross-entropy
    dongusunun YERINE GECMEZ, ama AYNI HEDEFE (dogru sonraki kelimeyi one
    cikarmak) kapali-form bir dogrusal cozumle YAKLASIR - v3/v4'teki Neural
    Reranker'in (gercek SGD) daha ucuz, kapali-form bir tamamlayicisidir."""

    def __init__(self, ridm, ngram, relation_attn=None, reservoir=None, lam=5.0):
        self.ridm = ridm
        self.ngram = ngram
        self.relation_attn = relation_attn
        self.reservoir = reservoir
        self.lam = lam
        self.W = None

    def _feature_matrix(self, ctx_ids):
        feats = [self.ridm.logits_from_ids(ctx_ids), self.ngram.probs(ctx_ids)]
        if self.relation_attn is not None:
            feats.append(self.relation_attn.forward(ctx_ids))
        if self.reservoir is not None:
            r = self.reservoir.score(ctx_ids)
            feats.append(r if r is not None else np.zeros(self.ridm.V))
        F = np.stack(feats, axis=1)
        F = (F - F.mean(axis=0, keepdims=True)) / (F.std(axis=0, keepdims=True) + 1e-6)
        return F

    def fit(self, token_ids, max_samples=500, seed=0):
        rng = np.random.RandomState(seed)
        W = self.ridm.window
        n = len(token_ids)
        if n <= W:
            return
        idxs = np.arange(W, n)
        if len(idxs) > max_samples:
            idxs = rng.choice(idxs, size=max_samples, replace=False)
        Xs, ys = [], []
        for i in idxs:
            ctx = token_ids[i - W : i]
            Xs.append(self._feature_matrix(ctx))
            y = np.zeros(self.ridm.V)
            y[token_ids[i]] = 1.0
            ys.append(y)
        if not Xs:
            return
        X = np.vstack(Xs)
        y = np.concatenate(ys)
        n_feat = X.shape[1]
        A = X.T @ X + self.lam * np.eye(n_feat)
        b = X.T @ y
        self.W = np.linalg.solve(A, b)

    def calibrated_logits(self, ctx_ids):
        if self.W is None:
            return None
        return self._feature_matrix(ctx_ids) @ self.W


# ======================================================
# 18) REASONING CONTROLLER - kapali-form planlayici / denetleyici
# ======================================================


class NeuralReranker:
    def __init__(self, n_features=4, hidden=16, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(hidden, n_features) * 0.3
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden) * 0.3
        self.b2 = 0.0

    def _forward(self, X):
        h = np.tanh(X @ self.W1.T + self.b1)
        z = h @ self.W2 + self.b2
        p = 1.0 / (1.0 + np.exp(-z))
        return p, h

    def _features(self, ridm, ngram, ctx_ids, cand_id):
        ridm_score = ridm.logits_from_ids(ctx_ids)[cand_id]
        ngram_p = ngram.probs(ctx_ids)[cand_id]
        freq = ridm.target_counts[cand_id] / (ridm.total_contexts + 1)
        ctx_vec = np.sum(ridm.context_vecs[[t for t in ctx_ids if 0 <= t < ridm.V]], axis=0) \
            if any(0 <= t < ridm.V for t in ctx_ids) else np.zeros(ridm.dim)
        cos = float(
            (ctx_vec @ ridm.context_vecs[cand_id])
            / (np.linalg.norm(ctx_vec) * np.linalg.norm(ridm.context_vecs[cand_id]) + 1e-8)
        )
        return np.array([ridm_score, ngram_p, freq, cos])

    def train(self, ridm, ngram, token_ids, top_n=8, epochs=5, lr=0.1, max_samples=1500, seed=0):
        rng = np.random.RandomState(seed)
        n = len(token_ids)
        w = ridm.window
        if n <= w:
            return
        idxs = np.arange(w, n)
        if len(idxs) > max_samples:
            idxs = rng.choice(idxs, size=max_samples, replace=False)

        samples_X, samples_y = [], []
        for i in idxs:
            ctx = token_ids[i - w : i]
            target = token_ids[i]
            scores = ridm.logits_from_ids(ctx)
            cand_ids = list(np.argsort(scores)[::-1][:top_n])
            if target not in cand_ids:
                cand_ids[-1] = target
            for c in cand_ids:
                feats = self._features(ridm, ngram, ctx, c)
                samples_X.append(feats)
                samples_y.append(1.0 if c == target else 0.0)

        X = np.array(samples_X)
        y = np.array(samples_y)
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + 1e-6
        Xn = (X - self.mu) / self.sigma

        m = len(Xn)
        for _ in range(epochs):
            order = rng.permutation(m)
            for idx in order:
                x = Xn[idx]
                target_y = y[idx]
                p, h = self._forward(x[None, :])
                p, h = p[0], h[0]
                dz = p - target_y
                dW2 = dz * h
                db2 = dz
                dh = dz * self.W2 * (1 - h ** 2)
                dW1 = np.outer(dh, x)
                db1 = dh
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

    def rerank(self, ridm, ngram, ctx_ids, top_n=8):
        scores = ridm.logits_from_ids(ctx_ids)
        cand_ids = list(np.argsort(scores)[::-1][:top_n])
        feats = np.array([self._features(ridm, ngram, ctx_ids, c) for c in cand_ids])
        Xn = (feats - self.mu) / self.sigma
        p, _ = self._forward(Xn)
        order = np.argsort(p)[::-1]
        return [(ridm.idx2word[cand_ids[i]], float(p[i])) for i in order]


# ======================================================
# Ornek korpuslar
# ======================================================
