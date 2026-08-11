"""Extreme Learning Machine / Reservoir tarzi kapali-form dogrusal-olmayan
derinlik (Sinirlama #3: tek-SVD'nin dogrusal sinirini asma).
"""
import numpy as np


class DeepReservoirStack:
    """Kapali-form 'derin' dogrusal-olmayan yigin (Extreme Learning Machine /
    Echo State Network felsefesi): gizli katman agirliklari RASTGELE ve
    SABITTIR (backprop yok); sadece SON katman (okuma/readout) ridge
    regresyonla (kapali-form, TEK adimda: (H^T H + lam*I)^-1 H^T Y) cozulur.
    Bu, cok-katmanli dogrusal-olmayanligi, agir gradyan-inisi dongusune
    girmeden saglar - v3'teki tek buyuk (dogrusal) SVD'nin ustune bir
    dogrusal-olmayan ic-buku ekler."""

    def __init__(self, in_dim, hidden_dims=(128, 128), seed=5):
        rng = np.random.RandomState(seed)
        self.layers = []
        d = in_dim
        for hd in hidden_dims:
            W = rng.randn(hd, d) / np.sqrt(d)
            b = rng.randn(hd) * 0.01
            self.layers.append((W, b))
            d = hd
        self.out_dim = d
        self.W_out = None

    def encode(self, x):
        h = x
        for W, b in self.layers:
            h = np.tanh(W @ h + b)
        return h

    def encode_batch(self, X):
        H = X
        for W, b in self.layers:
            H = np.tanh(H @ W.T + b[None, :])
        return H

    def fit_readout(self, X, Y, lam=1.0):
        H = self.encode_batch(X)
        A = H.T @ H + lam * np.eye(self.out_dim)
        B = H.T @ Y
        self.W_out = np.linalg.solve(A, B)

    def predict(self, x):
        h = self.encode(x)
        if self.W_out is None:
            return h
        return h @ self.W_out


class DeepReservoirScorer:
    """DeepReservoirStack'i, RIDM baglam vektorlerinden hedef kelime
    gommelerini (word_emb, k-boyutlu) tahmin edecek sekilde KAPALI-FORM
    (ridge regresyon) egiten sarmalayici. Cikan skor, HybridLM icinde
    ekstra bir 'dogrusal-olmayan gorus' olarak diger sinyallerle harmanlanir."""

    def __init__(self, ridm, hidden_dims=(128, 128), lam=2.0, seed=9):
        self.ridm = ridm
        self.stack = DeepReservoirStack(ridm.dim, hidden_dims=hidden_dims, seed=seed)
        self.lam = lam
        self.fitted = False

    def fit(self, token_ids, max_samples=1500, seed=0):
        rng = np.random.RandomState(seed)
        W = self.ridm.window
        n = len(token_ids)
        if n <= W:
            return
        idxs = np.arange(W, n)
        if len(idxs) > max_samples:
            idxs = rng.choice(idxs, size=max_samples, replace=False)
        X, Y = [], []
        for i in idxs:
            ctx_ids = [t for t in token_ids[i - W : i] if 0 <= t < self.ridm.V]
            if not ctx_ids:
                continue
            cvec = self.ridm.context_vecs[ctx_ids].sum(axis=0)
            X.append(cvec)
            Y.append(self.ridm.word_emb[token_ids[i]])
        if not X:
            return
        X = np.array(X)
        Y = np.array(Y)
        self.stack.fit_readout(X, Y, lam=self.lam)
        self.fitted = True

    def score(self, context_token_ids):
        if not self.fitted:
            return None
        ctx_ids = [t for t in context_token_ids if 0 <= t < self.ridm.V]
        if not ctx_ids:
            return None
        cvec = self.ridm.context_vecs[ctx_ids].sum(axis=0)
        approx_emb = self.stack.predict(cvec)
        return self.ridm.word_emb @ approx_emb


# ======================================================
# 11) PRODUCT-KEY MEMORY (PKM)
# ======================================================
