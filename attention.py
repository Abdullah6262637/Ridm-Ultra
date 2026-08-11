"""Kapali-form self-attention (rastgele-agirlikli, pozisyon-kodlamali) VE
veri-turevli (PPMI-SVD tabanli) 'ogrenilmis-iliski' attention'i (Sinirlama
#1) + coklu-blok residual yigin (Sinirlama #3-4). DURUSTLUK: hicbiri
gercek backprop ile OGRENILMEZ; bkz. sinif docstring'leri.
"""
import math

import numpy as np


class ClosedFormAttention:
    """GPT'deki ile AYNI temel matematik: softmax(QK^T/sqrt(d)) V, coklu-basli.
    Fark: Wq/Wk/Wv agirliklari GRADYAN ILE OGRENILMEZ - sabit rastgele (ya da
    RIDM'in SVD tabanindan (Vt_k) turetilmis, veriye duyarli fakat yine
    kapali-form) matrislerdir. Ayrica sinuzoidal pozisyon kodlamasi eklenir,
    boylece baglamin SIRASI artik sonucu etkiler (v3'teki duz toplamin
    kaybettigi bilgi).

    Bu katman "egitilmis attention" degildir; ama YAPISAL olarak gercek bir
    attention mekanizmasidir: her baglam kelimesi, ICERIGINE ve KONUMUNA gore
    degisen, sorguya-ozel bir agirlikla katkida bulunur (sadece sabit mesafe
    agirligina gore degil)."""

    def __init__(self, dim, n_heads=4, seed=11, use_ridm_basis=None):
        self.dim = dim
        self.n_heads = max(1, n_heads)
        self.head_dim = max(1, dim // self.n_heads)
        proj_dim = self.head_dim * self.n_heads
        self.proj_dim = proj_dim
        rng = np.random.RandomState(seed)

        if use_ridm_basis is not None and use_ridm_basis.shape[0] > 0:
            basis = use_ridm_basis  # (k, dim), RIDM'in SVD-anlamsal tabani
            reps = int(np.ceil(proj_dim / basis.shape[0]))
            base = np.tile(basis, (reps, 1))[:proj_dim]
            self.Wq = base + 0.05 * rng.randn(proj_dim, dim)
            self.Wk = base + 0.05 * rng.randn(proj_dim, dim)
        else:
            self.Wq = rng.randn(proj_dim, dim) / np.sqrt(dim)
            self.Wk = rng.randn(proj_dim, dim) / np.sqrt(dim)
        self.Wv = rng.randn(proj_dim, dim) / np.sqrt(dim)
        self.Wo = rng.randn(dim, proj_dim) / np.sqrt(proj_dim)
        self._pe_cache = {}

    def _pe(self, n, dim):
        key = (n, dim)
        if key not in self._pe_cache:
            pos = np.arange(max(n, 1))[:, None]
            i = np.arange(dim)[None, :]
            angle = pos / np.power(10000.0, (2 * (i // 2)) / max(dim, 1))
            pe = np.zeros((max(n, 1), dim))
            pe[:, 0::2] = np.sin(angle[:, 0::2])
            pe[:, 1::2] = np.cos(angle[:, 1::2])
            self._pe_cache[key] = pe
        return self._pe_cache[key]

    def forward(self, context_vecs, query_vec=None):
        """context_vecs: (n, dim) siraya-duyarli (en eski -> en yeni).
        Dondurur: (attention_output (dim,), position_weights (n,))"""
        n = context_vecs.shape[0]
        if n == 0:
            return np.zeros(self.dim), np.array([])
        X = context_vecs + self._pe(n, self.dim)
        if query_vec is not None:
            q_src = query_vec + self._pe(1, self.dim)[0]
        else:
            q_src = X[-1]

        Q = (self.Wq @ q_src).reshape(self.n_heads, self.head_dim)
        K = (X @ self.Wk.T).reshape(n, self.n_heads, self.head_dim)
        V = (X @ self.Wv.T).reshape(n, self.n_heads, self.head_dim)

        out_heads = []
        attn_all = np.zeros((self.n_heads, n))
        for h in range(self.n_heads):
            scores = (K[:, h, :] @ Q[h]) / math.sqrt(self.head_dim)
            scores = scores - scores.max()
            w = np.exp(scores)
            w = w / (w.sum() + 1e-12)
            attn_all[h] = w
            out_heads.append(w @ V[:, h, :])
        concat = np.concatenate(out_heads)
        out = self.Wo @ concat
        return out, attn_all.mean(axis=0)


# ======================================================
# 10) DEEP RESERVOIR / EXTREME LEARNING MACHINE (ELM)
# ======================================================


class CooccurrenceRelationBasis:
    """Token-token PENCERE-ICI es-olusum (co-occurrence) istatistiginden,
    GERCEKTEN VERIYE UYGUN (data-fit) Q/K taban vektorleri cikarir. SGD/
    backprop YOKTUR; bunun yerine PPMI-agirlikli (Positive Pointwise Mutual
    Information) co-occurrence matrisi KAPALI-FORM bir SVD ile factorize
    edilir (GloVe/LSA'nin yaptigina benzer sekilde - Levy&Goldberg 2014,
    'Neural Word Embedding as Implicit Matrix Factorization' PPMI-SVD'nin
    SGD ile egitilen word2vec'e YAKLASIK ESDEGER oldugunu gostermistir).
    Boylece Q/K projeksiyonlari ARTIK RASTGELE degil, korpustaki GERCEK
    birlikte-gorulme oruntulerinden turetilir.

    DURUSTLUK: Bu hala gradyan-ogrenmesi degildir; yine de VERIYE DUYARLI
    ve KAPALI-FORM (tek adimda SVD) bir 'ogrenme' bicimidir. Ogrenilen
    sadece IKINCI-DERECE (pairwise co-occurrence) istatistiktir; GPT'nin
    derin, gorev-ozel, cok-katmanli ogrendigi soyut rol-atama iliskileriyle
    ('Ali kimi tedavi etti?' turu coklu-adim cikarim) AYNI GUCTE DEGILDIR -
    ama 'Ali...doktor', 'Ali...tedavi' gibi DOGRUDAN es-olusan kelime
    ciftlerini, rastgele projeksiyonun asla yapamayacagi sekilde one cikarir.

    OLCEKLENEBILIRLIK (Sinirlama #4 icin kismi cozum): V buyudukce yogun
    (dense) VxV matris (orn. V=100k -> 80 GB) sıradan bir bilgisayarda
    imkansiz hale gelir. Bu yuzden co-occurrence sayaclari SCIPY SEYREK
    (sparse, CSR) matriste tutulur ve tam SVD yerine KIRPILMIS/RASTGELE
    SVD (scipy.sparse.linalg.svds, Halko ve ark. 2011 randomized SVD
    ailesinden) kullanilir - maliyet artik O(V^2) DEGIL, O(nnz * k)
    civarindadir (nnz = gercekte gorulen benzersiz cift sayisi, gercek
    dilde V^2'den COK kucuktur, Zipf yasasi geregi). Yine de V > birkac
    milyon oldugunda bu yaklasim da GERCEK bir tokenizer + dagitik/diskte
    islem hatti gerektirir - bu HALA bir sıradan-bilgisayar sinirini asar;
    burada saglanan sadece 'dense matristen sparse matrise' iyilestirmedir,
    sihirli bir cozum degildir."""

    def __init__(self, ridm, token_ids, window=None, k=None, seed=3, dense_threshold=4000):
        W = window or ridm.window
        V = ridm.V
        ids = np.asarray(token_ids, dtype=np.int64)
        n = len(ids)
        k = k or min(ridm.dim, V)
        k = max(1, min(k, V - 1 if V > 1 else 1))

        # es-olusum ciftlerini SEYREK bicimde biriktir (dense VxV DEGIL)
        rows, cols, data = [], [], []
        for d in range(1, W + 1):
            if n - d <= 0:
                continue
            rows.append(ids[d:])                 # 'sorgu' rolu (daha sonraki kelime)
            cols.append(ids[: n - d])             # 'anahtar' rolu (daha onceki kelime)
            data.append(np.ones(n - d))
        rows = np.concatenate(rows) if rows else np.array([], dtype=np.int64)
        cols = np.concatenate(cols) if cols else np.array([], dtype=np.int64)
        data = np.concatenate(data) if data else np.array([])

        try:
            import scipy.sparse as sp
            C = sp.coo_matrix((data, (rows, cols)), shape=(V, V)).tocsr()
            C.sum_duplicates()
            total = C.sum() + 1e-8
            row_sums = np.asarray(C.sum(axis=1)).ravel() / total + 1e-12
            col_sums = np.asarray(C.sum(axis=0)).ravel() / total + 1e-12
            C = C.tocoo()
            pij = C.data / total
            pmi = np.log(pij / (row_sums[C.row] * col_sums[C.col]))
            ppmi_data = np.clip(pmi, 0, None)
            keep = ppmi_data > 0
            ppmi = sp.coo_matrix((ppmi_data[keep], (C.row[keep], C.col[keep])), shape=(V, V)).tocsr()

            if V > dense_threshold and ppmi.nnz > 0:
                from scipy.sparse.linalg import svds
                k_eff = max(1, min(k, min(ppmi.shape) - 1))
                U_k, S_k, Vt_k = svds(ppmi, k=k_eff)
                order = np.argsort(S_k)[::-1]
                U_k, S_k, Vt_k = U_k[:, order], S_k[order], Vt_k[order, :]
            else:
                U, S, Vt = np.linalg.svd(ppmi.toarray(), full_matrices=False)
                U_k, S_k, Vt_k = U[:, :k], S[:k], Vt[:k, :]
        except ImportError:
            # scipy yoksa dense yola geri don (kucuk sozlukler icin guvenli)
            eff_V = min(V, dense_threshold)
            Cd = np.zeros((eff_V, eff_V))
            if len(rows):
                valid = (rows < eff_V) & (cols < eff_V)
                np.add.at(Cd, (rows[valid], cols[valid]), data[valid])
            total = Cd.sum() + 1e-8
            P_ij = Cd / total
            P_i = (Cd.sum(axis=1, keepdims=True) / total) + 1e-12
            P_j = (Cd.sum(axis=0, keepdims=True) / total) + 1e-12
            with np.errstate(divide="ignore"):
                pmi = np.log((P_ij + 1e-12) / (P_i @ P_j))
            ppmi_d = np.clip(pmi, 0, None)
            U, S, Vt = np.linalg.svd(ppmi_d, full_matrices=False)
            eff_k = min(k, eff_V)
            U_k_small, S_k, Vt_k_small = U[:, :eff_k], S[:eff_k], Vt[:eff_k, :]
            
            U_k = np.zeros((V, eff_k))
            Vt_k = np.zeros((eff_k, V))
            U_k[:eff_V, :] = U_k_small
            Vt_k[:, :eff_V] = Vt_k_small

        sqrt_s = np.sqrt(np.clip(S_k, 0, None))
        self.query_basis = U_k * sqrt_s        # (V,k) - 'hedef/sorgu' rolu vektorleri
        self.key_basis = (Vt_k.T) * sqrt_s      # (V,k) - 'baglam/anahtar' rolu vektorleri
        self.dim = k


class LearnedRelationAttention:
    """CooccurrenceRelationBasis'ten turetilen, GERCEKTEN VERIYE-UYUMLU
    (rastgele degil) Q/K rolleriyle calisan attention katmani. Skorlar
    dogrudan (query_basis[aday] . key_basis[baglam_kelimesi]) seklinde
    hesaplanir - Wq/Wk matris carpimi yerine, korpusta GERCEKTEN gozlemlenen
    ikili-iliski istatistiklerinin SVD-tabanindan okunur. Deger (value)
    kismi hala RIDM'in kelime gommelerini kullanir."""

    def __init__(self, ridm, relation_basis):
        self.ridm = ridm
        self.basis = relation_basis
        self._pe_cache = {}

    def _pe(self, n, dim):
        key = (n, dim)
        if key not in self._pe_cache:
            pos = np.arange(max(n, 1))[:, None]
            i = np.arange(dim)[None, :]
            angle = pos / np.power(10000.0, (2 * (i // 2)) / max(dim, 1))
            pe = np.zeros((max(n, 1), dim))
            pe[:, 0::2] = np.sin(angle[:, 0::2])
            pe[:, 1::2] = np.cos(angle[:, 1::2])
            self._pe_cache[key] = pe
        return self._pe_cache[key]

    def forward(self, context_token_ids):
        """Sozlukteki HER kelimeyi 'aday hedef' (query rolu) olarak
        skorlar; dondurulen (V,) vektorunun i. elemani, baglam verildiginde
        i. kelimenin ne kadar 'iliskili/beklenen' oldugunu gosterir."""
        ctx_ids = [t for t in context_token_ids if 0 <= t < self.ridm.V]
        if not ctx_ids:
            return np.zeros(self.ridm.V)
        keys = self.basis.key_basis[ctx_ids]                       # (n,k)
        keys = keys + self._pe(len(ctx_ids), keys.shape[1])
        queries = self.basis.query_basis                            # (V,k)
        scores = queries @ keys.T / math.sqrt(max(keys.shape[1], 1))  # (V,n)
        scores = scores - scores.max(axis=1, keepdims=True)
        w = np.exp(scores)
        w = w / (w.sum(axis=1, keepdims=True) + 1e-12)               # (V,n)
        values = self.ridm.context_vecs[ctx_ids]                     # (n,dim)
        attended = w @ values                                         # (V,dim) - her aday icin baglam-agirlikli deger
        cand_logits = np.einsum("vd,vd->v", attended, self.ridm.context_vecs)
        return cand_logits


# ======================================================
# 14) MULTI-SENSE (BAGLAM-DUYARLI) EMBEDDING
# ======================================================


class TransformerBlockStack:
    """Coklu-blok residual yigin: her blok = [Attention + Residual] ->
    [FFN(Reservoir) + Residual], v3/v4'teki TEK-hop yapisi yerine GPT'nin
    'katman katman islenen token' fikrine daha yakin bir mimari. Agirliklar
    hala backprop ile OGRENILMEZ (attention icin veri-turevli PMI-SVD tabani
    veya sabit rastgele projeksiyon; FFN icin sabit-rastgele tanh + kapali-
    form ridge okuma) ama artik GERCEK bir DERINLIK (L blok) ve RESIDUAL
    BAGLANTI (+basit RMS-norm benzeri olcekleme) vardir."""

    def __init__(self, dim, attention_layers, ffn_stack, n_blocks=3, seed=21):
        self.dim = dim
        self.attentions = attention_layers
        self.ffn = ffn_stack
        self.n_blocks = n_blocks
        rng = np.random.RandomState(seed)
        self.W_down = rng.randn(dim, ffn_stack.out_dim) / np.sqrt(ffn_stack.out_dim)

    @staticmethod
    def _norm(x):
        n = np.linalg.norm(x)
        return x / (n + 1e-8) * math.sqrt(len(x))

    def forward(self, context_vecs):
        if context_vecs.shape[0] == 0:
            return np.zeros(self.dim)
        x = context_vecs.sum(axis=0)
        seq = context_vecs.copy()
        for i in range(self.n_blocks):
            attn = self.attentions[i % len(self.attentions)]
            attn_out, _ = attn.forward(seq, query_vec=x)
            x = self._norm(x + attn_out)                       # residual + norm
            ffn_out = self.W_down @ self.ffn.encode(x)          # kapali-form dogrusal-olmayan FFN
            x = self._norm(x + ffn_out)                         # residual + norm
            seq = np.vstack([seq, x[None, :]])                  # derinlesen izi baglama ekle
        return x


# ======================================================
# 16) GERCEK BYTE-PAIR ENCODING (BPE) TOKENIZER
# ======================================================
