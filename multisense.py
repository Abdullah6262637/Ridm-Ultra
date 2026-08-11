"""Baglam-kumeleme tabanli coklu-anlam (polysemy) embeddingleri
(Sinirlama #2: statik/baglamsiz embedding).
"""
from collections import defaultdict

import numpy as np


class MultiSenseEmbedding:
    """Coklu-anlam (polysemy) icin KAPALI-FORM kumeleme: bir kelimenin TUM
    gorulme baglamlarini (etrafindaki ortalama baglam vektorlerini) toplar,
    bunlari K kumeye ayirir (basit Lloyd k-means, sabit birkac iterasyon -
    bu bir GRADYAN DONGUSU degil, KAPALI bir EM-tipi kumelemedir; Reisinger
    & Mooney 2010 'Multi-Prototype Vector-Space Models of Word Meaning'
    fikrinin sadelestirilmis versiyonu). Her kume bir 'anlam prototipi'dir.
    Kullanim aninda, kelimenin GUNCEL baglamina EN YAKIN prototip secilir -
    boylece 'banka (nehir kiyisi)' ile 'banka (finans kurumu)' farkli
    vektorlere ayrisabilir (yeterli/cesitli ornekle karsilasilirsa)."""

    def __init__(self, ridm, token_ids, n_senses=2, min_occurrences=6, seed=13):
        self.ridm = ridm
        self.n_senses = n_senses
        self.prototypes = {}
        W = ridm.window
        ids = np.asarray(token_ids)
        n = len(ids)
        if n <= W:
            return
        occ_contexts = defaultdict(list)
        for i in range(W, n):
            target = int(ids[i])
            ctx_ids = [t for t in ids[i - W : i] if 0 <= t < ridm.V]
            if not ctx_ids:
                continue
            occ_contexts[target].append(ridm.context_vecs[ctx_ids].mean(axis=0))
        rng = np.random.RandomState(seed)
        for word_id, ctx_list in occ_contexts.items():
            if len(ctx_list) < min_occurrences:
                continue
            X = np.array(ctx_list)
            k = min(n_senses, len(X))
            centers = X[rng.choice(len(X), size=k, replace=False)]
            for _ in range(5):
                d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
                assign = np.argmin(d, axis=1)
                new_centers = np.array([
                    X[assign == c].mean(axis=0) if np.any(assign == c) else centers[c]
                    for c in range(k)
                ])
                if np.allclose(new_centers, centers):
                    centers = new_centers
                    break
                centers = new_centers
            # ayni 'anlam' etrafinda kumelenmis merkezleri birlestir (gercekten ayrisik mi?)
            if k > 1 and np.linalg.norm(centers[0] - centers[1]) < 1e-6:
                centers = centers[:1]
            self.prototypes[word_id] = centers

    def disambiguate(self, word_id, local_context_vec):
        if word_id not in self.prototypes:
            return self.ridm.context_vecs[word_id]
        centers = self.prototypes[word_id]
        d = np.linalg.norm(centers - local_context_vec[None, :], axis=1)
        return centers[np.argmin(d)]

    def sense_count(self, word_id):
        return len(self.prototypes.get(word_id, []))


# ======================================================
# 15) TRANSFORMER BLOCK STACK - coklu (residual) attention+FFN blogu
# ======================================================
