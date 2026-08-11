"""Iliskisel bellek sistemleri: Sparse Distributed Memory (Kanerva) ve
Product-Key Memory (Lample ve ark. 2019). Her ikisi de 'yaz + unut +
pekistir' dongusunu (Sinirlama #8: memory forgetting/consolidation)
destekler - EZBERLEMEYIN: ikisi de benzer amaca (nadir iliski hatirlama)
hizmet eder; PKM cok daha buyuk kapasiteli, SDM daha basit/hizlidir.
Ayni anda ikisini de kullanmak REDUNDANT olabilir - bkz. cli.py profilleri.
"""
import numpy as np


class SparseDistributedMemory:
    """Kanerva'nin Sparse Distributed Memory fikrinin gercek-degerli (Hamming
    yerine kosinus benzerligi kullanan) versiyonu."""

    def __init__(self, addr_dim, content_dim, n_locations=512, activation_radius=0.25, seed=123):
        rng = np.random.RandomState(seed)
        self.hard_addresses = rng.randn(n_locations, addr_dim)
        self.hard_addresses /= np.linalg.norm(self.hard_addresses, axis=1, keepdims=True) + 1e-8
        self.content = np.zeros((n_locations, content_dim))
        self.write_counts = np.zeros(n_locations)
        self.radius = activation_radius
        self.n_locations = n_locations

    def _activated(self, address):
        a = address / (np.linalg.norm(address) + 1e-8)
        sims = self.hard_addresses @ a
        mask = sims >= (1 - self.radius)
        if not mask.any():
            top = np.argsort(sims)[::-1][:3]
            mask = np.zeros(self.n_locations, dtype=bool)
            mask[top] = True
        return mask, sims

    def write(self, address, content, weight=1.0):
        mask, _ = self._activated(address)
        self.content[mask] += weight * content
        self.write_counts[mask] += weight

    def read(self, address):
        mask, sims = self._activated(address)
        if not mask.any():
            return np.zeros(self.content.shape[1])
        counts = np.maximum(self.write_counts[mask], 1)[:, None]
        avg_content = self.content[mask] / counts
        w = np.clip(sims[mask], 1e-6, None)
        w = w / w.sum()
        return (avg_content * w[:, None]).sum(axis=0)

    def coverage(self):
        return float((self.write_counts > 0).mean())

    def decay(self, rate=0.98):
        """Ebbinghaus-tarzi UNUTMA egrisi: tum icerik/sayaclari sabit bir
        oranla kucultur. Periyodik cagrildikca eski/az-pekistirilen izler
        zayiflar - 'once (importance-agirlikli) yaz, sonra zamanla unut'
        dongusu (bkz. PKM.decay icin ayni fikir)."""
        self.content *= rate
        self.write_counts *= rate

    def consolidate(self, min_count=0.5):
        mask = self.write_counts < min_count
        count = int(mask.sum())
        self.content[mask] = 0.0
        self.write_counts[mask] = 0.0
        return count


# ======================================================
# RIDM cekirdegi (v2 + incremental SVD destegi)
# ======================================================


class ProductKeyMemory:
    """Buyuk-kapasiteli iliskisel bellek (Lample ve ark. 2019, 'Large Memory
    Layers with Product Keys' fikrinin kapali-form/egitimsiz versiyonu).
    N = n_sub^2 potansiyel 'hafiza hucresi' vardir ama HICBIRI onceden
    materyalize edilmez; sorgu iki yariya bolunur, her yari kucuk bir
    alt-anahtar kod defterinde (n_sub adet) en yakin komsulari aranir,
    capraz-carpimla (cartesian product) global adaylar O(n_sub) maliyetle
    (N=n_sub^2 yerine) uretilir. Sadece GERCEKTEN yazilan hucreler bir
    sozlukte saklanir (seyrek depolama) -> SDM'e gore cok daha fazla ayirt
    edici 'yuva' sunar, bellek patlamadan."""

    def __init__(self, dim, value_dim, n_sub=48, top_k=4, seed=17):
        self.half1 = dim // 2
        self.half2 = dim - self.half1
        rng = np.random.RandomState(seed)
        self.C1 = rng.randn(n_sub, self.half1)
        self.C1 /= np.linalg.norm(self.C1, axis=1, keepdims=True) + 1e-8
        self.C2 = rng.randn(n_sub, self.half2)
        self.C2 /= np.linalg.norm(self.C2, axis=1, keepdims=True) + 1e-8
        self.n_sub = n_sub
        self.top_k = top_k
        self.value_dim = value_dim
        self.values = {}
        self.counts = {}

    def _topk_codes(self, q):
        q1, q2 = q[: self.half1], q[self.half1 :]
        s1 = self.C1 @ q1
        s2 = self.C2 @ q2
        i1 = np.argsort(s1)[::-1][: self.top_k]
        i2 = np.argsort(s2)[::-1][: self.top_k]
        cands = []
        for a in i1:
            for b in i2:
                cands.append((s1[a] + s2[b], int(a), int(b)))
        cands.sort(key=lambda t: -t[0])
        return cands[: self.top_k]

    def write(self, q, value, weight=1.0):
        for score, i, j in self._topk_codes(q):
            key = (i, j)
            if key not in self.values:
                self.values[key] = np.zeros(self.value_dim)
                self.counts[key] = 0.0
            self.values[key] += weight * value
            self.counts[key] += weight

    def read(self, q):
        cands = self._topk_codes(q)
        total = 0.0
        acc = np.zeros(self.value_dim)
        for score, i, j in cands:
            key = (i, j)
            if key in self.values and self.counts[key] > 0:
                w = max(score, 1e-6)
                acc += w * (self.values[key] / self.counts[key])
                total += w
        if total <= 0:
            return np.zeros(self.value_dim)
        return acc / total

    def capacity(self):
        return self.n_sub * self.n_sub

    def utilization(self):
        return len(self.values) / max(1, self.capacity())

    def decay(self, rate=0.98):
        """Tum hucreleri sabit bir oranla zayiflatir (unutma egrisi)."""
        for key in self.values:
            self.values[key] *= rate
            self.counts[key] *= rate

    def consolidate(self, min_count=0.5):
        """PEKISTIRME + UNUTMA: az yazilmis (dusuk-onem, nadir erisilen)
        hucreleri budar; sik/onemli hucreler (yuksek sayac) korunur. Bu,
        'importance + recency' tabanli bellek yonetiminin kapali-form,
        kural-tabanli bir versiyonudur - gercek episodik/consolidation
        sistemleri kadar zengin degildir ama ayni ilkeyi (onemsizi unut,
        onemliyi pekistir) uygular."""
        weak = [key for key, c in self.counts.items() if c < min_count]
        for key in weak:
            del self.values[key]
            del self.counts[key]
        return len(weak)


# ======================================================
# 12) REASONING CHAIN - cok-adimli (multi-hop) rafine etme
# ======================================================
