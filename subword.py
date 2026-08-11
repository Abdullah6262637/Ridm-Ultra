"""Alt-kelime (subword) hashing tabanli OOV-toleransli vektorler."""
import numpy as np

from .utils import _stable_hash


class SubwordHasher:
    def __init__(self, dim, buckets=2 ** 16, n=3, seed=7):
        self.dim = dim
        self.buckets = buckets
        self.n = n
        rng = np.random.RandomState(seed)
        self.hash_emb = rng.randn(buckets, dim) / np.sqrt(dim)

    def _ngrams(self, word):
        w = f"<{word}>"
        n = self.n
        if len(w) <= n:
            return [w]
        return [w[i : i + n] for i in range(len(w) - n + 1)]

    def vector(self, word):
        idxs = [_stable_hash(g, self.buckets) for g in self._ngrams(word)]
        return self.hash_emb[idxs].mean(axis=0)


