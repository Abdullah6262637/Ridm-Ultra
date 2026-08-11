"""Klasik add-k yumusatmali n-gram taban cizgisi."""
import math
from collections import Counter, defaultdict

import numpy as np


class NgramBaseline:
    def __init__(self, vocab_size, n=3, add_k=0.1):
        self.vocab_size = vocab_size
        self.n = n
        self.add_k = add_k
        self.counts = defaultdict(Counter)

    def fit(self, token_ids):
        ctx_len = self.n - 1
        for i in range(ctx_len, len(token_ids)):
            ctx = tuple(token_ids[i - ctx_len : i])
            self.counts[ctx][token_ids[i]] += 1

    def probs(self, context_token_ids):
        ctx_len = self.n - 1
        ctx = tuple(context_token_ids[-ctx_len:]) if ctx_len > 0 else ()
        counter = self.counts.get(ctx, Counter())
        total = sum(counter.values())
        V = self.vocab_size
        denom = total + self.add_k * V
        probs = np.full(V, self.add_k / denom)
        for tok, c in counter.items():
            probs[tok] = (c + self.add_k) / denom
        return probs

    def evaluate(self, token_ids, max_samples=1000):
        ctx_len = self.n - 1
        n = len(token_ids)
        if n <= ctx_len:
            return {"accuracy": float("nan"), "perplexity": float("nan"), "n_samples": 0}
        correct, nll_sum, count = 0, 0.0, 0
        step = max(1, (n - ctx_len) // max_samples) if max_samples else 1
        for i in range(ctx_len, n, step):
            ctx = token_ids[i - ctx_len : i]
            target = token_ids[i]
            p = self.probs(ctx)
            correct += int(np.argmax(p) == target)
            nll_sum += -math.log(max(p[target], 1e-12))
            count += 1
            if max_samples and count >= max_samples:
                break
        ppl = math.exp(nll_sum / count) if count else float("nan")
        return {"accuracy": correct / count if count else float("nan"), "perplexity": ppl, "n_samples": count}


# ======================================================
# 9) KAPALI-FORM COK-BASLI SELF-ATTENTION (POZISYONEL)
# ======================================================
