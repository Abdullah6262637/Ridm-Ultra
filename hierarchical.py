"""Kisa/orta/uzun baglam katmanlari (Hierarchical Memory) ve baglam-
uzunlugunu dinamik secen Adaptive Window.
"""
import math

import numpy as np

from .constants import SENTENCE_BOUNDARY_TOKENS
from .core import RIDM
from .utils import _entropy


class HierarchicalContextMemory:
    def __init__(self, vocab, counts, windows=(2, 5, 15), **ridm_kwargs):
        self.windows = windows
        self.layers = [RIDM(vocab, counts=counts, window=w, **ridm_kwargs) for w in windows]

    def partial_fit(self, token_ids):
        for layer in self.layers:
            layer.partial_fit(token_ids)

    def finalize(self, k=64):
        for layer in self.layers:
            layer.finalize(k=k)

    def combined_probs(self, token_ids_full_context, temperature=1.0):
        dists = []
        confidences = []
        for layer in self.layers:
            ctx = token_ids_full_context[-layer.window :]
            p = layer.softmax_probs(ctx, temperature=temperature)
            dists.append(p)
            ent = _entropy(p)
            confidences.append(1.0 / (ent + 1e-3))
        confidences = np.array(confidences)
        weights = confidences / confidences.sum()
        mixed = np.zeros_like(dists[0])
        for w, p in zip(weights, dists):
            mixed += w * p
        return mixed, weights

    def evaluate(self, token_ids, max_samples=1000):
        max_w = max(self.windows)
        n = len(token_ids)
        if n <= max_w:
            return {"accuracy": float("nan"), "perplexity": float("nan"), "n_samples": 0}
        correct, nll_sum, count = 0, 0.0, 0
        step = max(1, (n - max_w) // max_samples) if max_samples else 1
        for i in range(max_w, n, step):
            ctx = token_ids[:i]
            target = token_ids[i]
            probs, _ = self.combined_probs(ctx)
            correct += int(np.argmax(probs) == target)
            nll_sum += -math.log(max(probs[target], 1e-12))
            count += 1
            if max_samples and count >= max_samples:
                break
        ppl = math.exp(nll_sum / count) if count else float("nan")
        return {"accuracy": correct / count if count else float("nan"), "perplexity": ppl, "n_samples": count}


# ======================================================
# 3) ADAPTIVE WINDOW
# ======================================================


def adaptive_context(token_ids, i, idx2word, min_w=2, max_w=15, boundary_words=SENTENCE_BOUNDARY_TOKENS):
    start = max(0, i - max_w)
    ctx = token_ids[start:i]
    cut = 0
    for j in range(len(ctx) - 1, -1, -1):
        w = idx2word.get(ctx[j], "")
        if w in boundary_words:
            cut = j + 1
            break
    trimmed = ctx[cut:]
    if len(trimmed) < min_w:
        trimmed = ctx[-min_w:] if len(ctx) >= min_w else ctx
    return trimmed


# ======================================================
# 4) GRAPH-BASED SEMANTIC MEMORY
# ======================================================
