"""Sozluk insasi, tokenlestirme ve egitim/test bolme yardimcilari."""
from collections import Counter

from .constants import UNK_TOKEN


def build_vocab(words, min_count=1, max_vocab=None):
    counts = Counter(words)
    items = [w for w, c in counts.items() if c >= min_count]
    items.sort(key=lambda w: (-counts[w], w))
    if max_vocab is not None:
        items = items[: max_vocab - 1]
    vocab = [UNK_TOKEN] + items
    return vocab, counts


def encode(words, word2idx):
    unk = word2idx[UNK_TOKEN]
    return [word2idx.get(w, unk) for w in words]


def train_test_split(token_ids, test_ratio=0.1):
    n = len(token_ids)
    split = max(1, int(n * (1 - test_ratio)))
    return token_ids[:split], token_ids[split:]
