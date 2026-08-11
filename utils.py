"""Genel yardimci fonksiyonlar (hashing, entropi, detokenizasyon)."""
import hashlib

import numpy as np

from .constants import SENTENCE_BOUNDARY_TOKENS, UNK_TOKEN


def _stable_hash(s, buckets):
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h, 16) % buckets


def _entropy(probs):
    p = np.clip(probs, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


# ======================================================
# Detokenizasyon - kelime-listesi (token dizisi) -> okunabilir cumle(ler)
# ======================================================
_NO_SPACE_BEFORE = SENTENCE_BOUNDARY_TOKENS | {",", ";", ":", ")", "]", "}"}
_NO_SPACE_AFTER = {"(", "[", "{"}


def detokenize(words):
    """Bosluk-ayrikli token listesini (ornegin HybridLM.generate() ciktisi)
    okunabilir, dogru noktalamali, cumle basi buyuk harfli metne cevirir.

    Onceden `generate()` ciktisi ' '.join(words) ile birlestiriliyordu; bu,
    noktalama isaretlerinden once fazladan bosluk birakiyor, cumle
    sinirlarini (SENTENCE_BOUNDARY_TOKENS) yok sayiyor ve hicbir buyuk/kucuk
    harf duzeltmesi yapmiyordu - uretilen metin gercek bir cumle gibi
    gorunmuyordu (kelimeler diziliyordu, cumle olusmumuyordu).
    """
    if not words:
        return ""
    parts = []
    capitalize_next = True
    wrote_any = False
    for w in words:
        if w == UNK_TOKEN:
            continue
        piece = w
        if capitalize_next and piece:
            piece = piece[0].upper() + piece[1:]
            capitalize_next = False
        if wrote_any and piece not in _NO_SPACE_BEFORE and parts[-1] not in _NO_SPACE_AFTER:
            parts.append(" ")
        parts.append(piece)
        wrote_any = True
        if w in SENTENCE_BOUNDARY_TOKENS:
            capitalize_next = True
    text = "".join(parts).strip()
    if text and text[-1] not in SENTENCE_BOUNDARY_TOKENS:
        text += "."
    return text


# ======================================================
# Alt-kelime hashing (v2'den)
# ======================================================
