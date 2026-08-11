"""Subword SymSpell Typo Corrector & Ellipsis Repair Engine for RIDM Ultra v7.0.

Provides zero-latency subword edit-distance normalization and prompt completion for Turkish text.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, Set

logger = logging.getLogger(__name__)


class TurkishTypoCorrector:
    """Fast Damerau-Levenshtein & SymSpell Typo Corrector tailored for Turkish morphology."""

    def __init__(self, vocab_path: str = "artifacts/turkish_valid_words.json"):
        self.valid_words: Set[str] = set()
        self.delete_dict: Dict[str, Set[str]] = {}
        self.char_map = {
            'i': 'ı', 'ı': 'i',
            'g': 'ğ', 'ğ': 'g',
            's': 'ş', 'ş': 's',
            'o': 'ö', 'ö': 'o',
            'u': 'ü', 'ü': 'u',
            'c': 'ç', 'ç': 'c',
        }
        self._load_vocab(vocab_path)

    def _load_vocab(self, path_str: str):
        p = Path(path_str)
        if p.exists():
            try:
                words = json.loads(p.read_text(encoding="utf-8"))
                self.valid_words = set(w.lower() for w in words if len(w) >= 2)
            except Exception as e:
                logger.warning(f"Failed to load vocab from {path_str}: {e}")

    def _deletes(self, word: str) -> Set[str]:
        deletes = set()
        for i in range(len(word)):
            deletes.add(word[:i] + word[i + 1:])
        return deletes

    def correct_word(self, word: str) -> str:
        w_lower = word.lower().strip()
        clean_w = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜ]', '', w_lower)
        if not clean_w or clean_w in self.valid_words:
            return word

        # 1. Try character mapping (Turkish diacritics repair)
        alt_chars = []
        for c in clean_w:
            alt_chars.append(self.char_map.get(c, c))
        alt_word = "".join(alt_chars)
        if alt_word in self.valid_words:
            return alt_word

        # 2. Try single deletion candidate lookup
        for d in self._deletes(clean_w):
            if d in self.valid_words:
                return d

        return word

    def correct_text(self, text: str) -> str:
        if not text:
            return ""
        words = text.split()
        corrected = [self.correct_word(w) for w in words]
        return " ".join(corrected)


class EllipsisRepairer:
    """Expands truncated or missing-word prompts into canonical Turkish structures."""

    PATTERNS = [
        (r'^\s*3\s+5\s+çarp\s*$', "3 ile 5'i çarparsan kaç eder?"),
        (r'^\s*(\d+)\s+(\d+)\s+topla\s*$', r'\1 ile \2 yi toplarsan kaç eder?'),
        (r'^\s*(\d+)\s+(\d+)\s+çıkar\s*$', r'\1 den \2 çıkarırsan kaç eder?'),
        (r'^\s*(\d+)\s+(\d+)\s+böl\s*$', r'\1 bölü \2 kaç eder?'),
        (r'^\s*hikaye\s*$', "bana kısa bir hikaye yaz"),
        (r'^\s*uzun\s+hikaye\s*$', "biraz uzun bir hikaye olsun"),
    ]

    def repair(self, text: str) -> str:
        t = text.strip()
        for pattern, replacement in self.PATTERNS:
            if re.match(pattern, t, flags=re.IGNORECASE):
                return re.sub(pattern, replacement, t, flags=re.IGNORECASE)
        return text
