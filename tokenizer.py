"""Gercek Byte-Pair Encoding (BPE) tokenizer - Sinirlama #5 (tokenizer).

v3/v4'un kaba 'split()' tokenizasyonunun yerini alabilecek, sayim-tabanli
kapali-form bir alt-kelime tokenizeridir.
"""
from collections import Counter, defaultdict


class BPETokenizer:
    """Gercek Byte-Pair Encoding (Sennrich ve ark. 2016, 'Neural Machine
    Translation of Rare Words with Subword Units'). Kelime-ici karakter
    ciftlerini, en sik gorulenden baslayarak yinelemeli olarak BIRLESTIRIR
    (KAPALI-FORM/SAYIM-TABANLI bir islem; gradyan-inisi degildir). Turkce
    gibi zengin-eklemeli dillerde 'kitaplarimizdan' gibi kelimeleri kok+ek
    alt-parcalarina (yaklasik) bolerek OOV sorununu azaltir - v3/v4'un
    kaba 'split()' tokenizasyonunun yerini alabilir."""

    def __init__(self):
        self.merges = []
        self.vocab = set()

    def train(self, words, num_merges=200, min_pair_freq=2):
        word_freq = Counter(words)
        splits = {w: list(w) + ["</w>"] for w in word_freq}
        for _ in range(num_merges):
            pair_counts = Counter()
            for w, freq in word_freq.items():
                symbols = splits[w]
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_pair_freq:
                break
            self.merges.append(best_pair)
            merged = "".join(best_pair)
            for w in splits:
                symbols = splits[w]
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                        new_symbols.append(merged)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                splits[w] = new_symbols
        for syms in splits.values():
            self.vocab.update(syms)

    def encode_word(self, word):
        symbols = list(word) + ["</w>"]
        for a, b in self.merges:
            merged = a + b
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, words):
        out = []
        for w in words:
            out.extend(self.encode_word(w))
        return out


class SentencePieceBPETokenizer:
    """Google SentencePiece-compatible Byte-Pair Encoding (BPE) Subword Tokenizer (arXiv:1808.06226).

    Treats whitespace as a special character (' ' / '\\u2581') for raw sentence tokenization
    without requiring language-specific pre-tokenization.
    """

    SPIECE_UNDERLINE = " "  # U+2581

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []
        self.subword2id: dict[str, int] = {}
        self.id2subword: dict[int, str] = {}
        self.special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        for idx, st in enumerate(self.special_tokens):
            self.subword2id[st] = idx
            self.id2subword[idx] = st

    def train_on_sentences(self, sentences: list[str], num_merges: int = 1500, min_freq: int = 2):
        word_freq = Counter()
        for sentence in sentences:
            raw_words = sentence.strip().split()
            for w in raw_words:
                sp_word = self.SPIECE_UNDERLINE + w
                word_freq[sp_word] += 1

        splits = {w: tuple(w) for w in word_freq}

        # Initialize pair counts and pair-to-word index
        pair_counts = Counter()
        pair_to_words = defaultdict(set)

        for w, freq in word_freq.items():
            symbols = splits[w]
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_counts[pair] += freq
                pair_to_words[pair].add(w)

        for _ in range(num_merges):
            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_freq:
                break

            self.merges.append(best_pair)
            merged = "".join(best_pair)

            affected_words = list(pair_to_words[best_pair])
            del pair_counts[best_pair]
            del pair_to_words[best_pair]

            for w in affected_words:
                symbols = list(splits[w])
                freq = word_freq[w]

                # Decrement old pair counts for this word
                for i in range(len(symbols) - 1):
                    p = (symbols[i], symbols[i + 1])
                    if p in pair_counts:
                        pair_counts[p] -= freq
                        if pair_counts[p] <= 0:
                            del pair_counts[p]

                # Perform merge on symbols
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                        new_symbols.append(merged)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1

                splits[w] = tuple(new_symbols)

                # Increment new pair counts for this word
                for i in range(len(new_symbols) - 1):
                    p = (new_symbols[i], new_symbols[i + 1])
                    pair_counts[p] += freq
                    pair_to_words[p].add(w)

        # Rebuild vocabulary mapping
        vocab_set = set(self.special_tokens)
        for syms in splits.values():
            vocab_set.update(syms)

        next_id = len(self.special_tokens)
        for token in sorted(vocab_set):
            if token not in self.subword2id:
                self.subword2id[token] = next_id
                self.id2subword[next_id] = token
                next_id += 1

    def encode_word(self, word: str) -> list[str]:
        sp_word = self.SPIECE_UNDERLINE + word if not word.startswith(self.SPIECE_UNDERLINE) else word
        symbols = list(sp_word)
        for a, b in self.merges:
            merged = a + b
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode_as_subwords(self, text: str) -> list[str]:
        words = text.strip().split()
        subwords = []
        for w in words:
            subwords.extend(self.encode_word(w))
        return subwords

    def encode_as_ids(self, text: str) -> list[int]:
        subwords = self.encode_as_subwords(text)
        unk_id = self.subword2id.get("<unk>", 1)
        return [self.subword2id.get(sw, unk_id) for sw in subwords]

    def decode(self, subwords: list[str]) -> str:
        text = "".join(subwords)
        return text.replace(self.SPIECE_UNDERLINE, " ").strip()

    def to_dict(self) -> dict:
        return {
            "merges": self.merges,
            "subword2id": self.subword2id,
            "id2subword": self.id2subword,
        }

    def from_dict(self, d: dict):
        self.merges = [tuple(pair) for pair in d.get("merges", [])]
        self.subword2id = d.get("subword2id", {})
        self.id2subword = {int(k): v for k, v in d.get("id2subword", {}).items()}


# ======================================================
# 17) AUTOREGRESSIVE CALIBRATOR - kapali-form (ridge) next-token kalibrasyonu
# ======================================================
