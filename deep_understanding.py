"""Derin Anlama ve Akıcı Cümle Oluşturma Motoru (Deep Understanding Engine)

RIDM Ultra'nın mevcut SVD/N-gram/Attention altyapısıyla entegre çalışarak
cümle kalitesini dramatik şekilde artıran modül.

Bileşenler:
  1. WordTypeModel     — Kelime türü geçiş olasılıkları (fonksiyon/içerik)
  2. CoherenceScorer   — SVD gömmeleriyle yerel tutarlılık ölçümü
  3. BeamCandidate     — Beam search için aday cümle yapısı
  4. DeepGenerator     — Beam search + gramer + tutarlılık + tekrar cezası
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from constants import SENTENCE_BOUNDARY_TOKENS, UNK_TOKEN


# ======================================================
# 1) KELİME TÜRÜ MODELİ (Gramer Kısıtlayıcı)
# ======================================================

# İngilizce'deki en yaygın fonksiyon kelimeleri (determiners, prepositions,
# pronouns, conjunctions, auxiliaries). Bunlar dışında kalan her şey
# "içerik kelimesi" (noun, verb, adjective, adverb) sayılır.
_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "of", "in", "to", "for", "on", "with", "at", "by",
    "from", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "shall", "should", "can",
    "could", "may", "might", "must", "and", "or", "but", "nor", "not", "no",
    "if", "then", "than", "that", "this", "these", "those", "it", "its",
    "he", "she", "we", "they", "i", "you", "me", "him", "her", "us", "them",
    "my", "your", "his", "our", "their", "which", "who", "whom", "whose",
    "what", "where", "when", "how", "why", "as", "so", "up", "out", "about",
    "into", "over", "after", "before", "between", "under", "through",
    "during", "without", "within", "along", "each", "every", "all", "both",
    "few", "more", "most", "other", "some", "such", "only", "own", "same",
    "too", "very", "just", "also", "now", "here", "there", "any", "many",
})

# Kelime türleri: F=fonksiyon, C=içerik, P=noktalama
WTYPE_FUNC = 0
WTYPE_CONTENT = 1
WTYPE_PUNCT = 2
_N_TYPES = 3


class WordTypeModel:
    """Kelime türü geçiş olasılıklarını öğrenir.

    Her kelimenin türünü belirler (fonksiyon/içerik/noktalama) ve
    ardışık tür geçiş istatistiklerini hesaplar. Üretim sırasında
    bir sonraki kelimenin beklenen türünü döner ve o türe uymayan
    adayların olasılığını azaltır.
    """

    def __init__(self):
        self.transitions = np.ones((_N_TYPES, _N_TYPES), dtype=np.float64)
        self.word_type_cache: dict[str, int] = {}

    @staticmethod
    def classify(word: str) -> int:
        """Kelimenin türünü belirler."""
        w = word.lower().rstrip(".,!?;:'\"")
        if not w:
            return WTYPE_PUNCT
        if w in _FUNCTION_WORDS:
            return WTYPE_FUNC
        if word in SENTENCE_BOUNDARY_TOKENS or word[-1] in ".!?":
            return WTYPE_PUNCT
        return WTYPE_CONTENT

    def fit(self, token_list: list[str]) -> None:
        """Korpustan kelime türü geçiş matrisini öğrenir."""
        types = [self.classify(w) for w in token_list]
        for i in range(len(types) - 1):
            self.transitions[types[i], types[i + 1]] += 1.0
        row_sums = self.transitions.sum(axis=1, keepdims=True)
        self.transitions = self.transitions / np.maximum(row_sums, 1e-12)
        for w in token_list:
            self.word_type_cache[w.lower().rstrip(".,!?;:'\"")] = self.classify(w)

    def next_type_probs(self, prev_word: str) -> np.ndarray:
        """Bir önceki kelimeye göre sonraki kelimenin tür olasılıklarını döner."""
        prev_type = self.classify(prev_word)
        return self.transitions[prev_type].copy()

    def type_boost(self, prev_word: str, candidates: np.ndarray,
                   idx2word: dict[int, str], strength: float = 2.0) -> np.ndarray:
        """Adayların olasılıklarını beklenen kelime türüne göre ayarlar."""
        type_probs = self.next_type_probs(prev_word)
        boosted = candidates.copy()
        for token_id in range(len(boosted)):
            if boosted[token_id] < 1e-12:
                continue
            word = idx2word.get(token_id, UNK_TOKEN)
            wtype = self.classify(word)
            boosted[token_id] *= (1.0 + strength * type_probs[wtype])
        total = boosted.sum()
        if total > 1e-12:
            boosted /= total
        return boosted


# ======================================================
# 2) TUTARLILIK PUANLAYICISI (Coherence Scorer)
# ======================================================

class CoherenceScorer:
    """SVD gömmeleri kullanarak yerel cümle tutarlılığını ölçer.

    Son N kelimenin ortalama vektörü ile aday kelimenin vektörü
    arasındaki kosinüs benzerliğini hesaplar. Yüksek tutarlılık,
    anlam bütünlüğü demektir.
    """

    def __init__(self, word_emb: np.ndarray, lookback: int = 5):
        self.word_emb = word_emb
        norms = np.linalg.norm(word_emb, axis=1, keepdims=True)
        self.word_emb_norm = word_emb / np.maximum(norms, 1e-8)
        self.lookback = lookback

    def score_candidates(self, recent_ids: list[int],
                         strength: float = 1.5) -> np.ndarray:
        """Son birkaç kelimenin bağlam vektörüne göre tüm adayları puanlar."""
        valid_ids = [tid for tid in recent_ids[-self.lookback:]
                     if 0 <= tid < len(self.word_emb)]
        if not valid_ids:
            return np.ones(len(self.word_emb), dtype=np.float64)

        context_vec = self.word_emb_norm[valid_ids].mean(axis=0)
        ctx_norm = np.linalg.norm(context_vec)
        if ctx_norm < 1e-8:
            return np.ones(len(self.word_emb), dtype=np.float64)
        context_vec /= ctx_norm

        similarities = self.word_emb_norm @ context_vec
        similarities = np.clip(similarities, -1.0, 1.0)
        boost = 1.0 + strength * (similarities + 1.0) / 2.0
        return boost


# ======================================================
# 3) BEAM SEARCH — ADAY CÜMLE YAPISI
# ======================================================

@dataclass
class BeamCandidate:
    """Beam search'te bir aday dizisini temsil eder."""
    token_ids: list[int] = field(default_factory=list)
    words: list[str] = field(default_factory=list)
    log_prob: float = 0.0
    coherence_sum: float = 0.0
    sentence_count: int = 0

    @property
    def score(self) -> float:
        """Normalize edilmiş toplam skor."""
        length = max(len(self.words), 1)
        avg_log_prob = self.log_prob / length
        avg_coherence = self.coherence_sum / length
        return avg_log_prob + 0.3 * avg_coherence

    def copy(self) -> BeamCandidate:
        return BeamCandidate(
            token_ids=list(self.token_ids),
            words=list(self.words),
            log_prob=self.log_prob,
            coherence_sum=self.coherence_sum,
            sentence_count=self.sentence_count,
        )


# ======================================================
# 4) DERİN ÜRETİCİ (Deep Generator)
# ======================================================

class DeepGenerator:
    """Beam search + gramer + tutarlılık + tekrar cezası ile
    akıcı ve anlamlı cümle üretir.

    Mevcut HybridLM'in olasılık motorunu kullanır ama üzerine:
      - Kelime türü geçiş kısıtlaması (WordTypeModel)
      - Yerel tutarlılık puanı (CoherenceScorer)
      - Agresif tekrar engelleme
      - Beam search (birden fazla aday paralel değerlendirme)
    ekler.
    """

    def __init__(self, hybrid_lm, word_type_model: WordTypeModel,
                 coherence_scorer: CoherenceScorer):
        self.hybrid = hybrid_lm
        self.wtm = word_type_model
        self.coherence = coherence_scorer

    def generate(self, seed_words: list[str], length: int = 40,
                 temperature: float = 0.8, beam_width: int = 5,
                 top_k: int = 15, num_sentences: int | None = 2,
                 repetition_penalty: float = 3.0,
                 coherence_strength: float = 1.5,
                 grammar_strength: float = 2.0) -> list[str]:
        """Beam search ile akıcı metin üretir."""
        w2i = self.hybrid.ridm.word2idx
        i2w = self.hybrid.ridm.idx2word
        unk = w2i[UNK_TOKEN]
        window = self.hybrid.ridm.window

        initial_ids = [w2i.get(w, unk) for w in seed_words]
        initial = BeamCandidate(
            token_ids=list(initial_ids),
            words=list(seed_words),
            log_prob=0.0,
        )
        beams = [initial]

        for step in range(length):
            all_candidates: list[BeamCandidate] = []

            for beam in beams:
                if num_sentences is not None and beam.sentence_count >= num_sentences:
                    all_candidates.append(beam)
                    continue

                ctx_ids = beam.token_ids[-window:] if beam.token_ids else []
                ctx_words = beam.words[-window:] if beam.words else []

                base_probs = self.hybrid.probs(
                    ctx_ids, temperature=temperature, context_words=ctx_words
                )
                base_probs = np.asarray(base_probs, dtype=np.float64)

                prev_word = beam.words[-1] if beam.words else "."
                base_probs = self.wtm.type_boost(
                    prev_word, base_probs, i2w, strength=grammar_strength
                )

                coherence_boost = self.coherence.score_candidates(
                    beam.token_ids[-5:], strength=coherence_strength
                )
                base_probs *= coherence_boost
                total = base_probs.sum()
                if total > 1e-12:
                    base_probs /= total

                if repetition_penalty > 1.0:
                    recent = beam.token_ids[-20:]
                    counts = Counter(recent)
                    for tid, cnt in counts.items():
                        if 0 <= tid < len(base_probs):
                            base_probs[tid] /= (repetition_penalty ** cnt)

                    if len(beam.words) >= 2:
                        last_bigram = (beam.words[-2].lower(), beam.words[-1].lower())
                        for token_id in range(len(base_probs)):
                            if base_probs[token_id] < 1e-12:
                                continue
                            candidate_word = i2w.get(token_id, UNK_TOKEN)
                            candidate_bigram = (beam.words[-1].lower(), candidate_word.lower())
                            for wi in range(len(beam.words) - 2):
                                existing_bigram = (beam.words[wi].lower(), beam.words[wi + 1].lower())
                                if candidate_bigram == existing_bigram:
                                    base_probs[token_id] *= 0.01
                                    break

                    total = base_probs.sum()
                    if total > 1e-12:
                        base_probs /= total

                base_probs[unk] = 0.0
                total = base_probs.sum()
                if total > 1e-12:
                    base_probs /= total

                if top_k and 0 < top_k < len(base_probs):
                    top_idx = np.argpartition(base_probs, -top_k)[-top_k:]
                    mask = np.zeros_like(base_probs)
                    mask[top_idx] = base_probs[top_idx]
                    s = mask.sum()
                    if s > 1e-12:
                        base_probs = mask / s

                top_indices = np.argsort(base_probs)[::-1][:beam_width]

                for token_id in top_indices:
                    prob = base_probs[token_id]
                    if prob < 1e-12:
                        continue

                    word = i2w.get(int(token_id), UNK_TOKEN)
                    if word == UNK_TOKEN:
                        continue

                    new_beam = beam.copy()
                    new_beam.token_ids.append(int(token_id))
                    new_beam.words.append(word)
                    new_beam.log_prob += np.log(max(prob, 1e-30))

                    coh = coherence_boost[token_id] if token_id < len(coherence_boost) else 0.0
                    new_beam.coherence_sum += coh

                    if word in SENTENCE_BOUNDARY_TOKENS or word.endswith(('.', '!', '?')):
                        new_beam.sentence_count += 1

                    all_candidates.append(new_beam)

            if not all_candidates:
                break

            all_candidates.sort(key=lambda c: c.score, reverse=True)
            beams = all_candidates[:beam_width]

            if num_sentences is not None:
                if all(b.sentence_count >= num_sentences for b in beams):
                    break

        best = max(beams, key=lambda c: c.score)
        return best.words[len(seed_words):]


# ======================================================
# 5) ENTEGRASYON FABRİKASI
# ======================================================

def build_deep_understanding(hybrid_lm, corpus_tokens: list[str]):
    """Derin anlama bileşenlerini oluşturur ve entegre bir
    DeepGenerator döner.

    Parametreler:
      hybrid_lm    — Eğitilmiş HybridLM örneği
      corpus_tokens — Eğitim korpusu token listesi

    Dönüş: (DeepGenerator, WordTypeModel, CoherenceScorer)
    """
    import time

    t0 = time.perf_counter()
    wtm = WordTypeModel()
    wtm.fit(corpus_tokens)
    t_wtm = time.perf_counter() - t0

    t0 = time.perf_counter()
    coherence = CoherenceScorer(
        hybrid_lm.ridm.word_emb, lookback=5
    )
    t_coh = time.perf_counter() - t0

    generator = DeepGenerator(hybrid_lm, wtm, coherence)

    print(f"[Derin Anlama] Kelime türü modeli eğitildi: {t_wtm:.3f} sn")
    print(f"[Derin Anlama] Tutarlılık puanlayıcısı hazır: {t_coh:.3f} sn")
    print(f"[Derin Anlama] Beam search üreticisi aktif "
          f"(beam_width=5, gramer+tutarlılık+tekrar_cezası)")

    return generator, wtm, coherence
