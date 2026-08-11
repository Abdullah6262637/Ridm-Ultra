"""Bellek-içi Okapi BM25 tam metin arama indeksi.

RAG için harici model indirmesi (embedding modeli vb.) gerektirmez — bu
ortamda zaten HuggingFace/bulut erişimi yok, kullanıcının kendi eğitim
makinesinde de garanti değil. BM25, ek bağımlılık olmadan iyi bir taban
retrieval kalitesi verir; ileride embedding tabanlı bir indeksle
(ör. modelin kendi gizli durumlarıyla) değiştirilebilir/birleştirilebilir.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    score: float


class BM25Index:
    """Klasik Okapi BM25. Disk formatı JSON'dur; tek makinede orta ölçek RAG için yeterlidir."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 pozitif, b [0, 1] aralığında olmalıdır.")
        self.k1, self.b = k1, b
        self.documents: dict[str, Document] = {}
        self._doc_freqs: dict[str, dict[str, int]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._df: Counter = Counter()
        self._total_length = 0

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def _avg_doc_length(self) -> float:
        return self._total_length / len(self.documents) if self.documents else 0.0

    def add(self, document: Document) -> None:
        if document.id in self.documents:
            raise ValueError(f"Belge id'si zaten var: {document.id}")
        terms = tokenize(document.text)
        counts = Counter(terms)
        self.documents[document.id] = document
        self._doc_freqs[document.id] = dict(counts)
        self._doc_lengths[document.id] = len(terms)
        self._total_length += len(terms)
        for term in counts:
            self._postings[term].add(document.id)
            self._df[term] += 1

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self._df.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> list[ScoredDocument]:
        """Sorguyla en az bir terimi paylaşan belgeleri BM25 skoruna göre sıralar."""
        terms = tokenize(query)
        if not terms or not self.documents:
            return []
        candidates: set[str] = set()
        for term in terms:
            candidates |= self._postings.get(term, set())
        scores: dict[str, float] = {}
        for doc_id in candidates:
            doc_length = self._doc_lengths[doc_id]
            freqs = self._doc_freqs[doc_id]
            score = 0.0
            for term in terms:
                frequency = freqs.get(term, 0)
                if frequency == 0:
                    continue
                idf = self._idf(term)
                denominator = frequency + self.k1 * (1 - self.b + self.b * doc_length / max(1e-9, self._avg_doc_length))
                score += idf * (frequency * (self.k1 + 1)) / denominator
            if score > 0:
                scores[doc_id] = score
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [ScoredDocument(self.documents[doc_id], score) for doc_id, score in ranked]

    def save(self, path: str | Path) -> None:
        payload = {"format_version": 1, "k1": self.k1, "b": self.b,
                  "documents": [{"id": doc.id, "text": doc.text, "metadata": doc.metadata} for doc in self.documents.values()]}
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls(k1=payload.get("k1", 1.5), b=payload.get("b", 0.75))
        for record in payload["documents"]:
            index.add(Document(record["id"], record["text"], record.get("metadata", {})))
        return index

    @classmethod
    def from_jsonl(cls, files: Sequence[str | Path], *, text_field: str = "text", id_field: str | None = None,
                   k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        index = cls(k1=k1, b=b)
        counter, skipped = 0, 0
        for raw_path in files:
            path = Path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(str(path))
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Geçersiz JSONL: {path}:{line_number}") from exc
                    text = record.get(text_field)
                    if not isinstance(text, str) or not text.strip():
                        skipped += 1
                        continue
                    doc_id = str(record[id_field]) if id_field and id_field in record else f"doc-{counter}"
                    metadata = {key: value for key, value in record.items() if key not in {text_field, id_field}}
                    try:
                        index.add(Document(doc_id, text, metadata))
                    except ValueError:
                        skipped += 1
                        continue
                    counter += 1
        if skipped:
            print(f"[rag-index] {skipped} kayıt atlandı (boş metin veya çakışan id).", flush=True)
        return index
