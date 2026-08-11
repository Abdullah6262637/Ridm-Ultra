"""Retrieval-Augmented Generation: BM25 ile getirilen belgeleri prompt'a gömer.

Bu katman modele veya tokenizer'a bağlı değildir; ``model.generate`` ile aynı
çağrı sözleşmesini kullanan herhangi bir sarmalayıcıyla çalışır (bkz.
``safety.acceptance.build_generate_fn``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

from .index import BM25Index, ScoredDocument

_CONTEXT_TEMPLATE = "Aşağıdaki bağlamı kullanarak soruyu yanıtla. Bağlamda cevap yoksa, bilmediğini söyle.\n\n{context}\n\nSoru: {query}\nYanıt:"


def format_context(retrieved: list[ScoredDocument], *, max_chars: int = 2000) -> str:
    """Getirilen belgeleri numaralı, kaynak etiketli bir bağlam bloğuna dönüştürür."""
    blocks, used = [], 0
    for rank, scored in enumerate(retrieved, 1):
        text = scored.document.text.strip()
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rsplit(" ", 1)[0] + "…"
        blocks.append(f"[{rank}] (kaynak: {scored.document.id}) {text}")
        used += len(text)
    return "\n\n".join(blocks)


def build_rag_prompt(query: str, retrieved: list[ScoredDocument], *, max_context_chars: int = 2000) -> str:
    context = format_context(retrieved, max_chars=max_context_chars) if retrieved else "(İlgili belge bulunamadı.)"
    return _CONTEXT_TEMPLATE.format(context=context, query=query)


@dataclass
class RAGResult:
    query: str
    answer: str
    prompt: str
    sources: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def retrieval_augmented_generate(index: BM25Index, query: str, generate_fn: Callable[[str], str], *, top_k: int = 3,
                                 max_context_chars: int = 2000) -> RAGResult:
    """Sorguyu indeksten getirilen belgelerle zenginleştirip ``generate_fn`` çağırır.

    ``generate_fn``, tek bir prompt string'i alıp üretilen metni döndüren
    herhangi bir çağrılabilirdir; bu sayede gerçek model, quantized model
    veya uzak bir serving API'si arkasında saydam biçimde değiştirilebilir.
    """
    retrieved = index.search(query, top_k=top_k)
    prompt = build_rag_prompt(query, retrieved, max_context_chars=max_context_chars)
    answer = generate_fn(prompt)
    sources = [{"id": scored.document.id, "score": scored.score, "metadata": scored.document.metadata} for scored in retrieved]
    return RAGResult(query=query, answer=answer, prompt=prompt, sources=sources)
