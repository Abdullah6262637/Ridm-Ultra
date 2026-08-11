"""Held-out perplexity ve olasılık-temelli çoktan seçmeli LLM değerlendirmesi."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
        return torch, DataLoader
    except ImportError as exc:
        raise RuntimeError("LLM değerlendirmesi için torch kurulmalıdır.") from exc


@dataclass(frozen=True)
class LanguageModelMetrics:
    loss: float
    perplexity: float
    accuracy: float
    tokens: int


@dataclass(frozen=True)
class MultipleChoiceMetrics:
    accuracy: float
    items: int


def evaluate_perplexity(model, dataset, device, batch_size: int = 4, max_batches: int | None = None) -> LanguageModelMetrics:
    torch, DataLoader = _require_torch()
    loader = DataLoader(dataset, batch_size=batch_size)
    was_training = model.training
    model.eval()
    nll, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for batch_index, packed in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            packed = packed.to(device)
            inputs, labels = packed[:, :-1], packed[:, 1:]
            output = model(inputs, labels)
            n_tokens = labels.numel()
            nll += float(output.loss) * n_tokens
            correct += int((output.logits.argmax(dim=-1) == labels).sum())
            count += n_tokens
    if was_training:
        model.train()
    if count == 0:
        raise ValueError("Değerlendirme kümesi örnek üretmedi.")
    loss = nll / count
    return LanguageModelMetrics(loss, math.exp(min(loss, 20.0)), correct / count, count)


def score_continuation(model, tokenizer, prompt: str, continuation: str, device) -> float:
    """Bir seçeneğin token başına ortalama log-olasılığını otoregresif hesaplar."""
    torch, _ = _require_torch()
    context = tokenizer.encode(prompt)
    continuation_ids = tokenizer.encode(continuation)
    if not continuation_ids:
        return float("-inf")
    total = 0.0
    model.eval()
    with torch.no_grad():
        for token in continuation_ids:
            if not context:
                context = [tokenizer.bos_id]
            inputs = torch.tensor([context[-model.config.max_seq_len:]], dtype=torch.long, device=device)
            logits = model(inputs).logits[0, -1]
            total += float(torch.log_softmax(logits.float(), dim=-1)[token])
            context.append(token)
    return total / len(continuation_ids)


def evaluate_multiple_choice_jsonl(model, tokenizer, path: str | Path, device) -> MultipleChoiceMetrics:
    correct, total = 0, 0
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item.get("question"), str) or not isinstance(item.get("choices"), list):
                raise ValueError(f"Geçersiz çoktan-seçmeli örnek: {path}:{line_number}")
            scores = [score_continuation(model, tokenizer, item["question"], str(choice), device) for choice in item["choices"]]
            prediction = max(range(len(scores)), key=scores.__getitem__)
            correct += int(prediction == int(item["answer"]))
            total += 1
    if not total:
        raise ValueError("Çoktan-seçmeli dosyada örnek yok.")
    return MultipleChoiceMetrics(correct / total, total)
