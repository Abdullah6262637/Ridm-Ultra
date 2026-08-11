"""SFT ve tercih (DPO) verisi için JSONL okuma, prompt maskesi ve batch birleştirme.

Kayıp yalnızca yanıt (completion) token'ları üzerinden hesaplanır: prompt
token'ları ``-100`` ile maskelenir. ``DecoderOnlyTransformer.forward`` iç
kaydırma yapmadığı için (bkz. ``runtime.trainer.Pretrainer``), burada da
``inputs = full[:-1]`` / ``targets = full[1:]`` kuralına uyulur.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence


def _require_torch():
    try:
        import torch
        from torch.utils.data import IterableDataset, get_worker_info
        return torch, IterableDataset, get_worker_info
    except ImportError as exc:
        raise RuntimeError("Hizalama eğitimi için `pip install .[llm]` ile torch kurulmalıdır.") from exc


def _read_jsonl(files: Sequence[str | Path]) -> Iterator[dict]:
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
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Geçersiz JSONL: {path}:{line_number}") from exc


def _build_masked_sequence(prompt_ids: list[int], response_ids: list[int], max_seq_len: int) -> tuple[list[int], list[int]]:
    full = (prompt_ids + response_ids)[: max_seq_len + 1]
    if len(full) < 2:
        raise ValueError("Örnek en az 2 token üretmelidir.")
    inputs, targets = full[:-1], full[1:]
    mask_len = len(prompt_ids) - 1
    if mask_len >= len(targets):
        raise ValueError("max_seq_len içinde yanıt için token kalmadı; prompt çok uzun veya max_seq_len çok küçük.")
    labels = [-100] * mask_len + targets[mask_len:]
    return inputs, labels


def encode_sft_example(tokenizer, prompt: str, response: str, max_seq_len: int) -> tuple[list[int], list[int]]:
    """Prompt+yanıtı tek dizide birleştirir; kayıp yalnızca yanıt token'larında hesaplanır."""
    prompt_ids = [tokenizer.bos_id, *tokenizer.encode(prompt)]
    response_ids = [*tokenizer.encode(response), tokenizer.eos_id]
    return _build_masked_sequence(prompt_ids, response_ids, max_seq_len)


def encode_preference_example(tokenizer, prompt: str, chosen: str, rejected: str,
                               max_seq_len: int) -> tuple[list[int], list[int], list[int], list[int]]:
    """Aynı prompt için tercih edilen/edilmeyen yanıtları ayrı ayrı maskeli diziye kodlar."""
    prompt_ids = [tokenizer.bos_id, *tokenizer.encode(prompt)]
    chosen_inputs, chosen_labels = _build_masked_sequence(prompt_ids, [*tokenizer.encode(chosen), tokenizer.eos_id], max_seq_len)
    rejected_inputs, rejected_labels = _build_masked_sequence(prompt_ids, [*tokenizer.encode(rejected), tokenizer.eos_id], max_seq_len)
    return chosen_inputs, chosen_labels, rejected_inputs, rejected_labels


class SFTDataset:
    """JSONL ``{"prompt", "response"}`` kayıtlarından maskeli SFT örnekleri üretir."""

    def __new__(cls, files: Sequence[str | Path], tokenizer, max_seq_len: int, *, prompt_field: str = "prompt",
                response_field: str = "response", shuffle_buffer: int = 4096, seed: int = 42,
                rank: int = 0, world_size: int = 1):
        torch, IterableDataset, get_worker_info = _require_torch()
        files = list(files)
        if not files:
            raise ValueError("En az bir SFT veri dosyası gerekir.")
        if max_seq_len < 8:
            raise ValueError("max_seq_len en az 8 olmalıdır.")

        class _Dataset(IterableDataset):
            def __iter__(self_nonlocal):
                import random
                info = get_worker_info()
                worker_id = info.id if info else 0
                workers = info.num_workers if info else 1
                logical_rank = rank * workers + worker_id
                logical_world = world_size * workers
                rng = random.Random(seed + logical_rank)
                buffer: list[tuple[list[int], list[int]]] = []
                for index, record in enumerate(_read_jsonl(files)):
                    if index % logical_world != logical_rank:
                        continue
                    prompt, response = record.get(prompt_field), record.get(response_field)
                    if not isinstance(prompt, str) or not isinstance(response, str) or not prompt.strip() or not response.strip():
                        continue
                    try:
                        example = encode_sft_example(tokenizer, prompt, response, max_seq_len)
                    except ValueError:
                        continue
                    buffer.append(example)
                    if len(buffer) < shuffle_buffer:
                        continue
                    yield buffer.pop(rng.randrange(len(buffer)))
                while buffer:
                    yield buffer.pop(rng.randrange(len(buffer)))

        return _Dataset()


class PreferenceDataset:
    """JSONL ``{"prompt", "chosen", "rejected"}`` kayıtlarından DPO örnekleri üretir."""

    def __new__(cls, files: Sequence[str | Path], tokenizer, max_seq_len: int, *, shuffle_buffer: int = 2048,
                seed: int = 42, rank: int = 0, world_size: int = 1):
        torch, IterableDataset, get_worker_info = _require_torch()
        files = list(files)
        if not files:
            raise ValueError("En az bir tercih veri dosyası gerekir.")
        if max_seq_len < 8:
            raise ValueError("max_seq_len en az 8 olmalıdır.")

        class _Dataset(IterableDataset):
            def __iter__(self_nonlocal):
                import random
                info = get_worker_info()
                worker_id = info.id if info else 0
                workers = info.num_workers if info else 1
                logical_rank = rank * workers + worker_id
                logical_world = world_size * workers
                rng = random.Random(seed + logical_rank)
                buffer = []
                for index, record in enumerate(_read_jsonl(files)):
                    if index % logical_world != logical_rank:
                        continue
                    prompt, chosen, rejected = record.get("prompt"), record.get("chosen"), record.get("rejected")
                    if not all(isinstance(value, str) and value.strip() for value in (prompt, chosen, rejected)):
                        continue
                    try:
                        example = encode_preference_example(tokenizer, prompt, chosen, rejected, max_seq_len)
                    except ValueError:
                        continue
                    buffer.append(example)
                    if len(buffer) < shuffle_buffer:
                        continue
                    yield buffer.pop(rng.randrange(len(buffer)))
                while buffer:
                    yield buffer.pop(rng.randrange(len(buffer)))

        return _Dataset()


def collate_sft_batch(examples, pad_id: int):
    """Değişken uzunluklu (input_ids, labels) örneklerini sağa pad'ler."""
    import torch
    length = max(len(input_ids) for input_ids, _ in examples)
    input_batch = [input_ids + [pad_id] * (length - len(input_ids)) for input_ids, _ in examples]
    label_batch = [labels + [-100] * (length - len(labels)) for _, labels in examples]
    return torch.tensor(input_batch, dtype=torch.long), torch.tensor(label_batch, dtype=torch.long)


def collate_preference_batch(examples, pad_id: int):
    """Chosen/rejected dizilerini kendi grupları içinde bağımsız pad'ler."""
    import torch

    def pad(sequences: list[list[int]], pad_value: int):
        length = max(len(sequence) for sequence in sequences)
        return torch.tensor([sequence + [pad_value] * (length - len(sequence)) for sequence in sequences], dtype=torch.long)

    chosen_inputs = pad([example[0] for example in examples], pad_id)
    chosen_labels = pad([example[1] for example in examples], -100)
    rejected_inputs = pad([example[2] for example in examples], pad_id)
    rejected_labels = pad([example[3] for example in examples], -100)
    return chosen_inputs, chosen_labels, rejected_inputs, rejected_labels
