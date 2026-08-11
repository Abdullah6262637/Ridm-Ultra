"""CPU çıkarımı için dinamik int8 (veya float16) quantization.

``torch.quantization.quantize_dynamic``, ``nn.Linear`` ağırlıklarını
nicelendirir, aktivasyonları çalışma zamanında dinamik olarak ölçekler.
Yalnızca CPU backend'inde (fbgemm/qnnpack) desteklenir — GPU quantization
(GPTQ/AWQ/bitsandbytes) gerçek bir CUDA ortamı ve ek bağımlılıklar
gerektirir; bu sandbox'ta ikisi de yok, bu yüzden burada bağımlılıksız,
CPU-hedefli bir yol izlenir. GPU üzerinde servis vermek isteyen taraf
fp16/bf16 checkpoint'i doğrudan (nicelemeden) kullanabilir.
"""
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("Quantization için `pip install .[llm]` ile torch kurulmalıdır.") from exc


def _state_dict_bytes(state_dict) -> int:
    total = 0
    for tensor in state_dict.values():
        if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
            total += tensor.numel() * tensor.element_size()
    return total


def quantize_checkpoint_dynamic(checkpoint_path: str | Path, output_path: str | Path, *, dtype: str = "qint8") -> dict:
    """Bir eğitim/hizalama checkpoint'ini CPU çıkarımı için dinamik nicelemeye tabi tutar."""
    torch = _require_torch()
    from ..model.config import ModelConfig
    from ..model.transformer import DecoderOnlyTransformer

    if dtype not in {"qint8", "float16"}:
        raise ValueError("dtype 'qint8' veya 'float16' olmalıdır.")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ModelConfig(**state["model_config"])
    model = DecoderOnlyTransformer(model_config)
    model.load_state_dict(state["model"])
    model.eval()

    original_bytes = _state_dict_bytes(model.state_dict())
    torch_dtype = getattr(torch, dtype)
    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch_dtype)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    torch.save({"format_version": 1, "kind": "quantized", "quantization": {"method": "dynamic", "dtype": dtype},
                "model_config": asdict(model_config), "model": quantized.state_dict()}, temporary)
    os.replace(temporary, output_path)

    quantized_bytes = _state_dict_bytes(quantized.state_dict())
    return {"output": str(output_path), "original_bytes": original_bytes, "quantized_bytes": quantized_bytes,
            "reduction_ratio": (1 - quantized_bytes / original_bytes) if original_bytes else 0.0}


def load_quantized_model(checkpoint_path: str | Path):
    """Nicelendirilmiş bir checkpoint'i CPU çıkarımı için yükler. Döner: (model, model_config)."""
    torch = _require_torch()
    from ..model.config import ModelConfig
    from ..model.transformer import DecoderOnlyTransformer

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state.get("kind") != "quantized":
        raise ValueError("Bu checkpoint nicelendirilmiş değil; önce quantize_checkpoint_dynamic çalıştırın.")
    model_config = ModelConfig(**state["model_config"])
    model = DecoderOnlyTransformer(model_config)
    model.eval()
    torch_dtype = getattr(torch, state["quantization"]["dtype"])
    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch_dtype)
    quantized.load_state_dict(state["model"])
    quantized.eval()
    return quantized, model_config
