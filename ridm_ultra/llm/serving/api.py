"""FastAPI tabanlı çıkarım servisi.

Tek checkpoint'i (nicelendirilmiş veya değil) bir kez belleğe yükler ve
``/generate`` ile isteğe bağlı ``/rag/generate`` uç noktalarını sunar.
Üretim ortamında bunun önüne bir reverse proxy + kimlik doğrulama katmanı
(ör. API anahtarı, mTLS) koymak kullanıcının sorumluluğundadır; burada
sağlanan ``api_key`` kontrolü yalnızca temel bir koruma katmanıdır.

NOT: Bu dosya kasıtlı olarak ``from __future__ import annotations``
kullanmaz. FastAPI, istek gövdesi/parametre ayrımını endpoint
imzalarındaki *gerçek* tip nesnelerinden çıkarır; postponed evaluation
(PEP 563) ile string'e çevrilen tipler, ``create_app`` içinde yerel olarak
tanımlanan Pydantic modelleri için çözümlenemez ve FastAPI sessizce
``request``'i query parametresi sanar (422 hatası verir).
"""
import os
from typing import Optional


def _require_serving_deps():
    try:
        import torch
        from fastapi import FastAPI, Header, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("Servis için `pip install .[llm] fastapi uvicorn` gerekir.") from exc
    return torch, FastAPI, Header, HTTPException, BaseModel, Field


def create_app(checkpoint_path: str, tokenizer_path: str, *, device: Optional[str] = None,
              quantized: bool = False, rag_index_path: Optional[str] = None, api_key: Optional[str] = None):
    """Yapılandırılmış bir FastAPI uygulaması döndürür. ``uvicorn`` ile çalıştırılır."""
    torch, FastAPI, Header, HTTPException, BaseModel, Field = _require_serving_deps()
    from ..data.tokenizer import HuggingFaceTokenizer
    from ..model.config import ModelConfig
    from ..model.transformer import DecoderOnlyTransformer
    from ..retrieval import BM25Index, retrieval_augmented_generate
    from ..safety.acceptance import build_generate_fn

    resolved_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = HuggingFaceTokenizer.from_file(tokenizer_path)

    if quantized:
        from .quantize import load_quantized_model
        if resolved_device.type != "cpu":
            raise ValueError("Dinamik int8 nicelenmiş modeller yalnızca CPU'da çalıştırılabilir.")
        model, model_config = load_quantized_model(checkpoint_path)
    else:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_config = ModelConfig(**state["model_config"])
        model = DecoderOnlyTransformer(model_config)
        model.load_state_dict(state["model"])
        model.to(resolved_device)
    model.eval()

    rag_index = BM25Index.load(rag_index_path) if rag_index_path else None
    required_key = api_key if api_key is not None else os.environ.get("RIDM_API_KEY")

    class GenerateRequest(BaseModel):
        prompt: str
        max_new_tokens: int = Field(default=200, ge=1, le=2048)
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)
        top_k: Optional[int] = Field(default=50, ge=1)
        top_p: Optional[float] = Field(default=0.95, gt=0.0, le=1.0)

    class GenerateResponse(BaseModel):
        text: str

    class RAGRequest(BaseModel):
        query: str
        top_k: int = Field(default=3, ge=1, le=20)
        max_new_tokens: int = Field(default=200, ge=1, le=2048)
        temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    app = FastAPI(title="RIDM Ultra Inference API")
    app.state.model, app.state.tokenizer, app.state.device = model, tokenizer, resolved_device
    app.state.model_config, app.state.rag_index = model_config, rag_index

    def _check_auth(x_api_key: Optional[str]) -> None:
        if required_key and x_api_key != required_key:
            raise HTTPException(status_code=401, detail="Geçersiz veya eksik X-API-Key.")

    @app.get("/health")
    def health():
        return {"status": "ok", "device": str(resolved_device), "quantized": quantized,
                "parameters_config": {"hidden_size": model_config.hidden_size, "n_layers": model_config.n_layers,
                                      "vocab_size": model_config.vocab_size, "max_seq_len": model_config.max_seq_len}}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest, x_api_key: Optional[str] = Header(default=None)):
        _check_auth(x_api_key)
        room = max(1, model_config.max_seq_len - 1)
        ids = [tokenizer.bos_id, *tokenizer.encode(request.prompt)]
        if len(ids) > room:
            ids = [tokenizer.bos_id, *ids[-(room - 1):]]
        with torch.no_grad():
            output = model.generate(torch.tensor([ids], dtype=torch.long, device=resolved_device),
                                    max_new_tokens=request.max_new_tokens, temperature=request.temperature,
                                    top_k=request.top_k, top_p=request.top_p, eos_id=tokenizer.eos_id)
        return GenerateResponse(text=tokenizer.decode(output[0, len(ids):].tolist()))

    @app.post("/rag/generate")
    def rag_generate(request: RAGRequest, x_api_key: Optional[str] = Header(default=None)):
        _check_auth(x_api_key)
        if rag_index is None:
            raise HTTPException(status_code=400, detail="Bu servis bir RAG indeksi olmadan başlatıldı (--rag-index verin).")
        generate_fn = build_generate_fn(model, tokenizer, resolved_device, max_new_tokens=request.max_new_tokens,
                                        temperature=request.temperature)
        result = retrieval_augmented_generate(rag_index, request.query, generate_fn, top_k=request.top_k)
        return result.to_dict()

    return app
