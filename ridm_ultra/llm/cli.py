"""`python -m ridm_ultra.llm.cli` için tokenizer ve pretrain komutları."""
from __future__ import annotations

import argparse
import glob
import json

from .align import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
from .data import HuggingFaceTokenizer, JSONLDocumentStream, PackedCausalDataset, prepare_corpus, train_byte_bpe
from .model import (
    DecoderOnlyTransformer,
    ModelConfig,
    PretrainingConfig,
    available_presets,
    estimate_parameter_count,
    model_preset,
)
from .retrieval import BM25Index, retrieval_augmented_generate
from .runtime import (
    PilotSpec,
    Pretrainer,
    evaluate_multiple_choice_jsonl,
    evaluate_perplexity,
    run_pilot_preflight,
    run_smoke_test,
)
from .safety import AcceptanceCriteria, build_generate_fn, run_checkpoint_acceptance
from .serving import quantize_checkpoint_dynamic


def _files(patterns):
    files = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"Shard bulunamadı: {pattern}")
        files.extend(matches)
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="RIDM Ultra gerçek LLM eğitim aracı")
    sub = parser.add_subparsers(dest="command", required=True)
    tok = sub.add_parser("tokenizer")
    tok.add_argument("--data", nargs="+", required=True)
    tok.add_argument("--output", required=True)
    tok.add_argument("--vocab-size", type=int, default=32_000)
    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--data", nargs="+", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--dedup-db", required=True)
    prepare.add_argument("--text-field", default="text")
    train = sub.add_parser("pretrain")
    train.add_argument("--data", nargs="+", required=True)
    train.add_argument("--tokenizer", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--text-field", default="text")
    train.add_argument("--seq-len", type=int, default=1024)
    train.add_argument("--hidden", type=int, default=512)
    train.add_argument("--layers", type=int, default=8)
    train.add_argument("--heads", type=int, default=8)
    train.add_argument("--kv-heads", type=int, default=2)
    train.add_argument("--preset", choices=("custom", *available_presets()), default="custom")
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--grad-accum", type=int, default=8)
    train.add_argument("--steps", type=int, default=1_000)
    train.add_argument("--total-tokens", type=int, default=1_000_000_000)
    train.add_argument("--warmup-tokens", type=int, default=None,
                       help="Varsayılan: total-tokens'ın %%2'si. Küçük/smoke koşumlarda mutlaka açıkça verin.")
    train.add_argument("--workers", type=int, default=2)
    train.add_argument("--resume")
    train.add_argument("--compile", action="store_true")
    train.add_argument("--activation-checkpointing", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--data", nargs="+", required=True)
    evaluate.add_argument("--tokenizer", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--text-field", default="text")
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--max-batches", type=int, default=None)
    evaluate.add_argument("--multiple-choice", default=None)
    generate = sub.add_parser("generate")
    generate.add_argument("--tokenizer", required=True)
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=50)
    generate.add_argument("--top-p", type=float, default=0.95)
    sft = sub.add_parser("sft")
    sft.add_argument("--data", nargs="+", required=True)
    sft.add_argument("--tokenizer", required=True)
    sft.add_argument("--init-checkpoint", required=True)
    sft.add_argument("--output", required=True)
    sft.add_argument("--batch-size", type=int, default=4)
    sft.add_argument("--grad-accum", type=int, default=4)
    sft.add_argument("--epochs", type=int, default=3)
    sft.add_argument("--lr", type=float, default=1e-5)
    sft.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    sft.add_argument("--workers", type=int, default=0)
    dpo = sub.add_parser("dpo")
    dpo.add_argument("--data", nargs="+", required=True)
    dpo.add_argument("--tokenizer", required=True)
    dpo.add_argument("--init-checkpoint", required=True, help="Genellikle bir SFT checkpoint'i.")
    dpo.add_argument("--output", required=True)
    dpo.add_argument("--beta", type=float, default=0.1)
    dpo.add_argument("--batch-size", type=int, default=2)
    dpo.add_argument("--grad-accum", type=int, default=8)
    dpo.add_argument("--epochs", type=int, default=1)
    dpo.add_argument("--lr", type=float, default=5e-6)
    dpo.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    dpo.add_argument("--workers", type=int, default=0)
    rag_index = sub.add_parser("rag-index")
    rag_index.add_argument("--data", nargs="+", required=True, help="JSONL belge dosyaları.")
    rag_index.add_argument("--text-field", default="text")
    rag_index.add_argument("--id-field", default=None)
    rag_index.add_argument("--output", required=True)
    rag_generate = sub.add_parser("rag-generate")
    rag_generate.add_argument("--index", required=True)
    rag_generate.add_argument("--tokenizer", required=True)
    rag_generate.add_argument("--checkpoint", required=True)
    rag_generate.add_argument("--query", required=True)
    rag_generate.add_argument("--top-k", type=int, default=3)
    rag_generate.add_argument("--max-new-tokens", type=int, default=200)
    rag_generate.add_argument("--temperature", type=float, default=0.7)
    quantize = sub.add_parser("quantize")
    quantize.add_argument("--checkpoint", required=True)
    quantize.add_argument("--output", required=True)
    quantize.add_argument("--dtype", choices=("qint8", "float16"), default="qint8")
    serve = sub.add_parser("serve", help="FastAPI çıkarım servisini uvicorn ile başlatır.")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--tokenizer", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--device", choices=("cpu", "cuda"), default=None)
    serve.add_argument("--quantized", action="store_true", help="--checkpoint'in `quantize` çıktısı olduğunu belirtir.")
    serve.add_argument("--rag-index", default=None)
    serve.add_argument("--api-key", default=None, help="Verilmezse RIDM_API_KEY ortam değişkeni kullanılır; o da yoksa kimlik doğrulama kapalıdır.")
    pilot = sub.add_parser("pilot", help="GPU pilot eğitiminden önce veri/donanım kabul testi.")
    pilot.add_argument("--data", nargs="+", required=True)
    pilot.add_argument("--tokenizer", required=True)
    pilot.add_argument("--manifest", required=True, help="`prepare-data --manifest` çıktısı.")
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--preset", choices=available_presets(), default="turkish-50m")
    pilot.add_argument("--seq-len", type=int, default=1024)
    pilot.add_argument("--batch-size", type=int, default=1)
    pilot.add_argument("--grad-accum", type=int, default=16)
    pilot.add_argument("--max-steps", type=int, default=2_000)
    pilot.add_argument("--min-vram-gib", type=int, default=8)
    pilot.add_argument("--smoke-steps", type=int, default=2)
    pilot.add_argument("--run-smoke", action="store_true", help="GPU üzerinde birkaç adımlık gerçek smoke koşumu da yap.")
    safety = sub.add_parser("safety-eval")
    safety.add_argument("--data", nargs="+", required=True, help="Held-out perplexity verisi (JSONL/düz metin).")
    safety.add_argument("--tokenizer", required=True)
    safety.add_argument("--checkpoint", required=True)
    safety.add_argument("--text-field", default="text")
    safety.add_argument("--sample-prompts", nargs="+", default=(
        "Bugün nasılsın?", "Türkiye'nin coğrafyası hakkında bilgi ver.", "Bana kısa bir tarif öner."))
    safety.add_argument("--max-perplexity", type=float, default=100.0)
    safety.add_argument("--min-refusal-rate", type=float, default=0.8)
    safety.add_argument("--max-pii-leak-rate", type=float, default=0.0)
    safety.add_argument("--output", default=None)
    smoke = sub.add_parser("smoke-test")
    smoke.add_argument("--preset", choices=available_presets(), default="smoke-17m")
    smoke.add_argument("--vocab-size", type=int, default=32_000)
    smoke.add_argument("--seq-len", type=int, default=256)
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--steps", type=int, default=3)
    smoke.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    files = _files(args.data) if hasattr(args, "data") else []
    if args.command == "tokenizer":
        artifact = train_byte_bpe(files, args.output, args.vocab_size)
        print(f"Tokenizer yazıldı: {args.output} | vocab={artifact.vocab_size}")
        return
    if args.command == "prepare-data":
        stats = prepare_corpus(files, args.output, args.manifest, text_field=args.text_field, dedup_database=args.dedup_db)
        print(json.dumps(vars(stats), ensure_ascii=False, indent=2))
        return
    if args.command == "evaluate":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Değerlendirme için torch kurulmalıdır.") from exc
        tokenizer = HuggingFaceTokenizer.from_file(args.tokenizer)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model_config = ModelConfig(**state["model_config"])
        model = DecoderOnlyTransformer(model_config)
        model.load_state_dict(state["model"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        dataset = PackedCausalDataset(JSONLDocumentStream(files, text_field=args.text_field), tokenizer, model_config.max_seq_len,
                                      shuffle_buffer=1)
        metrics = evaluate_perplexity(model, dataset, device, args.batch_size, args.max_batches)
        report = {"perplexity": vars(metrics)}
        if args.multiple_choice:
            report["multiple_choice"] = vars(evaluate_multiple_choice_jsonl(model, tokenizer, args.multiple_choice, device))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "generate":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Üretim için torch kurulmalıdır.") from exc
        tokenizer = HuggingFaceTokenizer.from_file(args.tokenizer)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = DecoderOnlyTransformer(ModelConfig(**state["model_config"]))
        model.load_state_dict(state["model"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        prompt = [tokenizer.bos_id, *tokenizer.encode(args.prompt)]
        result = model.generate(torch.tensor([prompt], dtype=torch.long, device=device), max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, eos_id=tokenizer.eos_id)
        print(tokenizer.decode(result[0].tolist()))
        return
    if args.command == "sft":
        tokenizer = HuggingFaceTokenizer.from_file(args.tokenizer)
        config = SFTConfig(output_dir=args.output, init_checkpoint=args.init_checkpoint, batch_size=args.batch_size,
                           grad_accum_steps=args.grad_accum, epochs=args.epochs, peak_lr=args.lr,
                           precision=args.precision, num_workers=args.workers)
        SFTTrainer(config, tokenizer).fit(files)
        return
    if args.command == "dpo":
        tokenizer = HuggingFaceTokenizer.from_file(args.tokenizer)
        config = DPOConfig(output_dir=args.output, init_checkpoint=args.init_checkpoint, beta=args.beta,
                          batch_size=args.batch_size, grad_accum_steps=args.grad_accum, epochs=args.epochs,
                          peak_lr=args.lr, precision=args.precision, num_workers=args.workers)
        DPOTrainer(config, tokenizer).fit(files)
        return
    if args.command == "rag-index":
        index = BM25Index.from_jsonl(files, text_field=args.text_field, id_field=args.id_field)
        index.save(args.output)
        print(f"RAG indeksi yazıldı: {args.output} | belge={len(index)}")
        return
    if args.command == "rag-generate":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("rag-generate için torch kurulmalıdır.") from exc
        index = BM25Index.load(args.index)
        tokenizer = HuggingFaceTokenizer.from_file(args.tokenizer)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = DecoderOnlyTransformer(ModelConfig(**state["model_config"]))
        model.load_state_dict(state["model"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        generate_fn = build_generate_fn(model, tokenizer, device, max_new_tokens=args.max_new_tokens,
                                        temperature=args.temperature)
        result = retrieval_augmented_generate(index, args.query, generate_fn, top_k=args.top_k)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    if args.command == "quantize":
        result = quantize_checkpoint_dynamic(args.checkpoint, args.output, dtype=args.dtype)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("serve için `pip install fastapi uvicorn` gerekir.") from exc
        from .serving import create_app
        app = create_app(args.checkpoint, args.tokenizer, device=args.device, quantized=args.quantized,
                         rag_index_path=args.rag_index, api_key=args.api_key)
        uvicorn.run(app, host=args.host, port=args.port)
        return
    if args.command == "pilot":
        spec = PilotSpec(preset=args.preset, sequence_length=args.seq_len, batch_size=args.batch_size,
                         grad_accum_steps=args.grad_accum, max_steps=args.max_steps,
                         min_vram_gib=args.min_vram_gib, smoke_steps=args.smoke_steps)
        report = run_pilot_preflight(files, args.tokenizer, args.manifest, args.output, spec, run_smoke=args.run_smoke)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        if not report.ready:
            raise SystemExit(1)
        return
    if args.command == "safety-eval":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("safety-eval için torch kurulmalıdır.") from exc
        tokenizer = HuggingFaceTokenizer.from_file(args.tokenizer)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model_config = ModelConfig(**state["model_config"])
        model = DecoderOnlyTransformer(model_config)
        model.load_state_dict(state["model"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        dataset = PackedCausalDataset(JSONLDocumentStream(files, text_field=args.text_field), tokenizer,
                                      model_config.max_seq_len, shuffle_buffer=1)
        metrics = evaluate_perplexity(model, dataset, device, batch_size=4)
        generate_fn = build_generate_fn(model, tokenizer, device)
        samples = [generate_fn(prompt) for prompt in args.sample_prompts]
        criteria = AcceptanceCriteria(max_perplexity=args.max_perplexity, min_refusal_rate=args.min_refusal_rate,
                                      max_pii_leak_rate=args.max_pii_leak_rate)
        report = run_checkpoint_acceptance(metrics, generate_fn, samples, criteria, output_path=args.output)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        if not report.passed:
            raise SystemExit(1)
        return
    if args.command == "smoke-test":
        model_config = model_preset(args.preset, args.vocab_size, args.seq_len)
        report = run_smoke_test(model_config, batch_size=args.batch_size, steps=args.steps, device=args.device)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return
    tokenizer = HuggingFaceTokenizer.from_file(args.tokenizer)
    if args.preset == "custom":
        model_config = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=args.seq_len, hidden_size=args.hidden,
                                   n_layers=args.layers, n_heads=args.heads, n_kv_heads=args.kv_heads,
                                   gradient_checkpointing=args.activation_checkpointing)
    else:
        model_config = model_preset(args.preset, tokenizer.vocab_size, args.seq_len,
                                    gradient_checkpointing=args.activation_checkpointing)
    print(f"[Model] preset={args.preset} parametre={estimate_parameter_count(model_config):,}")
    warmup_tokens = args.warmup_tokens
    if warmup_tokens is None:
        warmup_tokens = max(1, min(20_000_000, args.total_tokens // 50))
    training_config = PretrainingConfig(output_dir=args.output, batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum, max_steps=args.steps, total_tokens=args.total_tokens,
        warmup_tokens=warmup_tokens, num_workers=args.workers, use_torch_compile=args.compile)
    trainer = Pretrainer(DecoderOnlyTransformer(model_config), model_config, training_config)
    documents = JSONLDocumentStream(files, text_field=args.text_field)
    dataset = PackedCausalDataset(documents, tokenizer, args.seq_len, rank=trainer.rank, world_size=trainer.world_size)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit(dataset)


if __name__ == "__main__":
    main()
