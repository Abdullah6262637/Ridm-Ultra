#!/usr/bin/env bash
# Tüm LLM hattını (tokenizer -> pilot -> pretrain -> sft -> dpo -> safety-eval
# -> rag -> quantize) küçük sentetik verilerle CPU üzerinde uçtan uca çalıştırır.
# Gerçek eğitim değildir; sadece kodun/wiring'in bozuk olmadığını doğrular.
#
# Kullanım (proje kökünden):
#   pip install -e ".[llm]"
#   bash examples/smoke/run_pipeline.sh /tmp/ridm-smoke-out
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

D="${1:-/tmp/ridm-smoke-out}"
mkdir -p "$D"
CLI="python3 -m ridm_ultra.llm.cli"
EX="examples/smoke"

echo "== 0) sentetik korpus üret =="
python3 "$EX/generate_corpus.py"
mv "$EX/corpus.jsonl" "$D/corpus.jsonl"

echo "== 1) tokenizer =="
$CLI tokenizer --data "$D/corpus.jsonl" --output "$D/tok.json" --vocab-size 2000

echo "== 2) prepare-data =="
$CLI prepare-data --data "$D/corpus.jsonl" --output "$D/prepared.jsonl" \
  --manifest "$D/manifest.json" --dedup-db "$D/dedup.sqlite"

echo "== 3) pilot preflight (GPU yoksa ready:false beklenir; veri hattı yine de doğrulanır) =="
$CLI pilot --data "$D/prepared.jsonl" --tokenizer "$D/tok.json" --manifest "$D/manifest.json" \
  --output "$D/pilot" --preset smoke-17m --seq-len 128 || true

echo "== 4) pretrain (küçük özel model, CPU) =="
$CLI pretrain --data "$D/prepared.jsonl" --tokenizer "$D/tok.json" --output "$D/pretrain-out" \
  --preset custom --hidden 128 --layers 4 --heads 4 --kv-heads 2 --seq-len 128 \
  --batch-size 4 --grad-accum 2 --steps 40 --total-tokens 200000 --warmup-tokens 2000 --workers 0

echo "== 5) evaluate =="
$CLI evaluate --data "$D/prepared.jsonl" --tokenizer "$D/tok.json" \
  --checkpoint "$D/pretrain-out/checkpoint-latest.pt" --batch-size 4

echo "== 6) sft =="
$CLI sft --data "$EX/sft_data.jsonl" --tokenizer "$D/tok.json" \
  --init-checkpoint "$D/pretrain-out/checkpoint-latest.pt" --output "$D/sft-out" \
  --epochs 3 --batch-size 2 --grad-accum 1

echo "== 7) dpo =="
$CLI dpo --data "$EX/preference_data.jsonl" --tokenizer "$D/tok.json" \
  --init-checkpoint "$D/sft-out/sft-latest.pt" --output "$D/dpo-out" \
  --epochs 2 --batch-size 2 --grad-accum 1

echo "== 8) safety-eval (eşikler kasıtlı gevşek; küçük/az eğitimli model beklenen davranış) =="
$CLI safety-eval --data "$D/prepared.jsonl" --tokenizer "$D/tok.json" \
  --checkpoint "$D/dpo-out/dpo-latest.pt" \
  --max-perplexity 1e9 --min-refusal-rate 0.0 --max-pii-leak-rate 1.0 \
  --output "$D/safety-report.json" || true

echo "== 9) rag-index + rag-generate =="
$CLI rag-index --data "$EX/rag_docs.jsonl" --id-field id --output "$D/bm25.json"
$CLI rag-generate --index "$D/bm25.json" --tokenizer "$D/tok.json" \
  --checkpoint "$D/dpo-out/dpo-latest.pt" --query "Türkiye'nin başkenti neresidir?" \
  --top-k 2 --max-new-tokens 15

echo "== 10) quantize =="
$CLI quantize --checkpoint "$D/dpo-out/dpo-latest.pt" --output "$D/quantized.pt"

echo "== TÜM HAT BAŞARIYLA TAMAMLANDI: $D =="
