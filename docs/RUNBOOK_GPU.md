# GPU Koşum Kılavuzu: 50M (1B token) → 125M (2,5B token)

Bu doküman, orijinal 7 maddelik plandaki her aşama için **gerçek bir GPU
makinesinde** çalıştırılacak tam komutları ve her aşamadan sonraki kabul
kriterlerini verir. Bu sandbox'ta (GPU yok, HuggingFace/bulut erişimi kapalı)
bu komutlar çalıştırılamadı; hepsi CPU üzerinde küçük sentetik veriyle
uçtan uca doğrulandı (bkz. `examples/smoke/`), gerçek ölçekte davranışları
teyit edilmedi. Sayısal hedefler (adım sayısı, VRAM, süre) kaba tahminlerdir;
kendi donanımınızda `pilot` ve `smoke-test` ile doğrulayın.

Tüm komutlar proje kökünden, `pip install -e ".[llm]"` sonrası çalıştırılır.

## 0) Ortam kurulumu

```bash
pip install -e ".[llm,serving]"
python -m ridm_ultra.llm.cli smoke-test --preset turkish-50m --device cuda --seq-len 1024
```

`smoke-test` birkaç gerçek forward/backward adımı çalıştırır; `tokens_per_second`
ve `peak_memory_bytes` çıktısından adım süresi ve VRAM bütçesini kabaca
kestirebilirsiniz.

## 1) Tokenizer + veri hazırlama + GPU pilot kabul testi

```bash
python -m ridm_ultra.llm.cli tokenizer --data corpus/raw-*.jsonl --output artifacts/tokenizer.json --vocab-size 32000

python -m ridm_ultra.llm.cli prepare-data --data corpus/raw-*.jsonl \
  --output corpus/prepared.jsonl --manifest corpus/manifest.json --dedup-db corpus/dedup.sqlite

python -m ridm_ultra.llm.cli pilot --data corpus/prepared.jsonl --tokenizer artifacts/tokenizer.json \
  --manifest corpus/manifest.json --output artifacts/pilot --preset turkish-50m --seq-len 1024 \
  --min-vram-gib 16 --run-smoke
```

**Kabul kriteri:** `pilot.preflight.json` içinde `"ready": true` ve `"issues": []`
olmalı. `run_smoke` etkinse rapordaki `smoke.tokens_per_second` değerinden
1B token için kaba süre tahmini: `1_000_000_000 / tokens_per_second` saniye.

## 2) Ana ön-eğitim: turkish-50m, ~1B token

Tek GPU:

```bash
python -m ridm_ultra.llm.cli pretrain --data corpus/prepared.jsonl --tokenizer artifacts/tokenizer.json \
  --output artifacts/pretrain-50m --preset turkish-50m --seq-len 1024 \
  --batch-size 8 --grad-accum 16 --total-tokens 1000000000 --steps 60000 --workers 4
```

Çoklu GPU (tek makine, N kart) — eğitim motoru `RANK`/`WORLD_SIZE`/`LOCAL_RANK`'i
`torchrun`'dan otomatik okur, kod değişikliği gerekmez:

```bash
torchrun --standalone --nproc_per_node=4 -m ridm_ultra.llm.cli pretrain \
  --data corpus/prepared.jsonl --tokenizer artifacts/tokenizer.json \
  --output artifacts/pretrain-50m --preset turkish-50m --seq-len 1024 \
  --batch-size 8 --grad-accum 4 --total-tokens 1000000000 --steps 60000 --workers 4
```

`--warmup-tokens` verilmezse `total-tokens`'ın ~%2'si (üst sınır 20M) kullanılır;
1B token bütçesi için varsayılan (20M) makuldür.

**Kabul kriteri:**
```bash
python -m ridm_ultra.llm.cli evaluate --data corpus/held-out.jsonl --tokenizer artifacts/tokenizer.json \
  --checkpoint artifacts/pretrain-50m/checkpoint-latest.pt --multiple-choice eval/mc-benchmark.jsonl
```
Perplexity, held-out veride kararlı biçimde düşüyor ve `smoke-test` ile aynı
hızda mı ilerliyor kontrol edin. Kesin bir "geçti" eşiği korpusa bağlıdır;
kendi held-out setinizle bir taban çizgi belirleyin.

## 3) SFT + tercih optimizasyonu (DPO)

```bash
python -m ridm_ultra.llm.cli sft --data sft/train-*.jsonl --tokenizer artifacts/tokenizer.json \
  --init-checkpoint artifacts/pretrain-50m/checkpoint-latest.pt --output artifacts/sft-50m \
  --epochs 3 --batch-size 16 --grad-accum 2 --lr 1e-5

python -m ridm_ultra.llm.cli dpo --data preference/train-*.jsonl --tokenizer artifacts/tokenizer.json \
  --init-checkpoint artifacts/sft-50m/sft-latest.pt --output artifacts/dpo-50m \
  --epochs 1 --batch-size 4 --grad-accum 8 --beta 0.1 --lr 5e-6
```

SFT verisi `{"prompt": "...", "response": "..."}`, tercih verisi
`{"prompt": "...", "chosen": "...", "rejected": "..."}` biçiminde JSONL olmalı.
DPO tipik olarak SFT checkpoint'i üzerine, düşük öğrenme oranıyla ve tek
epoch'la uygulanır — çok epoch veya yüksek `lr`, referans modelden aşırı
sapmaya (reward hacking benzeri davranış) yol açabilir.

**Kabul kriteri:** DPO log satırlarındaki `tercih_dogrulugu` (tercih edilen
yanıtın referansa göre daha olası bulunma oranı) eğitim ilerledikçe 0.5'in
üzerine çıkmalı; düşerse `lr`/`beta`'yı azaltın.

## 4) Güvenlik, PII ve red-team kabul testi

```bash
python -m ridm_ultra.llm.cli safety-eval --data corpus/held-out.jsonl --tokenizer artifacts/tokenizer.json \
  --checkpoint artifacts/dpo-50m/dpo-latest.pt \
  --max-perplexity 40 --min-refusal-rate 0.9 --max-pii-leak-rate 0.0 \
  --output artifacts/safety-report.json
```

Komut, eşiklerden biri geçilemezse çıkış kodu 1 ile başarısız olur (CI/CD
kapı testi olarak kullanılabilir). **Önemli sınırlama:** `refusal_rate`
anahtar-kelime tabanlı bir sezgiseldir, yalnızca "açık reddetme dili var mı"
sorusuna cevap verir — modelin isteği *örtük biçimde* yerine getirip
getirmediğini anlamaz. `"needs_review"` işaretli her örnek, checkpoint'i bir
sonraki aşamaya (servis) geçirmeden önce insan tarafından okunmalıdır. Bu
otomasyon insan incelemesinin yerini almaz, sadece ölçeklendirir.

## 5) RAG/retrieval entegrasyonu

```bash
python -m ridm_ultra.llm.cli rag-index --data knowledge/docs-*.jsonl --id-field doc_id --output artifacts/rag-index.json

python -m ridm_ultra.llm.cli rag-generate --index artifacts/rag-index.json --tokenizer artifacts/tokenizer.json \
  --checkpoint artifacts/dpo-50m/dpo-latest.pt --query "..." --top-k 4
```

Bu, harici model indirmesi gerektirmeyen BM25 tabanlı bir taban hattıdır.
Daha büyük bilgi tabanlarında (>~1M belge) bellek-içi indeks yetersiz kalır;
o noktada belge sayısına göre bir vektör veritabanına (FAISS, pgvector vb.)
geçmek gerekir — bu proje şu an onu içermiyor.

## 6) Quantization ve servis

```bash
python -m ridm_ultra.llm.cli quantize --checkpoint artifacts/dpo-50m/dpo-latest.pt --output artifacts/dpo-50m-int8.pt

RIDM_API_KEY=... python -m ridm_ultra.llm.cli serve --checkpoint artifacts/dpo-50m-int8.pt \
  --tokenizer artifacts/tokenizer.json --quantized --host 0.0.0.0 --port 8000 --rag-index artifacts/rag-index.json
```

Dinamik int8 nicelemesi yalnızca CPU'da çalışır (PyTorch kısıtı). GPU'da
servis vermek için nicelemesiz fp16/bf16 checkpoint'i `--checkpoint` olarak
verin ve `--quantized` bayrağını kullanmayın. `serve` komutu tek süreçte
uvicorn çalıştırır; üretimde bunun önüne bir reverse proxy (nginx/Caddy),
TLS ve gerçek bir kimlik doğrulama/hız sınırlama katmanı koyun —
`RIDM_API_KEY` yalnızca temel bir koruma sağlar.

## 7) turkish-125m'ye ölçekleme: çok GPU, ~2,5B token

```bash
torchrun --standalone --nproc_per_node=8 -m ridm_ultra.llm.cli pretrain \
  --data corpus/prepared.jsonl --tokenizer artifacts/tokenizer.json \
  --output artifacts/pretrain-125m --preset turkish-125m --seq-len 1024 \
  --batch-size 4 --grad-accum 8 --total-tokens 2500000000 --steps 150000 --workers 8
```

Çok makineli (multi-node) koşum için `torchrun --nnodes=... --node_rank=... --master_addr=...`
parametrelerini standart PyTorch dağıtık eğitim kurallarına göre ekleyin;
eğitim motoru zaten ortam tabanlı `RANK`/`WORLD_SIZE` okuduğu için ek kod
değişikliği gerekmez. 125M/2,5B kombinasyonu ~Chinchilla oranını hedefler
(bkz. `model/presets.py` docstring'i); 1B token ile 125M modeli eğitmek
teknik olarak mümkündür ama veri-altı-eğitime (undertraining) yol açar.

Sonrasında adım 3-6'yı `turkish-125m` checkpoint'i üzerinde tekrarlayın.

## Genel kabul akışı özeti

```
prepare-data → pilot (kapı 1: veri+donanım) → pretrain → evaluate
  → sft → dpo → safety-eval (kapı 2: kalite+güvenlik) → quantize → serve
```

Her "kapı" (`pilot`, `safety-eval`) çıkış kodu 1 ile başarısız olacak şekilde
tasarlanmıştır; bir CI/CD pipeline'ında `set -e` ile zincirlerseniz, eşik
geçilemeyen bir checkpoint otomatik olarak bir sonraki aşamaya geçemez.
