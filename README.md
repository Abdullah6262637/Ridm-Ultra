# RIDM Ultra v6

RIDM Ultra, kapalı-form dil modeli deneyleri için veri alma, eğitim,
doğrulama, değerlendirme, model kaydı ve donanım seçimini tek yapıda sunar.

## Hızlandırma katmanı

- `native/ridm_kernels.cpp`: C++17 çekirdek; AVX2/FMA ve AVX-512 çalışma
  zamanı dağıtımı, OpenMP paralelliği ve güvenli skaler geri dönüş.
- `ComputeBackend`: `auto`, `native`, `torch` veya `numpy` seçimi.
- `--device cuda`: PyTorch CUDA ile GPU üzerinde SVD ve matris işlemleri.
  CUDA yoksa işlem sessizce CPU'ya düşmez, açık hata verir.

Yerel çekirdeği derlemek için [native/README.md](native/README.md) adımlarını
izleyin. `native/ridm_kernels.dll` mevcut olduğunda `--backend auto` onu
otomatik seçer.

## Eğitim

```powershell
ridm-ultra --corpus-file ders_notlari.txt --profile balanced --backend auto --threads 8 --save-model model.npz
ridm-ultra --corpus-file ders_notlari.txt --backend torch --device cuda --save-model model.npz
```

Programatik, tekrarlanabilir eğitim akışı:

```python
from ridm_ultra import TextDataset, Trainer, TrainingConfig

data = TextDataset.from_file("ders_notlari.txt")
trainer = Trainer(TrainingConfig(dim=512, rank=128, batch_size=65536,
                                 backend="auto", threads=8))
result = trainer.fit(data)
trainer.save(result, "artifacts/ridm")
print(result.validation, result.test, result.backend)
```

`Trainer`, sıralı eğitim/doğrulama/test bölmesi, sözlük oluşturma,
batch sınırlarında bağlamı koruma, değerlendirme ve `model.npz` +
`training.json` kayıtlarını sağlar.

## Gerçek Transformer LLM eğitimi

LLM katmanı RIDM'den ayrıdır ve PyTorch + Rust `tokenizers` gerektirir:

```powershell
pip install -e ".[llm]"
python -m ridm_ultra.llm.cli tokenizer --data data\*.jsonl --output artifacts\tokenizer.json --vocab-size 32000
python -m ridm_ultra.llm.cli pretrain --data data\*.jsonl --tokenizer artifacts\tokenizer.json --output artifacts\llm-smoke --seq-len 1024 --hidden 512 --layers 8 --heads 8 --kv-heads 2
```

Yeni LLM hattı JSONL/düz metin shard'larını RAM'e tamamını yüklemeden okur,
belgeleri EOS ile sequence'lere paketler ve CUDA'da PyTorch'un
`scaled_dot_product_attention` mekanizması üzerinden FlashAttention seçimini
otomatik kullanır. Her checkpoint atomik yazılır; eğitim, token sayısına
bağlı warmup + cosine schedule ile devam eder.

LLM kaynak ağacı artık görev sınırlarına göre düzenlenmiştir:

```text
ridm_ultra/llm/
  data/      # tokenizer, streaming shard okuma, kalite/deduplikasyon
  model/     # config, preset, Transformer ve KV-cache
  runtime/   # eğitim, benchmark ve değerlendirme
```

Önce eğitim verisini denetlenebilir artifact'e dönüştürün:

```powershell
python -m ridm_ultra.llm.cli prepare-data --data raw\*.jsonl --output data\clean.jsonl --manifest data\clean.manifest.json --dedup-db data\dedup.sqlite
python -m ridm_ultra.llm.cli evaluate --data data\validation.jsonl --tokenizer artifacts\tokenizer.json --checkpoint artifacts\llm-smoke\checkpoint-latest.pt
```

`prepare-data`, Unicode normalizasyonu, e-posta/telefon/URL redaksiyonu,
kalite eşiği, exact SHA-256 tekrar temizliği ve SimHash-LSH yakın tekrar
kontrolünü disk tabanlı SQLite indeksinde uygular. Manifest, çıktı hash'ini,
kaynakları ve filtre istatistiklerini kaydeder.

Çok GPU'lu eğitimde aynı komutu `torchrun` ile başlatın; eğitim motoru rank,
world-size ve local-rank bilgilerini ortamdan alır:

```powershell
torchrun --standalone --nproc_per_node=4 -m ridm_ultra.llm.cli pretrain --data data\train-*.jsonl --tokenizer artifacts\tokenizer.json --output artifacts\llm-50m --seq-len 1024 --hidden 512 --layers 8 --heads 8 --kv-heads 2
```

Eğitilmiş checkpoint ile KV-cache kullanan üretim:

```powershell
python -m ridm_ultra.llm.cli generate --tokenizer artifacts\tokenizer.json --checkpoint artifacts\llm-smoke\checkpoint-latest.pt --prompt "Yapay zeka eğitimi" --max-new-tokens 120
```

Önce model/hardware uyumluluğunu gerçek forward-backward adımıyla sınayın:

```powershell
python -m ridm_ultra.llm.cli smoke-test --preset smoke-17m --device cuda --seq-len 256
python -m ridm_ultra.llm.cli smoke-test --preset turkish-50m --device cuda --seq-len 1024 --batch-size 1
```

## Dürüst donanım durumu

Bu çalışma alanındaki GCC kurulumu `libgomp.spec` eksik olduğu için mevcut
`ridm_kernels.dll` tek iş parçacıklı derlendi. Kaynak kod ve CMake hedefi
OpenMP'yi kullanıma hazırdır; tam MinGW/MSVC OpenMP kurulumu ile yeniden
derlendiğinde `ComputeBackend.info` çıktısında `openmp` görünür. CUDA için
uygun CUDA'lı PyTorch kurulumu gerekir.

## SFT, DPO, güvenlik/RAG/servis katmanları

`ridm_ultra/llm/` altına dört yeni alt paket eklendi:

- **`align/`** — prompt-maskeli SFT ve dondurulmuş referans modele göre DPO
  (tercih optimizasyonu). CLI: `sft`, `dpo`.
- **`safety/`** — model çıktısında PII sızıntı taraması, kategori bazlı
  red-team prova bataryası + anahtar-kelime tabanlı reddetme sınıflandırıcısı,
  ve bunları perplexity ile birleştiren checkpoint kabul testi. CLI:
  `safety-eval`. **Sınırlama:** reddetme sınıflandırıcısı sezgiseldir, kesin
  hüküm için insan incelemesinin yerini almaz.
- **`retrieval/`** — harici model indirmesi gerektirmeyen, bağımlılıksız
  Okapi BM25 indeksi ve RAG generate sarmalayıcısı. CLI: `rag-index`,
  `rag-generate`.
- **`serving/`** — CPU için dinamik int8 quantization ve FastAPI çıkarım
  servisi (`/health`, `/generate`, `/rag/generate`, API-key auth). CLI:
  `quantize`, `serve` (bkz. `pip install -e ".[llm,serving]"`).

Ayrıca `runtime/pilot.py`'deki GPU pilot kabul testi artık `pilot` komutuyla
CLI'den erişilebilir (önceden kodda vardı ama hiçbir komuta bağlı değildi).

Tüm bu katmanlar CPU üzerinde, küçük sentetik verilerle uçtan uca test
edildi — bkz. `examples/smoke/`. Gerçek ölçekli (1B-2,5B token, çok GPU)
koşum komutları ve her aşama için kabul kriterleri için `docs/RUNBOOK_GPU.md`
dosyasına bakın.
