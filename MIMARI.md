# RIDM Ultra v6 — Sistem Mimari Dokümantasyonu (ARCHITECTURE)

Bu doküman, **RIDM Ultra v6** projesinin uçtan uca tüm yazılım ve sistem mimarisini, tasarım kararlarını, bileşenler arası veri akışlarını ve **tüm dosyaların işlevlerini** detaylı ve kapsamlı bir şekilde açıklamak üzere hazırlanmıştır.

---

## 1. Genel Mimari Bakış ve Tasarım Felsefesi

**RIDM Ultra (Random Indexing & Deterministic Model)**, iki temel sistem katmanından oluşan hibrit bir dil modeli ve LLM altyapısıdır:

1. **RIDM Çekirdek Sistemi (Kapalı-Form / SVD Tabatlı Hibrit Model)**:
   - Geleneksel yinelemeli (iterative SGD/backprop) eğitim yükü gerektirmeksizin, rastgele indeksleme (Random Indexing), co-occurrence matris biriktirmesi ve kapalı-form Kesilmiş SVD (Truncated SVD) ile kelime vektörü ve bağlam alanlarını tek adımda oluşturur.
   - N-gram istatistikleri, ilişkisel bellek (SDM/PKM), anlamsal grafikler, kapalı-form self-attention, derin rezervuar ağları (ELM) ve entropi tabanlı dinamik çok adımlı çıkarım zincirlerini (Reasoning Controller) bir araya getiren çarpımsal bir enstrüman (HybridLM) sunar.
2. **RIDM Ultra LLM Katmanı (PyTorch / CUDA Transformer Ekosistemi)**:
   - Gerçek ölçekli (50M - 2.5B+ token) Transformer dil modelleri için veri hazırlama, akışlı okuma (streaming data pipeline), temizlik/deduplikasyon (SHA256 & SimHash-LSH), FlashAttention (PyTorch `scaled_dot_product_attention`), RoPE, GQA, SwiGLU, RMSNorm ve KV-Cache destekli üretim mimarisi sunar.
   - SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization), PII ve Red-Team güvenlik paketleri, sıfır-bağımlılıklı Okapi BM25 RAG ve int8 dinamik kuantizasyonlu FastAPI sunucu katmanlarını kapsar.

---

## 2. Sistem Bileşen Haritası ve Mimari Şema

```
+-----------------------------------------------------------------------------------+
|                                  CLI & ENTRYPOINTS                                |
|    ridm-ultra (cli.py)           |           python -m ridm_ultra.llm.cli         |
+----------------------------------+------------------------------------------------+
                                   |
         +-------------------------+-------------------------+
         |                                                   |
         v                                                   v
+----------------------------------+       +----------------------------------------+
|       RIDM CORE ECOSYSTEM        |       |       RIDM ULTRA LLM SUBSYSTEM         |
|  (Closed-Form & Hybrid Engine)   |       |    (PyTorch / CUDA Transformer Engine)  |
+----------------------------------+       +----------------------------------------+
| - core.py (RIDM SVD Engine)      |       | - model/ (Transformer, GQA, FlashAttn)  |
| - backend.py (CPU/CUDA Dispatch) |       | - data/ (Streaming, Quality, SimHash)  |
| - attention.py (CF-Attn & PPMI)  |       | - runtime/ (Trainer, DDP, Eval, Pilot) |
| - memory.py (SDM & PKM)          |       | - align/ (SFT & DPO Preference)        |
| - reasoning.py (Multi-Hop Chain) |       | - safety/ (PII, Probes, Acceptance)    |
| - hybrid.py (HybridLM Ensemble)  |       | - retrieval/ (BM25 Index, RAG Pipeline)|
| - native/ (C++ AVX2/512 Kernels) |       | - serving/ (Dynamic Quant, FastAPI)    |
+----------------------------------+       +----------------------------------------+
```

---

## 3. Dosya Dosya Detaylı Sistem Analizi

### 3.1. Kök Dizin & RIDM Çekirdek Modülleri (Root Modules)

#### 1. [`__init__.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/__init__.py)
- **Açıklama**: Paketin ana giriş noktasıdır.
- **İşlev**: Paket dışına aktarılan ana sınıfları (`RIDM`, `TextDataset`, `Trainer`, `TrainingConfig`, `ComputeBackend`, `HybridLM` vb.) export eder ve versiyon bilgisini sunar.

#### 2. [`core.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/core.py)
- **Açıklama**: RIDM mimarisinin kalbini oluşturan kapatılmış formlu (closed-form) dil modelidir.
- **Detay**:
  - `RIDM` sınıfı: Rastgele indekslenmiş bağlam vektörleri (`context_vecs`) ve hedef kelimelerin birliktelik matrisini (`M`) oluşturur.
  - `partial_fit`: `backend.py` üzerindeki C++ veya CUDA hızlandırılmış `accumulate_contexts` metodunu çağırarak matrisi akışlı olarak günceller.
  - `finalize`: Normalize edilmiş matrise Truncated SVD uygulayarak $W_{emb} = U_k \cdot S_k$ kelime gömmelerini çıkarır.
  - `incremental_update`: Brand & Hall yöntemiyle SVD ayrışımını sıfırdan hesaplamadan artımlı (online) olarak günceller.
  - `drift_estimate`: Yeniden hesaplanan SVD ile mevcut durum arasındaki sapmayı (drift) ölçer.

#### 3. [`backend.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/backend.py)
- **Açıklama**: Donanım farkındalıklı matris ve vektör hesaplama katmanıdır.
- **Detay**:
  - `ComputeBackend` sınıfı: `auto`, `native`, `torch` veya `numpy` seçeneklerine göre hesaplama yükünü yönlendirir.
  - `native/ridm_kernels.dll` varsa ctypes üzerinden doğrudan AVX2/AVX-512 destekli C++ kodunu çağırır.
  - `device='cuda'` istendiğinde PyTorch CUDA SVD ve tensor işlemlerini çalıştırır; CUDA yoksa CPU'ya düşmeyip açık hata üretir.
  - `BackendInfo`: Sistemde aktif çalışan backend, donanım özellikleri (AVX2, AVX-512, OpenMP) ve thread sayısını raporlar.

#### 4. [`attention.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/attention.py)
- **Açıklama**: Kapalı-form dikkat (attention) mekanizmaları ve derin blok yığınlarıdır.
- **Detay**:
  - `ClosedFormAttention`: $Softmax(Q K^T / \sqrt{d_k}) V$ yapısını kullanır. Projeksiyon ağırlıkları backprop ile öğrenilmez; sabit rastgele veya SVD tabanından ($V_t^k$) türetilir. Sinüzoidal pozisyon kodlaması içerir.
  - `CooccurrenceRelationBasis`: Kelimelerin pencere içi co-occurrence istatistiklerini seyrek (sparse CSR) matriste biriktirir, PPMI dönüşümü uygular ve randomized SVD ile veriye duyarlı $Q/K$ taban vektörleri üretir.
  - `LearnedRelationAttention`: PPMI-SVD tabanından okunan veriye uygun $Q/K$ rollerini kullanarak kelime çifti ilişkilerini skorlar.
  - `TransformerBlockStack`: Birden fazla `[Attention + Residual]` -> `[Reservoir FFN + Residual]` bloğunu RMSNorm benzeri ölçeklemeyle derinleştirir.

#### 5. [`training.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/training.py)
- **Açıklama**: RIDM çekirdeği için programatik eğitim ve değerlendirme orkestratörüdür.
- **Detay**:
  - `TrainingConfig`: Boyut (`dim`), pencere (`window`), rank (`k`), batch boyutu ve backend konfigürasyonlarını tutar.
  - `Trainer`: Veriyi train/val/test bölümlerine ayırır, pencere sınırındaki bağlam kaybını önlemek için önceki $W$ tokenı sonraki batch'e devrederek `partial_fit` çalıştırır, `finalize` ile modeli sabitler ve `.npz` + `training.json` artifact'lerini kaydeder.

#### 6. [`memory.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/memory.py)
- **Açıklama**: İlişkisel bellek ve hafıza saklama/unutma modülleridir.
- **Detay**:
  - `SparseDistributedMemory` (SDM): Kanerva'nın seyrek dağıtık bellek fikrinin gerçek-değerli cosine versiyonudur. Ebbinghaus tipi `decay` ile eski verileri zayıflatır.
  - `ProductKeyMemory` (PKM): Lample ve arkadaşlarının (2019) Product-Key Memory mimarisinin kapalı-form versiyonudur. Sorguyu ikiye bölüp kartezyen ürünle $N = n_{sub}^2$ potansiyel hücre arasından $O(n_{sub})$ maliyetle aday üretir. Nadir/zayıf bellek hücrelerini budayan `consolidate` mekanizmasına sahiptir.

#### 7. [`reasoning.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/reasoning.py)
- **Açıklama**: Çok adımlı çıkarım ve adaptif denetim katmanıdır.
- **Detay**:
  - `ReasoningChain`: Tek bir geçiş yerine bağlam vektörünü attention, anlamsal grafik ve PKM bellek modülleri arasında sabit-noktalı olarak rafine eder.
  - `ReasoningController`: Bağlamın entropisine bakarak kaç "hop" çalışacağına dinamik karar verir. Kolay bağlamlarda az, belirsiz bağlamlarda derin çıkarım adımı planlar.

#### 8. [`hybrid.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/hybrid.py)
- **Açıklama**: Tüm alt sistemleri bir araya getiren hibrit ensemble modeldir.
- **Detay**:
  - `HybridLM`: RIDM olasılıkları ile N-gram olasılıklarını $\alpha$ ağırlığıyla çarpar; üstüne Graph Spreading, SDM, RAG, Reasoning, Reservoir, Relation Attention, Block Stack ve Calibrator sinyallerini çarpımsal bonus olarak ekler.
  - `tune_alpha`: Doğrulama kümesinde grid search ile en ideal N-gram / RIDM dengesini bulur.
  - `generate`: Top-k ve sıcaklık (temperature) parametreleriyle metin üretir.

#### 9. [`graph_retrieval.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/graph_retrieval.py)
- **Açıklama**: Graph tabanlı anlamsal bellek ve LSH destekli RAG getirim katmanıdır.
- **Detay**:
  - `SemanticGraph`: Kelime gömmeleri arasındaki kosinüs benzerliği üzerinden en yakın $M$ komşuyu indeksler; `spreading_activation` ile kavramsal yayılım hesabı yapar.
  - `LSHIndex`: SimHash (rastgele hiperdüzlem) algoritmasıyla çalışan çok-tablolu (multi-table) Approximate Nearest Neighbor (ANN) indeksidir.
  - `SimpleRAG`: Belgeleri vektörleştirip LSH veya kaba-kuvvet (brute-force) ile getirir.

#### 10. [`hierarchical.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/hierarchical.py)
- **Açıklama**: Çoklu pencere ölçekli bellek ve adaptif pencere seçimidir.
- **Detay**:
  - `HierarchicalContextMemory`: Farklı pencere boyutlarına (ör. 2, 5, 15) sahip RIDM katmanlarını entropi ağırlıklı birleştirir.
  - `adaptive_context`: Cümle sınır kelimelerine (`.`, `!`, `?` vb.) bakarak bağlam boyutunu dinamik kırpar.

#### 11. [`reservoir.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/reservoir.py)
- **Açıklama**: Extreme Learning Machine (ELM) esaslı kapalı-form doğrusal-olmayan derinlik katmanıdır.
- **Detay**:
  - `DeepReservoirStack`: Gizli katman ağırlıkları sabit ve rastgeledir; sadece son okuma (readout) katmanı kapalı-form Ridge regresyon ($H^T H + \lambda I)^{-1} H^T Y$ ile eğitilir.
  - `DeepReservoirScorer`: Bağlamdan hedef gömmeyi tahmin eden Ridge puanlayıcıdır.

#### 12. [`multisense.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/multisense.py)
- **Açıklama**: Çoklu anlam (polysemy) ayrıştırma modülüdür.
- **Detay**:
  - `MultiSenseEmbedding`: Bir kelimenin geçtiği tüm bağlam vektörlerini k-means ile kümeleyerek farklı anlam prototipleri (ör. "banka" finans vs. nehir kıyısı) oluşturur.

#### 13. [`moe.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/moe.py)
- **Açıklama**: Uzman sistemlerin güven/entropi bazlı yönlendirilmesidir.
- **Detay**:
  - `MixtureOfExperts`: Farklı `HybridLM` uzmanlarının çıktılarını ters-entropi (güven) ağırlıklarıyla birleştirir.

#### 14. [`calibration.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/calibration.py)
- **Açıklama**: Kalibrasyon ve sıralama (reranking) katmanıdır.
- **Detay**:
  - `AutoregressiveCalibrator`: Tüm alt sistem skorlarını hedef token olasılığına göre kapalı-form Ridge regresyon ile kalibre eder.
  - `NeuralReranker`: Projedeki tek SGD/backprop kullanan katmanlardan biridir. Aday tokenların skor, frekans ve kosinüs özelliklerini kullanarak 2 katmanlı YSA ile yeniden sıralar.

#### 15. [`benchmark.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/benchmark.py)
- **Açıklama**: Benchmark değerlendirme motorudur.
- **Detay**:
  - `BenchmarkRunner`: MMLU, ARC veya HellaSwag formatındaki yerel JSONL dosyalarını okuyup çoktan seçmeli doğruluk hesaplar. Çevrimdışı testler için sentetik analoji ve cloze testleri sunar.

#### 16. [`vocab.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/vocab.py)
- **İşlev**: Sözlük oluşturma (`build_vocab`), token-index eşleme (`encode`) ve veri kümesi bölme (`train_test_split`).

#### 17. [`tokenizer.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/tokenizer.py)
- **İşlev**: Saf Python ile yazılmış BPE (Byte-Pair Encoding) tokenizer eğitme ve kodlama yapısı.

#### 18. [`subword.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/subword.py)
- **İşlev**: Sözlük dışı (OOV) kelimeler için n-gram hash tabanlı alt-kelime vektör üretimi (`SubwordHasher`).

#### 19. [`corpus.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/corpus.py)
- **İşlev**: Türkçe ve İngilizce için varsayılan sentetik korpus üreteci.

#### 20. [`datasets.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/datasets.py)
- **İşlev**: Dosya ve metin kaynaklarından veriyi akışlı okuyan `TextDataset` sınıfı.

#### 21. [`ngram.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ngram.py)
- **İşlev**: Laplace/Kneser-Ney düzeltmeli istatistiksel N-gram dili modeli.

#### 22. [`utils.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/utils.py)
- **İşlev**: Entropi hesaplama vb. ortak matematiksel yardımcı fonksiyonlar.

#### 23. [`constants.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/constants.py)
- **İşlev**: Sistem genelinde kullanılan sabitler (`UNK_TOKEN`, cümle sonu işaretçileri).

#### 24. [`cli.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/cli.py)
- **İşlev**: `ridm-ultra` komut satırı arayüzü. `lite`, `balanced` ve `full` profilleri ile çakışan modülleri yönetir, modeli eğitir, değerlendirir ve benchmark'ları çalıştırır.

#### 25. [`pyproject.toml`](file:///c:/Users/HP/Desktop/ridm%20ultra/pyproject.toml)
- **İşlev**: Proje bağımlılıkları, sürümleri, script CLI tanımlamaları ve Python linter/formatter (Ruff ve Black) konfigürasyonları.

#### 26. [`sonar-project.properties`](file:///c:/Users/HP/Desktop/ridm%20ultra/sonar-project.properties)
- **İşlev**: SonarQube / SonarScanner statik kod analizi yapılandırması (Python, C++ ve proje kaynak kodları için kalite kuralları, dışlamalar ve indeksleme).

#### 27. [`.prettierrc`](file:///c:/Users/HP/Desktop/ridm%20ultra/.prettierrc) & [`.prettierignore`](file:///c:/Users/HP/Desktop/ridm%20ultra/.prettierignore)
- **İşlev**: Prettier biçimlendirme kuralları (Markdown, JSON, YAML vb. repo dosyalarının standart kod stili takibi).

#### 28. [`.eslintrc.json`](file:///c:/Users/HP/Desktop/ridm%20ultra/.eslintrc.json) & [`.eslintignore`](file:///c:/Users/HP/Desktop/ridm%20ultra/.eslintignore)
- **İşlev**: ESLint statik kod analiz ve stil kuralları konfigürasyonu.

---

### 3.2. Yerel Hızlandırma Katmanı (`native/`)

#### 1. [`native/ridm_kernels.cpp`](file:///c:/Users/HP/Desktop/ridm%20ultra/native/ridm_kernels.cpp)
- **Açıklama**: C++17 yüksek performanslı CPU çekirdeğidir.
- **Detay**:
  - AVX2 ve AVX-512 FMA SIMD komut setlerini çalışma zamanında (runtime CPU dispatch) algılar.
  - OpenMP ile bağlam matrisi biriktirmesini (`ridm_accumulate_contexts_f32`) paralelleştirir; `atomic update` ile çakışmaları engeller.
  - Matris-vektör çarpımını (`ridm_matvec_f32`) AVX2/AVX512 SIMD şeritleriyle hızlandırır.

#### 2. [`native/CMakeLists.txt`](file:///c:/Users/HP/Desktop/ridm%20ultra/native/CMakeLists.txt)
- **İşlev**: `ridm_kernels.dll` / `.so` kütüphanesini MinGW/MSVC/GCC ile derleyen CMake yapılandırması.

#### 3. [`native/benchmark.cpp`](file:///c:/Users/HP/Desktop/ridm%20ultra/native/benchmark.cpp)
- **İşlev**: C++ çekirdeğinin bant genişliği ve FLOPS başarımını ölçen bağımsız benchmark yürütücüsü.

---

### 3.3. RIDM Ultra LLM Alt Sistemi (`ridm_ultra/llm/`)

#### 1. [`ridm_ultra/llm/cli.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/cli.py)
- **Açıklama**: LLM ekosisteminin tekil CLI komut merkezidir.
- **Detay**: `pretrain`, `prepare-data`, `tokenizer`, `evaluate`, `generate`, `smoke-test`, `sft`, `dpo`, `safety-eval`, `rag-index`, `rag-generate`, `quantize`, `serve`, `pilot` komutlarını barındırır.

#### 2. Model Katmanı (`ridm_ultra/llm/model/`)
- [`transformer.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/model/transformer.py): PyTorch tabanlı modern Transformer mimarisi. FlashAttention (`F.scaled_dot_product_attention`), RoPE (Rotary Embeddings), GQA (Grouped Query Attention), SwiGLU aktivasyonu, RMSNorm ve çıkarım için KV-Cache yapısını barındırır.
- [`config.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/model/config.py): Hiperparametre ve konfigürasyon veri sınıfları.
- [`presets.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/model/presets.py): `smoke-17m`, `turkish-50m` gibi hazır ön-ayarlı model yapılandırmaları.

#### 3. Veri Katmanı (`ridm_ultra/llm/data/`)
- [`quality.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/data/quality.py): Unicode normalizasyonu, PII temizleme (e-posta/telefon redaksiyonu), tam SHA-256 deduplikasyonu ve SQLite tabanlı SimHash-LSH yakın-tekrar temizliği.
- [`streaming.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/data/streaming.py): Devasa JSONL dosyalarını RAM sınırını aşmadan paketleyerek okuyan akışlı veri yükleyici.
- [`tokenizer.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/data/tokenizer.py): HuggingFace Rust `tokenizers` kütüphanesi sarmalayıcısı.

#### 4. Çalışma Zamanı Katmanı (`ridm_ultra/llm/runtime/`)
- [`trainer.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/runtime/trainer.py): `torchrun` çoklu-GPU DDP (Distributed Data Parallel) desteği, cosine LR schedule, gradyan kırpma ve atomik checkpoint kaydı.
- [`evaluation.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/runtime/evaluation.py): Validation perplexity ve loss ölçüm motoru.
- [`benchmark.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/runtime/benchmark.py): LLM hız ve throughput başarım ölçümü.
- [`pilot.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/runtime/pilot.py): GPU donanım kabul ve stabilite testi.

#### 5. Hizalama Katmanı (`ridm_ultra/llm/align/`)
- [`sft.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/align/sft.py): Prompt maskelemeli Supervised Fine-Tuning (SFT).
- [`preference.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/align/preference.py): Dondurulmuş referans model üzerinden Direct Preference Optimization (DPO).
- [`data.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/align/data.py): SFT ve DPO için veri hazırlama yardımcıları.

#### 6. Güvenlik Katmanı (`ridm_ultra/llm/safety/`)
- [`pii.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/safety/pii.py): Çıktılarda PII sızıntısı tarayıcısı.
- [`probes.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/safety/probes.py): Red-teaming test bataryası ve sezgisel reddetme sınıflandırıcısı.
- [`acceptance.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/safety/acceptance.py): Perplexity ve güvenlik skorlarını birleştiren checkpoint kabul testi.

#### 7. Getirim & Servis Katmanları (`ridm_ultra/llm/retrieval/` & `serving/`)
- [`retrieval/index.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/retrieval/index.py) & [`generate.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/retrieval/generate.py): Bağımlılıksız Okapi BM25 indeksleme ve RAG üreticisi.
- [`serving/quantize.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/serving/quantize.py): CPU için PyTorch int8 dinamik kuantizasyon.
- [`serving/api.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/llm/serving/api.py): API-key doğrulamalı FastAPI çıkarım servisi (`/health`, `/generate`, `/rag/generate`).

---

### 3.4. RIDM Ultra Chat Ekosistemi (`ridm_ultra/chat/`)

- [`types.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/chat/types.py): Tip güvenli mesaj, oturum, kullanım ve model katman tanımları (`ChatMessage`, `ChatSession`, `ModelTier`, `TokenUsage`).
- [`interfaces.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/chat/interfaces.py): Temel bileşen arayüzleri (`BaseModelAdapter`, `BaseMemoryManager`, `BaseRouter`, `BaseChatRepository`).
- [`adapters.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/chat/adapters.py): Yerel PyTorch Transformer ve harici Cloud API adaptörleri ile bağımlılık enjeksiyon deposu (`AdapterFactory`).
- [`memory.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/chat/memory.py): Kayar pencere ve otomatik arka plan özetleyici (`HierarchicalMemoryManager`).
- [`router.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/chat/router.py): Niyet ve karmaşıklık sınıflandırmalı semantik yönlendirici (`SemanticRouter`).
- [`repository.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/chat/repository.py): `InMemoryChatRepository` ve sıfır bağımlılıklı `SQLiteChatRepository` oturum saklama katmanı.
- [`engine.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/chat/engine.py): Asenkron streaming (`AsyncGenerator`), RAG sorgu dönüşümü ve oturum yönetimi sunan `ChatEngine` ana facade orkestratörü.

---

### 3.5. Test Otomasyonu & CI Altyapısı (`tests/` & `.github/`)

- [`tests/conftest.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/tests/conftest.py): Pytest ortak fixture tanımları (`sample_ridm`, `sample_chat_engine`, `temp_db_path`).
- [`tests/test_core.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/tests/test_core.py): RIDM SVD çekirdeği, ComputeBackend, artımlı güncelleme ve serialization testleri.
- [`tests/test_attention.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/tests/test_attention.py): Kapalı-Form Dikkat, PPMI-SVD Cooccurrence tabanı ve TransformerBlockStack testleri.
- [`tests/test_memory.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/tests/test_memory.py): SDM ve PKM bellek unutma ve pekiştirme testleri.
- [`tests/test_chat.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/tests/test_chat.py): ChatEngine asenkron streaming, Semantik Yönlendirici, Kayar Pencere ve SQLite depolayıcı testleri.
- [`tests/test_api.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/tests/test_api.py): FastAPI TestClient ile `/health`, `/api/v1/sessions` ve `/api/v1/chat/stream` SSE akış entegrasyon testleri.
- [`.github/workflows/ci.yml`](file:///c:/Users/HP/Desktop/ridm%20ultra/.github/workflows/ci.yml): Çoklu Python sürümlerinde otomasyonlu ruff, black me pytest CI iş akışı.
- [`.pre-commit-config.yaml`](file:///c:/Users/HP/Desktop/ridm%20ultra/.pre-commit-config.yaml): Commit öncesi kalite kontrol hook'ları.

---

### 3.6. Veri İçe Aktarma & RAG Katmanı (`scripts/` & `data/raw/`)

- [`scripts/ingest_desktop_data.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/scripts/ingest_desktop_data.py): Masaüstündeki `.parquet` veri kümelerini (FineWeb-Edu vb.) PyArrow batch akışıyla belleği şişirmeden `data/raw/` dizinine aktaran, metrikleri hesaplayan ve `data/raw/rag_documents.jsonl` pasaj indeksini oluşturan betik.
- [`scripts/hardware_audit.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/scripts/hardware_audit.py): Sistem donanımını (CPU, RAM, GPU/CUDA) otomatik tespit eden ve 1.98 milyar token için tam matematiksel TFLOPS / VRAM eğitimi hesaplayan denetçi.
- [`TRAINING_ESTIMATE.md`](file:///c:/Users/HP/Desktop/ridm%20ultra/TRAINING_ESTIMATE.md): İşlemci vs. GPU eğitimi süre karşılaştırmalarını ve RIDM Ultra kapalı-form SVD üstünlüğünü detaylandıran rapor.
- [`scripts/evaluate_svd_vs_llm.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/scripts/evaluate_svd_vs_llm.py): Kapalı-Form SVD vektör uzayı ile Otoragresif LLM üretim yeteneklerini karşılaştırmalı test eden ve değerlendiren betik.
- [`scripts/train_ridm_full_dataset.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/scripts/train_ridm_full_dataset.py): FineWeb-Edu Parquet kümesinin tamamı (~1.22 milyar token) üzerinde OpenMP C++ SIMD kapalı-form SVD sıfır-geri-yayılımlı eğitimi gerçekleştiren ana üretim betiği.
- [`artifacts/ridm_fineweb_embeddings.npz`](file:///c:/Users/HP/Desktop/ridm%20ultra/artifacts/ridm_fineweb_embeddings.npz): 1.22 milyar token üzerinden eğitilmiş 32,000 kelimelik kapalı-form SVD gömme ağırlıkları.
- [`artifacts/ridm_fineweb_metadata.json`](file:///c:/Users/HP/Desktop/ridm%20ultra/artifacts/ridm_fineweb_metadata.json): Eğitilen modelin metrikleri, hız istatistikleri ve kelime dağarcığı metadata kaydı.
- [`corpus.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/corpus.py): `load_rag_documents` ile `data/raw/rag_documents.jsonl` dosyalarını okuyarak RAG ve dil modelleri için hazır belge dizisi üreten yardımcı.
- [`ridm_ultra/api/dependencies.py`](file:///c:/Users/HP/Desktop/ridm%20ultra/ridm_ultra/api/dependencies.py): `get_rag_engine()` içinde aktarılan FineWeb-Edu pasajlarını `SimpleRAG` katmanına otomatik yükleyen bağımlılık enjektörü.

---

## 4. Akış ve Veri Yolculuğu Diagramı (Pipeline)

```
[Ham Veri: TXT / JSONL]
          |
          v
[LLM Data Quality: Unicode, PII Masking, SHA256 & SimHash Deduplication]
          |
          +-----------------------------------+
          |                                   |
          v (RIDM Core Path)                  v (LLM PyTorch Path)
[RIDM Random Indexing]              [BPE Tokenizer Training]
          |                                   |
[OpenMP / C++ Kernel Accumulation]  [Streaming Data Loader & Packing]
          |                                   |
[Truncated SVD (CPU/CUDA)]          [PyTorch DDP Pre-training (FlashAttn, RoPE, GQA)]
          |                                   |
[HybridLM Ensemble (SDM, PKM, RAG)] [SFT / DPO Alignment & Safety Evaluation]
          |                                   |
[Reasoning Controller & Multi-Hop]  [Dynamic int8 Quantization & FastAPI Serving]
```

---

## 5. Özet

RIDM Ultra projesi, hem **hesaplama maliyeti düşük kapalı-form hibrit araştırmaları** hem de **endüstri standardı PyTorch/CUDA tabanlı gerçek Transformer modelleri** için uçtan uca modüler, ölçeklenebilir ve donanım duyarlı bir altyapı sunmaktadır.
