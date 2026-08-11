# CPU Smoke Testi (uçtan uca, sentetik veri)

Bu dizin, tüm LLM hattını (`tokenizer → prepare-data → pilot → pretrain →
evaluate → sft → dpo → safety-eval → rag-index/generate → quantize`) tek bir
CPU makinesinde, saniyeler içinde, gerçek eğitim gerektirmeden doğrulamak
için hazırlanmıştır. **Gerçek bir model üretmez** — amaç, kodun ve komutların
birbirine doğru bağlandığını (wiring) kanıtlamaktır. Gerçek ölçekli eğitim
için `docs/RUNBOOK_GPU.md`'ye bakın.

```bash
pip install -e ".[llm]"
bash examples/smoke/run_pipeline.sh /tmp/ridm-smoke-out
```

İçerik:
- `generate_corpus.py` — 400 kayıtlık, kalite filtresini (`min_chars=80`)
  geçecek uzunlukta sentetik Türkçe ön-eğitim korpusu üretir. Kasıtlı olarak
  çok tekrarlıdır; bu sayede `prepare-data`'nın near-duplicate (SimHash)
  filtresinin gerçekten çalıştığını da gösterir (~400 kayıttan ~136'sı kalır).
- `sft_data.jsonl` — `{"prompt", "response"}` formatında minik SFT örnekleri.
- `preference_data.jsonl` — `{"prompt", "chosen", "rejected"}` formatında
  minik DPO örnekleri.
- `rag_docs.jsonl` — RAG indeksi için minik belge kümesi.
- `run_pipeline.sh` — yukarıdaki tüm adımları sırayla çalıştırır.

Beklenen davranış: `pilot` adımı GPU olmayan makinelerde kasıtlı olarak
`"ready": false` döner (bu doğru davranıştır, hata değildir). `safety-eval`
adımı eşikleri kasıtlı gevşek tutar çünkü birkaç adımda eğitilen minik model
tutarlı Türkçe üretemez; gerçek reddetme davranışı gerçek ölçekli, gerçekten
hizalanmış bir model gerektirir.
