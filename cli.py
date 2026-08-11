"""RIDM-ULTRA v5 komut-satiri arayuzu.

Bu modul, paketin tum alt-sistemlerini bir araya getirip egitir, degerlendirir
ve isteğe bagli olarak bir benchmark kosar. Onceki tek-dosya surumlerdeki
main() ile ISLEVSEL OLARAK AYNIDIR; ek olarak:

  - --profile {lite,balanced,full}: modul-cakismasi/redundansi (Sinirlama #2)
    icin hazir on-ayar setleri. 'balanced', SDM+PKM gibi BIRBIRINE BENZER
    amacli iki iliskisel bellegi AYNI ANDA calistirmanin gereksizligini
    onlemek icin varsayilan olarak sadece PKM'i (daha genis kapasiteli
    olani) birakir.
  - --benchmark-file / --run-synthetic-benchmarks: Sinirlama #5 (gercek
    benchmark eksikligi) icin BenchmarkRunner kosum destegi.
"""

import argparse
import sys
import time

import numpy as np

from .attention import ClosedFormAttention, CooccurrenceRelationBasis, LearnedRelationAttention, TransformerBlockStack
from .benchmark import BenchmarkRunner
from .calibration import AutoregressiveCalibrator, NeuralReranker
from .core import RIDM
from .corpus import build_default_corpus
from .graph_retrieval import SemanticGraph, SimpleRAG
from .hierarchical import HierarchicalContextMemory, adaptive_context
from .hybrid import HybridLM
from .memory import ProductKeyMemory, SparseDistributedMemory
from .multisense import MultiSenseEmbedding
from .ngram import NgramBaseline
from .reasoning import ReasoningChain, ReasoningController
from .reservoir import DeepReservoirScorer
from .tokenizer import BPETokenizer
from .vocab import build_vocab, encode, train_test_split

# Sinirlama #2 icin: hangi profil hangi modulleri KAPATIR (redundansi azaltmak
# ve hizli/orta/tam calisma modlari sunmak icin). Kullanici tek tek --no-X
# bayraklariyla bunun UZERINE de yazabilir (ekstra kapatma islevi gorur;
# bir profilin kapattigini tekil bayrakla tekrar 'acmanin' CLI'da karsiligi
# yoktur - bkz. asagidaki NOT).
PROFILE_DISABLE = {
    "lite": [
        "no_sdm", "no_pkm", "no_graph", "no_rag", "no_hierarchical", "no_reranker",
        "no_relation_attn", "no_multisense", "no_block_stack", "no_calibrator", "no_controller",
    ],
    "balanced": [
        "no_sdm",              # PKM zaten daha genis kapasiteli/esdeger amacli -> ikisini birden calistirma
        "no_rag", "no_hierarchical", "no_block_stack", "no_multisense",
    ],
    "full": [],
}


def _print_overlap_map():
    print(
        "\n[Modul Cakisma Haritasi] (Sinirlama #2 notu)\n"
        "  - SDM ve PKM AYNI ISI yapar (baglam->hedef iliskisel hatirlama);\n"
        "    farklari sadece kapasite/hiz dengesidir. 'balanced'/'lite' profilleri\n"
        "    varsayilan olarak sadece PKM'i birakir.\n"
        "  - ClosedFormAttention (rastgele) ile LearnedRelationAttention (PPMI-SVD)\n"
        "    FARKLI bilgi kaynaklarindan beslenir (pozisyon+icerik vs. es-olusum\n"
        "    istatistigi) - bu ikisi REDUNDANT DEGILDIR, tamamlayicidir.\n"
        "  - ReasoningChain (statik hop) ile ReasoningController (dinamik hop)\n"
        "    AYNI zinciri farkli sekilde calistirir; Controller varsa HybridLM\n"
        "    Controller'i kullanir, statik ReasoningChain bonusu devre disi kalir\n"
        "    (kodda otomatik, cift-sayim YAPILMAZ).\n"
    )


def build_arg_parser():
    ap = argparse.ArgumentParser(description="RIDM-ULTRA (v5, modulerize): kapali-form hibrit dil modeli sistemi")
    ap.add_argument("--corpus-file", type=str, default=None)
    ap.add_argument("--corpus-lang", choices=["en", "tr"], default="tr")
    ap.add_argument("--repeat", type=int, default=800)
    ap.add_argument("--dim", type=int, default=150)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", choices=["auto", "native", "numpy", "torch"], default="auto",
                    help="Matris çekirdeği: auto yerel C++ kernelini seçer.")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="cuda seçimi PyTorch CUDA gerektirir.")
    ap.add_argument("--threads", type=int, default=None, help="CPU/OpenMP iş parçacığı sayısı.")
    ap.add_argument("--save-model", type=str, default=None, help="Eğitilmiş RIDM modelinin .npz yolu.")
    ap.add_argument("--eval-split", type=float, default=0.15)
    ap.add_argument("--max-eval-samples", type=int, default=400)
    ap.add_argument("--ngram-n", type=int, default=3)
    ap.add_argument("--profile", choices=["lite", "balanced", "full"], default="full",
                     help="Onceden tanimli modul-acik/kapali seti (Sinirlama #2 notuna bakiniz).")
    ap.add_argument("--no-sdm", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--no-hierarchical", action="store_true")
    ap.add_argument("--no-reranker", action="store_true")
    ap.add_argument("--no-attention", action="store_true")
    ap.add_argument("--no-pkm", action="store_true")
    ap.add_argument("--no-reservoir", action="store_true")
    ap.add_argument("--no-reasoning", action="store_true")
    ap.add_argument("--no-relation-attn", action="store_true")
    ap.add_argument("--no-multisense", action="store_true")
    ap.add_argument("--no-block-stack", action="store_true")
    ap.add_argument("--no-calibrator", action="store_true")
    ap.add_argument("--no-controller", action="store_true")
    ap.add_argument("--use-bpe", action="store_true")
    ap.add_argument("--use-lsh-rag", action="store_true")
    ap.add_argument("--bpe-merges", type=int, default=100)
    ap.add_argument("--attn-heads", type=int, default=5)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--pkm-nsub", type=int, default=48)
    ap.add_argument("--pkm-topk", type=int, default=4)
    ap.add_argument("--reservoir-hidden", type=int, default=96)
    ap.add_argument("--senses", type=int, default=2)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--lsh-tables", type=int, default=4)
    ap.add_argument("--lsh-bits", type=int, default=8)
    ap.add_argument("--generate", type=str, default=None)
    ap.add_argument("--gen-length", type=int, default=40,
                     help="Uretilecek en fazla token sayisi (guvenlik ust siniri).")
    ap.add_argument("--gen-temp", type=float, default=0.9)
    ap.add_argument("--gen-topk", type=int, default=6)
    ap.add_argument("--gen-topp", type=float, default=None,
                     help="Nucleus sampling esigi (orn. 0.9); verilmezse kapali.")
    ap.add_argument("--gen-sentences", type=int, default=1,
                     help="Uretimin kac tam cumlede (./!/?) durdurulacagi.")
    ap.add_argument("--gen-repetition-penalty", type=float, default=1.3,
                     help="1.0 = kapali; >1 tekrarlayan kelimeleri cezalandirir.")
    ap.add_argument("--benchmark-file", type=str, default=None,
                     help="MMLU/ARC/HellaSwag-formatinda yerel bir JSONL dosyasi (bkz. benchmark.py).")
    ap.add_argument("--run-synthetic-benchmarks", action="store_true",
                     help="Gercek veri olmadan calisan kucuk, YERLESIK saglik-kontrolu gorevlerini calistirir.")
    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    for flag in PROFILE_DISABLE.get(args.profile, []):
        setattr(args, flag, True)
    print(f"[Profil] '{args.profile}' secildi.")
    _print_overlap_map()

    if args.corpus_file:
        with open(args.corpus_file, "r", encoding="utf-8") as f:
            words = f.read().split()
        if not words:
            print("HATA: korpus dosyasi bos.", file=sys.stderr)
            sys.exit(1)
    else:
        words = build_default_corpus(args.corpus_lang, args.repeat)

    bpe = None
    if args.use_bpe:
        t0 = time.time()
        bpe = BPETokenizer()
        bpe.train(words, num_merges=args.bpe_merges)
        example = " ".join(bpe.encode_word(words[0])) if words else ""
        words = bpe.encode(words)
        print(f"[BPE Tokenizer] {len(bpe.merges)} birlesme ogrenildi ({time.time()-t0:.3f} sn) | "
              f"ornek: '{example}' | alt-kelime sozlugu: {len(bpe.vocab)}")

    vocab, counts = build_vocab(words)
    word2idx = {w: i for i, w in enumerate(vocab)}
    token_ids = encode(words, word2idx)
    print(f"Korpus: {len(token_ids)} token, sozluk: {len(vocab)} kelime")

    train_ids, test_ids = train_test_split(token_ids, test_ratio=args.eval_split)
    train_ids, val_ids = train_test_split(train_ids, test_ratio=0.1)
    print(f"Egitim: {len(train_ids)} | Dogrulama: {len(val_ids)} | Test: {len(test_ids)}")

    t0 = time.time()
    ridm = RIDM(vocab, counts=counts, dim=args.dim, window=args.window, seed=args.seed,
                backend=args.backend, device=args.device, threads=args.threads)
    ridm.partial_fit(train_ids)
    ridm.finalize(k=args.k)
    print(f"\n[RIDM cekirdek] egitim suresi: {time.time()-t0:.3f} sn")
    print(f"[Hesaplama] {ridm.compute.info.to_dict()}")

    t0 = time.time()
    ngram = NgramBaseline(vocab_size=ridm.V, n=args.ngram_n)
    ngram.fit(train_ids)
    print(f"[N-gram] egitim suresi: {time.time()-t0:.3f} sn")

    sdm = None
    if not args.no_sdm:
        t0 = time.time()
        sdm = SparseDistributedMemory(addr_dim=args.dim, content_dim=ridm.k, n_locations=256, activation_radius=0.35, seed=args.seed)
        W = ridm.window
        for i in range(W, len(train_ids)):
            ctx_words = [ridm.idx2word[t] for t in train_ids[i - W:i]]
            addr = ridm.context_vector_for(ctx_words)
            target_id = train_ids[i]
            importance = float(ridm.idf[target_id])
            content = ridm.word_emb[target_id]
            sdm.write(addr, content, weight=importance)
        print(f"[SDM] yazim suresi: {time.time()-t0:.3f} sn | doluluk: {sdm.coverage():.1%}")

    t0 = time.time()
    drift_before = ridm.drift_estimate(train_ids[:200])
    ridm.incremental_update(val_ids[: min(50, len(val_ids))])
    drift_after = ridm.drift_estimate(train_ids[:200])
    print(f"[Incremental SVD] guncelleme suresi: {time.time()-t0:.3f} sn | "
          f"drift once={drift_before:.4f} sonra={drift_after:.4f} (referans icin; buyume normal)")

    graph = None
    if not args.no_graph:
        graph = SemanticGraph(ridm, top_m=5)
        print(f"[Semantic Graph] {ridm.V} dugum, dugum basi en yakin 5 komsu indekslendi")

    rag = None
    if not args.no_rag:
        docs = [" ".join(words[i:i + 8]) for i in range(0, min(len(words), 4000), 400)]
        rag = SimpleRAG(ridm, docs, use_lsh=args.use_lsh_rag, lsh_tables=args.lsh_tables, lsh_bits=args.lsh_bits, seed=args.seed)
        backend = "LSH (ANN)" if args.use_lsh_rag else "brute-force"
        print(f"[RAG] {len(docs)} belge parcasi indekslendi (getirim: {backend})")

    attention = None
    if not args.no_attention:
        attention = ClosedFormAttention(dim=args.dim, n_heads=args.attn_heads, seed=args.seed, use_ridm_basis=ridm._Vt_k)
        print(f"[Closed-Form Attention] {args.attn_heads} baslikli, pozisyon-kodlamali dikkat katmani hazir")

    relation_attn = None
    if not args.no_relation_attn:
        t0 = time.time()
        relation_basis = CooccurrenceRelationBasis(ridm, train_ids, window=args.window, seed=args.seed)
        relation_attn = LearnedRelationAttention(ridm, relation_basis)
        print(f"[Learned Relation Attention] PPMI-SVD iliski tabani cikarildi ({time.time()-t0:.3f} sn, k={relation_basis.dim})")

    multisense = None
    if not args.no_multisense:
        t0 = time.time()
        multisense = MultiSenseEmbedding(ridm, train_ids, n_senses=args.senses, seed=args.seed)
        n_poly = sum(1 for wid in multisense.prototypes if multisense.sense_count(wid) > 1)
        print(f"[Multi-Sense Embedding] {time.time()-t0:.3f} sn | birden fazla anlam-prototipi bulunan kelime sayisi: {n_poly}")

    pkm = None
    if not args.no_pkm:
        t0 = time.time()
        pkm = ProductKeyMemory(dim=args.dim, value_dim=args.dim, n_sub=args.pkm_nsub, top_k=args.pkm_topk, seed=args.seed)
        W = ridm.window
        for i in range(W, len(train_ids)):
            ctx_ids = train_ids[i - W:i]
            q = ridm.context_vecs[[t for t in ctx_ids if 0 <= t < ridm.V]].sum(axis=0)
            target_id = train_ids[i]
            importance = float(ridm.idf[target_id])
            target_vec = ridm.context_vecs[target_id]
            pkm.write(q, target_vec, weight=importance)
        util_before = pkm.utilization()
        pkm.decay(rate=0.97)
        pruned = pkm.consolidate(min_count=0.3)
        print(f"[Product-Key Memory] yazim suresi: {time.time()-t0:.3f} sn | "
              f"kapasite: {pkm.capacity()} hucre | kullanim(once): {util_before:.2%} -> "
              f"unutma+pekistirme sonrasi: {pkm.utilization():.2%} ({pruned} zayif hucre budandi)")

    reservoir = None
    if not args.no_reservoir:
        t0 = time.time()
        reservoir = DeepReservoirScorer(ridm, hidden_dims=(args.reservoir_hidden, args.reservoir_hidden), seed=args.seed)
        reservoir.fit(train_ids, max_samples=1200, seed=args.seed)
        print(f"[Deep Reservoir (ELM)] kapali-form ridge egitim suresi: {time.time()-t0:.3f} sn")

    reasoning = None
    if not args.no_reasoning and attention is not None:
        reasoning = ReasoningChain(ridm, attention, graph=graph, pkm=pkm, sdm=None, hops=args.hops, early_stop_eps=0.02)
        print(f"[Reasoning Chain] {args.hops} adimli rafine etme dongusu hazir (erken-durma esigi=0.02)")

    controller = None
    if not args.no_controller and reasoning is not None:
        controller = ReasoningController(ridm, reasoning, max_hops=args.max_hops, min_hops=1, entropy_high=2.0)
        print(f"[Reasoning Controller] entropi-tabanli dinamik hop planlamasi hazir (1..{args.max_hops} hop)")

    block_stack = None
    if not args.no_block_stack and reservoir is not None:
        t0 = time.time()
        block_attentions = [
            ClosedFormAttention(dim=args.dim, n_heads=args.attn_heads, seed=args.seed + 100 + i, use_ridm_basis=ridm._Vt_k)
            for i in range(args.blocks)
        ]
        block_stack = TransformerBlockStack(args.dim, block_attentions, reservoir.stack, n_blocks=args.blocks, seed=args.seed)
        print(f"[Transformer Block Stack] {args.blocks} residual blok hazir ({time.time()-t0:.3f} sn)")

    calibrator = None
    if not args.no_calibrator:
        t0 = time.time()
        calibrator = AutoregressiveCalibrator(ridm, ngram, relation_attn=relation_attn, reservoir=reservoir, lam=5.0)
        calibrator.fit(train_ids, max_samples=500, seed=args.seed)
        print(f"[Autoregressive Calibrator] kapali-form ridge kalibrasyon suresi: {time.time()-t0:.3f} sn")

    best_alpha, _ = HybridLM.tune_alpha(ridm, ngram, val_ids, max_samples=min(300, len(val_ids)))
    hybrid = HybridLM(ridm, ngram, alpha=best_alpha, graph=graph, sdm=sdm, rag=rag,
                       reasoning=reasoning, reservoir=reservoir,
                       relation_attn=relation_attn, calibrator=calibrator,
                       block_stack=block_stack, controller=controller)

    hier = None
    if not args.no_hierarchical:
        t0 = time.time()
        hier = HierarchicalContextMemory(vocab, counts, windows=(2, 5, 12), dim=args.dim, seed=args.seed)
        hier.partial_fit(train_ids)
        hier.finalize(k=args.k)
        print(f"[Hierarchical Memory] 3 katman egitim suresi: {time.time()-t0:.3f} sn")

    reranker = None
    if not args.no_reranker:
        t0 = time.time()
        reranker = NeuralReranker(n_features=4, hidden=16, seed=args.seed)
        reranker.train(ridm, ngram, train_ids, top_n=8, epochs=4, max_samples=800, seed=args.seed)
        print(f"[Neural Reranker] GERCEK SGD egitim suresi: {time.time()-t0:.3f} sn")

    print("\n=== SONUC TABLOSU (test kumesi) ===")
    print(f"{'Model':44s} {'dogruluk':>10s} {'perplexity':>12s}")
    res_ridm = ridm.evaluate(test_ids, max_samples=args.max_eval_samples)
    res_ngram = ngram.evaluate(test_ids, max_samples=args.max_eval_samples)
    res_hybrid = hybrid.evaluate(test_ids, max_samples=args.max_eval_samples)
    print(f"{'RIDM (cekirdek)':44s} {res_ridm['accuracy']:10.1%} {res_ridm['perplexity']:12.2f}")
    print(f"{args.ngram_n}-gram{'':38s} {res_ngram['accuracy']:10.1%} {res_ngram['perplexity']:12.2f}")
    print(f"{'Hibrit v5 (' + args.profile + ' profil)':44s} {res_hybrid['accuracy']:10.1%} {res_hybrid['perplexity']:12.2f}")
    if hier is not None:
        res_hier = hier.evaluate(test_ids, max_samples=args.max_eval_samples)
        print(f"{'Hiyerarsik (2/5/12 pencere)':44s} {res_hier['accuracy']:10.1%} {res_hier['perplexity']:12.2f}")

    mid = len(token_ids) // 2
    adw = adaptive_context(token_ids, mid, ridm.idx2word, min_w=2, max_w=args.window)
    print(f"\n[Adaptive Window] pozisyon {mid} icin secilen baglam ({len(adw)} token): "
          f"{[ridm.idx2word[t] for t in adw]}")

    sample_ctx = None
    for cand in (["kedi", "masaya"], ["the", "cat"]):
        if all(w in ridm.word2idx for w in cand):
            sample_ctx = cand
            break
    if sample_ctx is None and ridm.V > 1:
        sample_ctx = [ridm.idx2word[1], ridm.idx2word[min(2, ridm.V - 1)]]

    if sample_ctx:
        base = ridm.predict_topk(sample_ctx, k=5)
        print(f"\n'{' '.join(sample_ctx)}' -> RIDM ham siralama: {base}")
        ctx_ids = [ridm.word2idx[w] for w in sample_ctx]
        cvecs = ridm.context_vecs[ctx_ids]
        if reranker is not None:
            reranked = reranker.rerank(ridm, ngram, ctx_ids, top_n=8)
            print(f"'{' '.join(sample_ctx)}' -> Reranker sonrasi: {reranked[:5]}")
        if attention is not None:
            _, weights = attention.forward(cvecs)
            print(f"[Attention] konum agirliklari: {np.round(weights, 3)}")
        if relation_attn is not None:
            rel_scores = relation_attn.forward(ctx_ids)
            top = np.argsort(rel_scores)[::-1][:5]
            print(f"[Learned Relation Attention] en yuksek skorlu adaylar: "
                  f"{[(ridm.idx2word[i], round(float(rel_scores[i]), 3)) for i in top]}")
        if block_stack is not None:
            out_vec = block_stack.forward(cvecs)
            b_logits = ridm.word_emb @ (ridm._Vt_k @ out_vec)
            top = np.argsort(b_logits)[::-1][:5]
            print(f"[Transformer Block Stack] {block_stack.n_blocks} bloklu yigin sonrasi en olasi adaylar: "
                  f"{[(ridm.idx2word[i], round(float(b_logits[i]), 3)) for i in top]}")
        if calibrator is not None:
            cal_logits = calibrator.calibrated_logits(ctx_ids)
            if cal_logits is not None:
                top = np.argsort(cal_logits)[::-1][:5]
                print(f"[Autoregressive Calibrator] kalibre edilmis en olasi adaylar: "
                      f"{[(ridm.idx2word[i], round(float(cal_logits[i]), 3)) for i in top]}")
        if reasoning is not None:
            _, trace = reasoning.refine(ctx_ids)
            print(f"[Reasoning Chain] hop-basi degisim normu: {[round(t, 4) for t in trace]}")
        if controller is not None:
            _, info = controller.plan_and_run(ctx_ids)
            print(f"[Reasoning Controller] entropi={info['entropy']:.3f} -> planlanan hop={info['planned_hops']} "
                  f"| gerceklesen hop={info['effective_hops']}")

    # ---------- Sinirlama #5: benchmark kosumu ----------
    if args.benchmark_file or args.run_synthetic_benchmarks:
        runner = BenchmarkRunner(hybrid, ridm)
        print("\n=== BENCHMARK SONUCLARI ===")
        if args.benchmark_file:
            try:
                res = runner.run_jsonl_file(args.benchmark_file)
                print(f"[Ozel benchmark: {args.benchmark_file}] dogruluk: {res['accuracy']:.1%} ({res['n_items']} soru)")
            except (OSError, ValueError) as e:
                print(f"[Ozel benchmark] HATA: {e}")
        if args.run_synthetic_benchmarks:
            cloze = runner.synthetic_cloze_task(test_ids, n_items=100, seed=args.seed)
            print(f"[Sentetik: cloze-tamamlama] dogruluk: {cloze['accuracy']:.1%} ({cloze['n_items']} ogre) "
                  f"[GERCEK BENCHMARK DEGIL - saglik kontrolu]")
            if args.corpus_lang == "tr":
                pairs = [("kedi", "kediler", "kopek", "kopekler"), ("kacti", "kactilar", "cikti", "ciktilar")]
            else:
                pairs = [("cat", "cats", "dog", "dogs")]
            analogy = runner.synthetic_analogy_task(pairs)
            print(f"[Sentetik: analoji] dogruluk: {analogy['accuracy']:.1%} ({analogy['n_items']} ogre) "
                  f"[GERCEK BENCHMARK DEGIL - saglik kontrolu]")

    if args.generate:
        from .utils import detokenize
        seed_words = args.generate.split()
        gen_words = hybrid.generate(
            seed_words, length=args.gen_length, temperature=args.gen_temp,
            top_k=args.gen_topk, top_p=args.gen_topp, seed=args.seed,
            num_sentences=args.gen_sentences, repetition_penalty=args.gen_repetition_penalty,
        )
        print(f"\n[URETIM/ham token] '{args.generate}' -> {' '.join(gen_words)}")
        print(f"[URETIM/cumle]     '{args.generate}' -> {detokenize(gen_words)}")

    if args.save_model:
        ridm.save(args.save_model)
        print(f"[Model kaydı] {args.save_model}")


if __name__ == "__main__":
    main()
