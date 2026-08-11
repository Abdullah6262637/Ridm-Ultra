"""Demo script showing RIDM Ultra's closed-form zero-backprop training speed on CPU."""
import time
from pathlib import Path

from corpus import load_rag_documents
from ridm_ultra import RIDM, build_vocab


def main():
    print("=== RIDM ULTRA CLOSED-FORM ZERO-BACKPROP TRAINING DEMO ===")
    rag_path = Path("data/raw/rag_documents.jsonl")
    if not rag_path.exists():
        print(f"[!] {rag_path} not found.")
        return

    print("[*] Loading passages from FineWeb-Edu ingested dataset...")
    t0 = time.time()
    passages = load_rag_documents(rag_jsonl_path=str(rag_path), max_docs=5000)
    raw_text = " ".join(passages)
    words = raw_text.split()
    print(f"[*] Loaded {len(passages):,} passages ({len(words):,} words) in {time.time() - t0:.2f} seconds.")

    print("\n[*] Building Vocabulary (Top 10,000 words)...")
    t0 = time.time()
    vocab, counts = build_vocab(words, max_vocab=10000)
    print(f"[*] Built Vocab of {len(vocab):,} words in {time.time() - t0:.2f} seconds.")

    print("\n[*] Initializing RIDM Ultra Closed-Form Engine (dim=128, window=5)...")
    ridm = RIDM(vocab=vocab, counts=counts, dim=128, window=5, seed=42)

    print("\n[*] Accumulating Co-occurrence Contexts (OpenMP C++ SIMD Kernel)...")
    t0 = time.time()
    # Map words to token IDs
    token_map = {w: i for i, w in enumerate(vocab)}
    token_ids = [token_map.get(w, 0) for w in words]

    # Run C++ kernel context accumulation
    ridm.partial_fit(token_ids)
    accum_time = time.time() - t0
    tok_per_sec = len(token_ids) / accum_time if accum_time > 0 else 0
    print(f"  -> Accumulated {ridm.total_contexts:,} context pairs ({len(token_ids):,} tokens) in {accum_time:.4f} seconds!")
    print(f"  -> C++ Kernel Throughput: {tok_per_sec:,.0f} tokens/second!")

    print("\n[*] Finalizing Closed-Form Truncated SVD...")
    t0 = time.time()
    ridm.finalize(k=64)
    svd_time = time.time() - t0
    print(f"  -> Truncated SVD computed in {svd_time:.4f} seconds!")
    print(f"  -> Final Word Embeddings Shape: {ridm.word_emb.shape}")

    print("\n[SUCCESS] RIDM Ultra Closed-Form Zero-Backprop Training Completed Successfully!")


if __name__ == "__main__":
    main()
