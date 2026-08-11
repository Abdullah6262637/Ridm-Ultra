"""Full-scale closed-form zero-backprop training script for RIDM Ultra on FineWeb-Edu Parquet datasets."""
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ridm_ultra import RIDM


def get_parquet_files(raw_dir: Path):
    files = list(raw_dir.glob("*.parquet"))
    files.sort(key=lambda p: 0 if "turkish" in p.name.lower() else 1)
    return files



def main():
    print("======================================================================")
    print("      RIDM ULTRA FULL CLOSED-FORM ZERO-BACKPROP TRAINING ENGINE      ")
    print("======================================================================")
    raw_dir = Path("data/raw")
    parquet_files = get_parquet_files(raw_dir)

    if not parquet_files:
        print(f"[!] No .parquet files found in {raw_dir.resolve()}")
        return

    print(f"[*] Found {len(parquet_files)} Parquet dataset files in {raw_dir}:")
    for pf in parquet_files:
        print(f"  - {pf.name} ({pf.stat().st_size / (1024**3):.2f} GB)")

    # Step 1: Vocabulary Building Pass
    print("\n[*] Step 1/3: Building Vocabulary from Parquet Corpus...")
    t_start = time.time()
    word_counts = {}
    total_docs = 0
    total_tokens_scanned = 0

    vocab_sample_limit = 200_000  # Scan top 200k documents for fast vocabulary discovery

    for pf in parquet_files:
        parquet_file = pq.ParquetFile(pf)
        schema_names = parquet_file.schema.names
        text_col = "text" if "text" in schema_names else schema_names[0]

        for batch in parquet_file.iter_batches(batch_size=2000, columns=[text_col]):
            texts = batch.column(text_col).to_pylist()
            for t in texts:
                if not t or not isinstance(t, str):
                    continue
                words = t.split()
                total_tokens_scanned += len(words)
                total_docs += 1

                for w in words:
                    word_counts[w] = word_counts.get(w, 0) + 1

                if total_docs >= vocab_sample_limit:
                    break
            if total_docs >= vocab_sample_limit:
                break
        if total_docs >= vocab_sample_limit:
            break

    print(f"  -> Scanned {total_docs:,} documents ({total_tokens_scanned:,} tokens) in {time.time() - t_start:.2f}s.")

    # Sort and pick top 32,000 vocabulary words
    sorted_words = sorted(word_counts.items(), key=lambda item: -item[1])
    max_vocab = 32000
    vocab = ["<UNK>"] + [w for w, c in sorted_words[:max_vocab - 1]]
    counts = {w: c for w, c in sorted_words[:max_vocab - 1]}
    counts["<UNK>"] = 1

    print(f"  -> Vocabulary constructed: {len(vocab):,} unique words.")
    token_map = {w: i for i, w in enumerate(vocab)}

    # Step 2: OpenMP C++ SIMD Context Accumulation
    print("\n[*] Step 2/3: Accumulating Co-occurrence Contexts (C++ SIMD OpenMP Engine)...")
    ridm = RIDM(vocab=vocab, counts=counts, dim=128, window=5, seed=42)

    total_trained_tokens = 0
    t_accum_start = time.time()
    batch_counter = 0

    for pf in parquet_files:
        parquet_file = pq.ParquetFile(pf)
        schema_names = parquet_file.schema.names
        text_col = "text" if "text" in schema_names else schema_names[0]

        for batch in parquet_file.iter_batches(batch_size=5000, columns=[text_col]):
            texts = batch.column(text_col).to_pylist()
            batch_tokens = []
            for t in texts:
                if t and isinstance(t, str):
                    batch_tokens.extend([token_map.get(w, 0) for w in t.split()])

            if len(batch_tokens) > 5:
                ridm.partial_fit(batch_tokens)
                total_trained_tokens += len(batch_tokens)
                batch_counter += 1

                if batch_counter % 20 == 0:
                    elapsed = time.time() - t_accum_start
                    tps = total_trained_tokens / elapsed if elapsed > 0 else 0
                    print(f"  -> Progress: {total_trained_tokens:,} tokens processed | Speed: {tps:,.0f} tokens/sec")

    accum_time = time.time() - t_accum_start
    final_tps = total_trained_tokens / accum_time if accum_time > 0 else 0
    print("\n[+] Co-occurrence Accumulation Complete!")
    print(f"  - Total Tokens Trained : {total_trained_tokens:,}")
    print(f"  - Total Context Pairs  : {ridm.total_contexts:,}")
    print(f"  - Total Accumulation Time: {accum_time:.2f} seconds ({accum_time/60.0:.2f} minutes)")
    print(f"  - Average C++ Speed    : {final_tps:,.0f} tokens/second")

    # Step 3: Closed-Form Truncated SVD Finalization
    print("\n[*] Step 3/3: Finalizing Closed-Form Truncated SVD...")
    t_svd_start = time.time()
    ridm.finalize(k=64)
    svd_time = time.time() - t_svd_start
    print(f"  -> Truncated SVD Completed in {svd_time:.4f} seconds!")
    print(f"  -> Model Embeddings Matrix Shape: {ridm.word_emb.shape}")

    # Save Model Artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    model_weights_path = artifacts_dir / "ridm_fineweb_embeddings.npz"
    vocab_meta_path = artifacts_dir / "ridm_fineweb_metadata.json"

    np.savez_compressed(
        model_weights_path,
        word_emb=ridm.word_emb,
        context_vecs=ridm.context_vecs,
        k=ridm.k,
        vocab=np.array(vocab)
    )

    vocab_json_path = artifacts_dir / "ridm_fineweb_vocab.json"
    vocab_json_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    meta = {
        "vocab_size": len(vocab),
        "embedding_dim": 64,
        "total_trained_tokens": total_trained_tokens,
        "total_contexts": ridm.total_contexts,
        "accumulation_time_sec": accum_time,
        "svd_time_sec": svd_time,
        "tokens_per_second": final_tps,
        "timestamp": time.time(),
        "vocab": vocab
    }
    vocab_meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


    total_pipeline_time = time.time() - t_start
    print("\n======================================================================")
    print("     RIDM ULTRA CLOSED-FORM TRAINING SUCCESSFULLY COMPLETED!          ")
    print("======================================================================")
    print(f"  - Total Ingested Dataset Tokens : {total_trained_tokens:,}")
    print(f"  - Total Pipeline Duration       : {total_pipeline_time:.2f} sec ({total_pipeline_time/60.0:.2f} min)")
    print(f"  - Model Embeddings Saved to     : {model_weights_path.resolve()}")
    print(f"  - Model Metadata Saved to       : {vocab_meta_path.resolve()}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
