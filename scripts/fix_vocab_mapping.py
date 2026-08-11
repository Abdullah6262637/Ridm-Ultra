"""Rebuild and save complete 32,000 vocabulary mapping for FineWeb SVD embeddings."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    print("=== REBUILDING FULL 32,000 VOCABULARY MAPPING ===")
    raw_dir = Path("data/raw")
    parquet_files = list(raw_dir.glob("*.parquet"))

    word_counts = {}
    total_docs = 0
    vocab_limit = 200_000

    for pf in parquet_files:
        pfile = pq.ParquetFile(pf)
        text_col = "text" if "text" in pfile.schema.names else pfile.schema.names[0]
        for batch in pfile.iter_batches(batch_size=2000, columns=[text_col]):
            for t in batch.column(text_col).to_pylist():
                if t and isinstance(t, str):
                    for w in t.split():
                        word_counts[w] = word_counts.get(w, 0) + 1
                    total_docs += 1
                    if total_docs >= vocab_limit:
                        break
            if total_docs >= vocab_limit:
                break
        if total_docs >= vocab_limit:
            break

    sorted_words = sorted(word_counts.items(), key=lambda item: -item[1])
    vocab = ["<UNK>"] + [w for w, c in sorted_words[:31999]]

    npz_path = Path("artifacts/ridm_fineweb_embeddings.npz")
    if npz_path.exists():
        npz = np.load(npz_path)
        np.savez_compressed(
            npz_path,
            word_emb=npz["word_emb"],
            context_vecs=npz["context_vecs"],
            k=npz["k"],
            vocab=np.array(vocab)
        )

    vocab_json_path = Path("artifacts/ridm_fineweb_vocab.json")
    vocab_json_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    meta_json_path = Path("artifacts/ridm_fineweb_metadata.json")
    if meta_json_path.exists():
        meta = json.loads(meta_json_path.read_text(encoding="utf-8"))
        meta["vocab_size"] = len(vocab)
        meta["vocab"] = vocab
        meta_json_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    print(f"[SUCCESS] Rebuilt and saved complete {len(vocab):,} vocabulary mapping to {vocab_json_path.resolve()}")


if __name__ == "__main__":
    main()
