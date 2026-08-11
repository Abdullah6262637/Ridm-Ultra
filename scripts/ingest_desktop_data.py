"""Ingest FineWeb-Edu Parquet datasets from Desktop into RIDM Ultra data/raw/."""
import json
import math
import shutil
from pathlib import Path

import pyarrow.parquet as pq


def get_desktop_path() -> Path:
    return Path.home() / "Desktop"


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 3.8))


def main():
    print("=== RIDM ULTRA DESKTOP DATASET INGESTION ===")
    desktop_dir = get_desktop_path()
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list(desktop_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"[!] No .parquet files found on Desktop ({desktop_dir})")
        return

    print(f"[*] Found {len(parquet_files)} .parquet dataset file(s) on Desktop.")

    ingested_paths = []
    for pfile in parquet_files:
        dest_file = raw_dir / pfile.name
        print(f"[*] Copying {pfile.name} ({pfile.stat().st_size / (1024**3):.2f} GB) to {dest_file}...")
        if not dest_file.exists():
            shutil.copy2(pfile, dest_file)
        ingested_paths.append(dest_file)

    print("\n[*] Processing Parquet datasets with PyArrow batch streaming...")
    total_rows = 0
    total_chars = 0
    total_tokens = 0
    samples = []

    rag_jsonl_path = raw_dir / "rag_documents.jsonl"
    max_rag_passages = 5000  # Index top 5000 high-quality passages for RAG retrieval

    with open(rag_jsonl_path, "w", encoding="utf-8") as out_f:
        for ppath in ingested_paths:
            parquet_file = pq.ParquetFile(ppath)
            schema_names = parquet_file.schema.names
            text_col = "text" if "text" in schema_names else schema_names[0]
            print(f"  - File: {ppath.name} | Row Groups: {parquet_file.num_row_groups} | Text Column: '{text_col}'")

            for batch in parquet_file.iter_batches(batch_size=1000, columns=[text_col]):
                texts = batch.column(text_col).to_pylist()
                for t in texts:
                    if not t or not isinstance(t, str):
                        continue
                    length = len(t)
                    tokens = estimate_tokens(t)
                    total_rows += 1
                    total_chars += length
                    total_tokens += tokens

                    if len(samples) < 10:
                        samples.append({
                            "id": total_rows,
                            "char_count": length,
                            "est_tokens": tokens,
                            "snippet": t[:150].replace("\n", " ") + "..."
                        })

                    if total_rows <= max_rag_passages:
                        # Write passage slice into RAG documents index
                        passage_dict = {
                            "id": total_rows,
                            "text": t[:1000],  # Chunk into 1000 chars for efficient RAG retrieval
                            "char_count": min(length, 1000),
                            "est_tokens": estimate_tokens(t[:1000])
                        }
                        out_f.write(json.dumps(passage_dict, ensure_ascii=False) + "\n")

    print("\n=== DATASET INGESTION METRICS ===")
    print(f"  - Total Ingested Files  : {len(ingested_paths)}")
    print(f"  - Total Dataset Rows    : {total_rows:,}")
    print(f"  - Total Character Count : {total_chars:,}")
    print(f"  - Total Estimated Tokens: {total_tokens:,}")
    print(f"  - RAG Passages Indexed  : {min(total_rows, max_rag_passages):,} (Saved to {rag_jsonl_path})")

    print("\n=== TOP 10 SAMPLE RECORDS ===")
    for s in samples:
        print(f" [{s['id']}] Tokens: {s['est_tokens']:<5} | Chars: {s['char_count']:<6} | Snippet: {s['snippet']}")

    print("\n[SUCCESS] Parquet Dataset Ingestion Complete!")


if __name__ == "__main__":
    main()
