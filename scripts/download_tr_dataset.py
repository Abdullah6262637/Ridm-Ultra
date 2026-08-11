"""Download Turkish Instruction Datasets (OpenOrca-tr & Turkish-Alpaca) and save to data/raw."""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


def main():
    print("=== DOWNLOADING TURKISH INSTRUCTION DATASETS ===")
    data_raw_dir = Path("data/raw")
    data_raw_dir.mkdir(parents=True, exist_ok=True)

    combined_documents = []

    # 1. Download ucekmez/OpenOrca-tr
    print("[1/2] Downloading ucekmez/OpenOrca-tr...")
    try:
        ds_orca = load_dataset("ucekmez/OpenOrca-tr", split="train")
        print(f"      Loaded OpenOrca-tr: {len(ds_orca):,} rows.")

        for row in ds_orca:
            system = row.get("system_prompt", "") or ""
            question = row.get("question", "") or ""
            response = row.get("response", "") or ""

            text = f"{system} {question} {response}".strip()
            if text:
                combined_documents.append(text)
    except Exception as e:
        print(f"[!] Warning loading OpenOrca-tr: {e}")

    # 2. Download TFLai/Turkish-Alpaca
    print("[2/2] Downloading TFLai/Turkish-Alpaca...")
    try:
        ds_alpaca = load_dataset("TFLai/Turkish-Alpaca", split="train")
        print(f"      Loaded Turkish-Alpaca: {len(ds_alpaca):,} rows.")

        for row in ds_alpaca:
            instruction = row.get("instruction", "") or ""
            input_text = row.get("input", "") or ""
            output_text = row.get("output", "") or ""

            text = f"{instruction} {input_text} {output_text}".strip()
            if text:
                combined_documents.append(text)
    except Exception as e:
        print(f"[!] Warning loading Turkish-Alpaca: {e}")

    print(f"[*] Total Turkish documents collected: {len(combined_documents):,}")

    # Save to Parquet using ParquetWriter in small batches
    out_parquet = data_raw_dir / "turkish_instructions.parquet"
    schema = pa.schema([("text", pa.string())])

    with pq.ParquetWriter(out_parquet, schema, compression="snappy") as writer:
        batch_size = 20000
        for i in range(0, len(combined_documents), batch_size):
            batch_texts = combined_documents[i:i + batch_size]
            table = pa.Table.from_arrays([pa.array(batch_texts)], schema=schema)
            writer.write_table(table)

    print(f"[SUCCESS] Saved Turkish instruction dataset ({len(combined_documents):,} rows) to {out_parquet.resolve()}")

    # Save 15,000 passages to data/raw/rag_documents.jsonl for SimpleRAG
    out_rag = data_raw_dir / "rag_documents.jsonl"
    with out_rag.open("w", encoding="utf-8") as f:
        for doc in combined_documents[:15000]:
            f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
    print(f"[SUCCESS] Updated {out_rag.resolve()} with Turkish RAG documents.")



if __name__ == "__main__":
    main()
