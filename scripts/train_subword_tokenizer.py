import json
import sys
from pathlib import Path

import pandas as pd

from tokenizer import SentencePieceBPETokenizer

sys.stdout.reconfigure(encoding='utf-8')

def main():
    dataset_path = Path("data/raw/turkish_instructions.parquet")
    if not dataset_path.exists():
        print(f"[!] Dataset not found at {dataset_path}")
        return

    print(f"[*] Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    sample_texts = df["text"].dropna().sample(min(10000, len(df)), random_state=42).tolist()

    print(f"[*] Training SentencePieceBPETokenizer on {len(sample_texts)} Turkish sentences...")
    tok = SentencePieceBPETokenizer(vocab_size=32000)
    tok.train_on_sentences(sample_texts, num_merges=1500, min_freq=2)

    output_path = Path("artifacts/sentencepiece_tr_vocab.json")
    print(f"[*] Saving subword tokenizer model to {output_path}...")
    output_path.write_text(json.dumps(tok.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    # Test encoding and decoding
    test_word = "kitaplarımızdan"
    subwords = tok.encode_as_subwords(test_word)
    decoded = tok.decode(subwords)
    print("\n[TEST SANITY CHECK]")
    print(f"Original word : '{test_word}'")
    print(f"BPE Subwords  : {subwords}")
    print(f"Decoded text  : '{decoded}'")
    print("[+] SentencePiece BPE Subword Training Completed Successfully!")

if __name__ == "__main__":
    main()
