import os
import sys
import pickle
import time
import argparse
from tqdm import tqdm

# RIDM Ultra path resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from ridm_ultra.core import RIDM
from ridm_ultra.tokenizer import SentencePieceBPETokenizer

def _get_text(sample):
    """Extract text from Aya collection sample."""
    text = ""
    if "text" in sample:
        text = sample["text"]
    else:
        # Some instruction datasets have inputs and targets
        inputs = sample.get("inputs", "")
        targets = sample.get("targets", "")
        if inputs or targets:
            text = f"{inputs}\n{targets}"
    return text.strip()

def build_tokenizer(dataset_stream, num_samples=100_000, vocab_size=64000):
    print(f"[*] Training BPETokenizer on first {num_samples} conversational samples...")
    sentences = []
    count = 0
    for sample in tqdm(dataset_stream, total=num_samples, desc="Reading Tokenizer Data"):
        text = _get_text(sample)
        if text:
            # Aya dataset contains multiline instructions/chat
            for line in text.split("\n"):
                if line.strip():
                    sentences.append(line.strip())
            count += 1
            if count >= num_samples:
                break
                
    bpe = SentencePieceBPETokenizer(vocab_size=vocab_size)
    print("[*] Performing BPE Merges...")
    bpe.train_on_sentences(sentences, num_merges=vocab_size, min_freq=2)
    print(f"[*] Tokenizer trained! Actual vocab size: {len(bpe.id2subword)}")
    return bpe

def token_generator(dataset_stream, bpe):
    """Generator that yields token IDs continuously from the dataset stream."""
    for sample in dataset_stream:
        text = _get_text(sample)
        if text:
            # We encode and yield each token
            try:
                token_ids = bpe.encode_as_ids(text)
                for tid in token_ids:
                    yield tid
            except Exception:
                pass # Ignore bad encoding on weird unicode if any

def main():
    parser = argparse.ArgumentParser(description="Massive Multilingual Streaming SVD Training")
    parser.add_argument("--dataset", type=str, default="CohereForAI/aya_collection", help="HuggingFace dataset to stream")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--vocab-size", type=int, default=64000, help="BPE vocabulary size")
    parser.add_argument("--rank", type=int, default=128, help="SVD Projection Rank")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Tokens per SVD partial_fit batch")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Batches per checkpoint (10 * 100k = 1M tokens)")
    args = parser.parse_args()

    print(f"[*] Loading {args.dataset} in Streaming Mode...")
    
    # Enable streaming to prevent RAM overflow
    ds = load_dataset(args.dataset, "aya_dataset", split=args.split, streaming=True)
    
    # 1. Train Tokenizer
    # Stream the first 50k samples for the tokenizer (enough to build vocab)
    bpe = build_tokenizer(iter(ds), num_samples=50_000, vocab_size=args.vocab_size)
    
    os.makedirs("artifacts", exist_ok=True)
    bpe_path = "artifacts/massive_bpe.pkl"
    with open(bpe_path, "wb") as f:
        pickle.dump(bpe.to_dict(), f)
    print(f"[*] Tokenizer saved to {bpe_path}")

    # 2. Initialize Engine
    print(f"[*] Initializing RIDM Ultra Engine (Vocab: {len(bpe.id2subword)}, Rank: {args.rank})")
    ridm = RIDM(vocab_size=len(bpe.id2subword), window=5)
    
    # 3. Stream Training
    # We must create a fresh iterator from the dataset to start from the beginning
    fresh_stream = load_dataset(args.dataset, "aya_dataset", split=args.split, streaming=True)
    token_stream = token_generator(iter(fresh_stream), bpe)
    
    print("\n=======================================================")
    print(f"[*] STARTING MASSIVE STREAMING SVD (Batch: {args.batch_size})")
    print("=======================================================")
    print("Press Ctrl+C at any time to stop safely. Checkpoints will be saved automatically.")
    
    try:
        ridm.streaming_fit(
            data_iterator=token_stream,
            rank=args.rank,
            checkpoint_dir="checkpoints_massive",
            checkpoint_interval=args.checkpoint_interval,
            batch_size=args.batch_size
        )
    except KeyboardInterrupt:
        print("\n[*] Training safely interrupted by user.")
    except Exception as e:
        print(f"\n[!] Training error: {e}")

if __name__ == "__main__":
    main()
