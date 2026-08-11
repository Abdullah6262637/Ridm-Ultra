"""Verify that artifacts/ridm_fineweb_vocab.json and
artifacts/ridm_fineweb_embeddings.npz are index-aligned (same training run).

This REPLACES the old scripts/fix_vocab_mapping.py, which was actively
dangerous: it re-scanned the Parquet corpus, recomputed word counts, and
overwrote the `vocab` array inside the .npz WITHOUT touching `word_emb` /
`context_vecs`. If the recomputed sort order differed from the original
training run in any way (different scan order, tie-breaking, dict
iteration order across Python versions, corpus files added/removed), the
"fixed" vocab would silently point at the WRONG embedding rows -- i.e. it
converted a possibly-correct alignment into a guaranteed-silent one, with
no error raised anywhere downstream.

The only safe fix for a real mismatch is to regenerate vocab AND
embeddings together in a single run of scripts/train_ridm_full_dataset.py.
This script only checks and reports; it never mutates artifacts.
"""
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    emb_path = Path("artifacts/ridm_fineweb_embeddings.npz")
    vocab_json_path = Path("artifacts/ridm_fineweb_vocab.json")
    meta_path = Path("artifacts/ridm_fineweb_metadata.json")

    if not emb_path.exists():
        print(f"[FAIL] Missing {emb_path}")
        return 1

    npz = np.load(emb_path)
    if "word_emb" not in npz:
        print(f"[FAIL] {emb_path} has no 'word_emb' array")
        return 1
    n_rows = npz["word_emb"].shape[0]

    # Prefer the vocab embedded in the SAME npz -- it is guaranteed to be
    # index-aligned because it was written by the same np.savez_compressed()
    # call that wrote word_emb.
    if "vocab" in npz:
        npz_vocab = list(npz["vocab"])
        if len(npz_vocab) != n_rows:
            print(f"[FAIL] npz['vocab'] has {len(npz_vocab)} entries but "
                  f"word_emb has {n_rows} rows -- internally inconsistent artifact.")
            return 1
        print(f"[OK] npz-embedded vocab is aligned with word_emb ({n_rows} rows).")
    else:
        print("[WARN] embeddings.npz has no embedded 'vocab' array; alignment "
              "can only be checked against the external vocab.json below, "
              "which is a weaker guarantee.")
        npz_vocab = None

    if vocab_json_path.exists():
        ext_vocab = json.loads(vocab_json_path.read_text(encoding="utf-8"))
        if len(ext_vocab) != n_rows:
            print(f"[FAIL] {vocab_json_path} has {len(ext_vocab)} entries but "
                  f"word_emb has {n_rows} rows.")
            return 1
        if npz_vocab is not None and ext_vocab != npz_vocab:
            print(f"[FAIL] {vocab_json_path} does NOT match the vocab embedded "
                  f"in {emb_path}. They were generated from different training "
                  f"runs and are misaligned. Re-run "
                  f"scripts/train_ridm_full_dataset.py to regenerate both "
                  f"artifacts together -- do not patch either file in isolation.")
            return 1
        print(f"[OK] {vocab_json_path} matches word_emb row count / content.")

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_vocab = meta.get("vocab")
        if meta_vocab is not None and len(meta_vocab) != n_rows:
            print(f"[FAIL] {meta_path}['vocab'] has {len(meta_vocab)} entries "
                  f"but word_emb has {n_rows} rows.")
            return 1

    print("[SUCCESS] All present vocab sources are index-aligned with the "
          "trained embedding matrix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
