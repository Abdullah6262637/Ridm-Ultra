import asyncio
import json
import re
import sys
import numpy as np
from pathlib import Path
from ridm_ultra.backend import ComputeBackend

sys.stdout.reconfigure(encoding='utf-8')

vocab = json.loads(Path('artifacts/ridm_fineweb_vocab.json').read_text(encoding='utf-8'))
word2idx = {w: i for i, w in enumerate(vocab)}
word_emb = np.load('artifacts/ridm_fineweb_embeddings.npz')['word_emb'].astype(np.float32)
tr_words = set(json.loads(Path('artifacts/turkish_valid_words.json').read_text(encoding='utf-8')))

raw_ngram = json.loads(Path('artifacts/ngram_3gram_transitions.json').read_text(encoding='utf-8'))
bigram_next = {}
for key, counts in raw_ngram.items():
    w1_str, w2_str = key.split("_")
    w1_id, w2_id = int(w1_str), int(w2_str)
    bigram_next[(w1_id, w2_id)] = {int(k): v for k, v in counts.items()}

stop_words = {"ve", "için", "et", "bu", "o", "da", "de", "ile", "bir", "mi", "mu", "mı", "mü", "ne", "ki", "en", "her", "daha", "kadar", "ise", "hem"}
stop_ids = {word2idx[w] for w in stop_words if w in word2idx}

eng_junk = {
    'hitler', 'storm', 'hard', 'nasa', 'soda', 'eine', 'melissa', 'zhang', 'serum', 'familyasından', 'plus', 'ibn', 'für', 'dem', 'des', 'das', 'ir', 'au', 'inc', 'die', 'ich', 'sich', 'por', 'di', 'sent', 'hzamiri', 'ui', 'ang', 'chelsea', 'hill', 'kraliyet', 'and', 'or', 'not', 'can', 'will', 'would', 'could', 'should', 'all', 'any', 'some', 'no', 'make', 'do', 'see', 'take', 'go', 'know', 'think', 'use', 'find', 'give', 'tell', 'work', 'call', 'try', 'ask', 'need', 'feel', 'become', 'leave', 'put', 'people', 'your', 'group', 'kumar', 'khan', 'au', 'habitat', 'team', 'part', 'run', 'kids', 'key', 'led', 'food', 'hit', 'cover', 'ovat', 'ein', 'um', 'zu', 'ni', 'oh', 'never', 'pour', 'il', 'les', 'du', 'te', 'et'
}

mask = []
for w in vocab:
    wl = w.lower()
    clean_w = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜ]', '', wl)
    if not clean_w or clean_w in eng_junk:
        mask.append(False)
        continue
    if any(c in 'xwqXWQ' for c in w):
        mask.append(False)
        continue
    if w[0].isupper() and not any(c in 'ÇĞİÖŞÜ' for c in w[:2]):
        mask.append(False)
        continue
    is_tr_char = any(c in "çğışöüÇĞIŞÖÜ" for c in w)
    if (wl in tr_words or is_tr_char) and len(clean_w) >= 2:
        mask.append(True)
    else:
        mask.append(False)

valid_mask = np.array(mask, dtype=bool)
print(f"Strict clean Turkish mask count: {valid_mask.sum()} / {len(vocab)}")
sample = [w for w, m in zip(vocab, valid_mask) if m][:40]
print("Sample clean tokens:", sample)

backend = ComputeBackend(device="cpu")
prompt = "hayat nasil gidiyor"
prompt_ids = [word2idx.get(w, 0) for w in prompt.split() if w in word2idx]
current_ids = list(prompt_ids) if prompt_ids else [0]
ctx_vec = word_emb[current_ids].mean(axis=0)
ctx_vec /= (np.linalg.norm(ctx_vec) + 1e-8)
rng = np.random.RandomState(42)
tr_chars = set("abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")

out_words = []
for step_i in range(25):
    logits = backend.matvec(word_emb, ctx_vec)
    logits[~valid_mask] -= 1000.0

    if len(current_ids) >= 2:
        ctx_pair = (current_ids[-2], current_ids[-1])
        next_counts = bigram_next.get(ctx_pair)
        if next_counts:
            total = float(sum(next_counts.values()))
            for nxt_id, count in next_counts.items():
                if nxt_id < len(logits) and valid_mask[nxt_id]:
                    logits[nxt_id] += 35.0 * (count / total)

    if current_ids[-1] in stop_ids:
        for s_id in stop_ids:
            if s_id < len(logits):
                logits[s_id] -= 15.0

    recent_ids = set(current_ids[-8:])
    for r_id in recent_ids:
        if 0 <= r_id < len(logits):
            logits[r_id] -= 6.0

    next_idx = backend.sample_logits(logits, temperature=0.5, top_k=5, rng=rng)
    word = vocab[next_idx]
    clean_w = "".join(c for c in word if c in tr_chars)
    if clean_w:
        out_words.append(clean_w)
        current_ids.append(next_idx)
        next_vec = word_emb[next_idx]
        ctx_vec = 0.60 * ctx_vec + 0.40 * (next_vec / (np.linalg.norm(next_vec) + 1e-8))
        ctx_vec /= (np.linalg.norm(ctx_vec) + 1e-8)

print("\nGenerated text:")
print(" ".join(out_words))
