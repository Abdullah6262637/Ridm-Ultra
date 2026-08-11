import asyncio
import json
import re
import sys
import time
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

eng_junk = {
    'food', 'hit', 'pour', 'cover', 'pro', 'au', 'soy', 'men', 'i', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'out', 'get', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'this', 'that', 'you', 'are', 'is', 'it', 'be', 'or', 'an', 'as', 'do', 'if', 'my', 'no', 'so', 'up', 'us', 'we', 'he', 'she', 'they', 'have', 'like', 'one', 'two', 'three', 'four', 'five', 'program', 'come', 'got', 'last', 'back', 'main', 'file', 'return', 'true', 'false', 'null', 'none', 'class', 'void', 'public', 'private', 'static', 'die', 'ich', 'sich', 'por', 'ir', 'si', 'di', 'sent', 'hzamiri', 'ui', 'ang', 'chelsea', 'hill', 'kraliyet', 'and', 'or', 'not', 'can', 'will', 'would', 'could', 'should', 'all', 'any', 'some', 'no', 'make', 'do', 'see', 'take', 'go', 'know', 'think', 'use', 'find', 'give', 'tell', 'work', 'call', 'try', 'ask', 'need', 'feel', 'become', 'leave', 'put', 'people', 'your', 'group', 'kumar', 'khan', 'au', 'habitat'
}
eng_suffixes = ('ing', 'tion', 'sion', 'ment', 'ness', 'less', 'able', 'ible', 'ity', 'ive', 'ism', 'ist', 'ize', 'ous', 'ful', 'ed', 'est')
eng_patterns = ('th', 'sh', 'ch', 'gh', 'ph', 'wh', 'ck', 'qu')
valid_2letter = {"bu", "de", "da", "ne", "ki", "en", "ve", "ya", "su", "ev", "at", "ip", "iç", "öz", "ek", "ön", "ad", "al", "ol", "et", "mi", "mı", "mu", "mü"}

valid_mask = []
for w in vocab:
    w_lower = w.lower()
    clean_w = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜ]', '', w_lower)
    if not clean_w or clean_w in eng_junk:
        valid_mask.append(False)
        continue
    if any(clean_w.endswith(s) for s in eng_suffixes) or any(p in clean_w for p in eng_patterns):
        valid_mask.append(False)
        continue
    if len(clean_w) == 1 and not any(c in "çğışöüÇĞIŞÖÜ" for c in w):
        valid_mask.append(False)
        continue
    if len(clean_w) == 2 and clean_w not in valid_2letter and not any(c in "çğışöüÇĞIŞÖÜ" for c in w):
        valid_mask.append(False)
        continue
    
    is_tr_char = any(c in "çğışöüÇĞIŞÖÜ" for c in w)
    if is_tr_char or (clean_w in tr_words and w[0].islower()):
        valid_mask.append(True)
    else:
        valid_mask.append(False)

valid_mask = np.array(valid_mask, dtype=bool)
print(f"Valid mask count: {valid_mask.sum()} / {len(vocab)}")

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
                    logits[nxt_id] += 25.0 * (count / total)

    recent_ids = set(current_ids[-8:])
    for r_id in recent_ids:
        if 0 <= r_id < len(logits):
            logits[r_id] -= 5.0

    next_idx = backend.sample_logits(logits, temperature=0.6, top_k=5, rng=rng)
    word = vocab[next_idx]
    clean_w = "".join(c for c in word if c in tr_chars)
    if clean_w:
        out_words.append(clean_w)
        current_ids.append(next_idx)
        next_vec = word_emb[next_idx]
        ctx_vec = 0.60 * ctx_vec + 0.40 * (next_vec / (np.linalg.norm(next_vec) + 1e-8))
        ctx_vec /= (np.linalg.norm(ctx_vec) + 1e-8)

print("Generated text:")
print(" ".join(out_words))
