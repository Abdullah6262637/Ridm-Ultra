import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

vocab = json.loads(Path('artifacts/ridm_fineweb_vocab.json').read_text(encoding='utf-8'))
tr_valid = set(json.loads(Path('artifacts/turkish_valid_words.json').read_text(encoding='utf-8')))

eng_suffixes = ('ing', 'tion', 'sion', 'ment', 'ness', 'less', 'able', 'ible', 'ity', 'ive', 'ism', 'ist', 'ize', 'ous', 'ful', 'ed', 'est')
eng_patterns = ('th', 'sh', 'ch', 'gh', 'ph', 'wh', 'ck', 'qu')
eng_words = {
    'the', 'to', 'get', 'out', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'this', 'that', 'you', 'are', 'is', 'it', 'be', 'or', 'an', 'as', 'do', 'if', 'my', 'no', 'so', 'up', 'us', 'we', 'he', 'she', 'they', 'have', 'like',
    'one', 'two', 'three', 'four', 'five', 'program', 'come', 'got', 'last', 'back', 'main', 'file', 'return', 'true', 'false', 'null', 'none', 'class', 'void', 'public', 'private', 'static', 'die', 'ich', 'sich', 'por', 'ir', 'si', 'di', 'sent', 'hzamiri', 'ui', 'ang', 'chelsea', 'hill', 'kraliyet', 'and', 'or', 'not', 'can', 'will', 'would', 'could', 'should', 'all', 'any', 'some', 'no', 'make', 'do', 'see', 'take', 'go', 'know', 'think', 'use', 'find', 'give', 'tell', 'work', 'call', 'try', 'ask', 'need', 'feel', 'become', 'leave', 'put'
}

clean_vocab = []
for w in vocab:
    w_lower = w.lower()
    clean_w = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜ]', '', w_lower)
    if not clean_w or clean_w in eng_words:
        continue
    if any(clean_w.endswith(s) for s in eng_suffixes):
        continue
    if any(p in clean_w for p in eng_patterns):
        continue
    if clean_w in tr_valid or any(c in 'çğışöüÇĞIŞÖÜ' for c in w):
        clean_vocab.append(w)

print(f"Original vocab count: {len(vocab)}")
print(f"Clean pure Turkish vocab count: {len(clean_vocab)}")
print("Sample clean words:", clean_vocab[:40])
