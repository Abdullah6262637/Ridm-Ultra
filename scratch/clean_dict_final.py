import json
import re
from pathlib import Path

raw_vocab = json.loads(Path('artifacts/ridm_fineweb_vocab.json').read_text(encoding='utf-8'))
tr_valid_raw = set(json.loads(Path('artifacts/turkish_valid_words.json').read_text(encoding='utf-8')))

foreign_words = {
    'last', 'team', 'part', 'run', 'kids', 'key', 'ein', 'ovat', 'led', 'food', 'hit', 'cover', 'pro', 'au', 'soy', 'men', 'out', 'get', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'this', 'that', 'you', 'are', 'is', 'it', 'be', 'or', 'an', 'as', 'do', 'if', 'my', 'no', 'so', 'up', 'us', 'we', 'he', 'she', 'they', 'it', 'is', 'am', 'are', 'be', 'by', 'me', 'la', 'le', 'el', 'un', 'une', 'des', 'der', 'das', 'den', 'dem', 'von', 'aus', 'ist', 'sind', 'war', 'ja', 'nein', 'und', 'mit', 'sie', 'auf', 'sur', 'die', 'ich', 'sich', 'hzamiri', 'ui', 'ang', 'chelsea', 'hill', 'kraliyet', 'inc', 'plus', 'ibn', 'für', 'modal', 'dont', 'joka', 'jotka', 'nebo', 'ett', 'ith', 'na', 'dr', 'main', 'file', 'return', 'true', 'false', 'null', 'none', 'class', 'void', 'public', 'private', 'static', 'people', 'your', 'group', 'kumar', 'khan', 'habitat', 'got', 'come', 'where', 'when', 'what', 'why', 'how', 'who', 'which', 'there', 'here', 'then', 'than', 'these', 'those', 'top', 'new', 'best', 'good', 'day', 'time', 'world', 'life', 'first', 'all', 'more', 'over', 'into', 'some', 'them', 'other', 'only', 'most', 'make', 'just', 'look', 'know', 'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'well', 'way', 'even', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'um', 'si', 'zu', 'ni', 'oh', 'never', 'pour', 'il', 'les', 'du', 'te', 'et'
}

clean_words = set()
for w in tr_valid_raw:
    w_l = w.lower().strip()
    if not w_l or w_l in foreign_words:
        continue
    if any(p in w_l for p in ('th', 'sh', 'ch', 'gh', 'ph', 'wh', 'ck', 'qu', 'x', 'w', 'q')):
        continue
    if len(w_l) == 2 and w_l not in {"bu", "de", "da", "ne", "ki", "en", "ve", "ya", "su", "ev", "at", "ip", "iç", "öz", "ek", "ön", "ad", "al", "ol", "mi", "mı", "mu", "mü"}:
        continue
    clean_words.add(w_l)

Path('artifacts/turkish_valid_words.json').write_text(
    json.dumps(sorted(list(clean_words)), ensure_ascii=False), encoding='utf-8'
)
print(f"Purged turkish_valid_words.json to {len(clean_words)} clean words.")
