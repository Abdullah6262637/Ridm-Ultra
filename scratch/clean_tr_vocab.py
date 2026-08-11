import json
from pathlib import Path

raw_tr = json.loads(Path('artifacts/turkish_valid_words.json').read_text(encoding='utf-8'))

foreign_short = {
    'um', 'si', 'zu', 'ni', 'oh', 'never', 'pour', 'il', 'les', 'du', 'te', 'et', 'au', 'in', 'on', 'at', 'to', 'of', 'or', 'an', 'as', 'do', 'if', 'my', 'no', 'so', 'up', 'us', 'we', 'he', 'she', 'they', 'it', 'is', 'am', 'are', 'be', 'by', 'me', 'la', 'le', 'el', 'un', 'une', 'des', 'der', 'das', 'den', 'dem', 'von', 'aus', 'ist', 'sind', 'war', 'ja', 'nein', 'und', 'mit', 'sie', 'auf', 'sur', 'die', 'ich', 'sich', 'hzamiri', 'ui', 'ang', 'chelsea', 'hill', 'kraliyet', 'inc', 'plus', 'ibn', 'für', 'modal', 'dont', 'joka', 'jotka', 'nebo', 'ett', 'ith', 'na', 'dr', 'food', 'hit', 'cover', 'pro', 'soy', 'men', 'out', 'get', 'back', 'main', 'file', 'return', 'true', 'false', 'null', 'none', 'class', 'void', 'public', 'private', 'static', 'people', 'your', 'group', 'kumar', 'khan', 'habitat', 'got', 'come', 'where', 'when', 'what', 'why', 'how', 'who', 'which', 'there', 'here', 'then', 'than', 'this', 'that', 'these', 'those'
}

tr_clean = set()
for w in raw_tr:
    w_lower = w.lower().strip()
    if not w_lower or w_lower in foreign_short:
        continue
    # Drop words containing non-Turkish letter pairs
    if any(p in w_lower for p in ('th', 'sh', 'ch', 'gh', 'ph', 'wh', 'ck', 'qu', 'x', 'w', 'q')):
        continue
    # Drop 2-letter tokens except valid Turkish 2-letter words
    if len(w_lower) == 2 and w_lower not in {"bu", "de", "da", "ne", "ki", "en", "ve", "ya", "su", "ev", "at", "ip", "iç", "öz", "ek", "ön", "ad", "al", "ol", "mi", "mı", "mu", "mü"}:
        continue
    tr_clean.add(w_lower)

Path('artifacts/turkish_valid_words.json').write_text(
    json.dumps(sorted(list(tr_clean)), ensure_ascii=False), encoding='utf-8'
)
print(f"Updated artifacts/turkish_valid_words.json with {len(tr_clean)} pure Turkish words.")
