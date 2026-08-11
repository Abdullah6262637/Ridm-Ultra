import json
import re
from pathlib import Path

raw_tr = json.loads(Path('artifacts/turkish_valid_words.json').read_text(encoding='utf-8'))

foreign_exact = {
    'in', 'on', 'at', 'by', 'for', 'with', 'from', 'this', 'that', 'you', 'are', 'is', 'it', 'be', 'or', 'an', 'as', 'do', 'if', 'my', 'no', 'so', 'up', 'us', 'we', 'he', 'she', 'they', 'have', 'like', 'one', 'two', 'three', 'four', 'five',
    'program', 'come', 'got', 'last', 'back', 'main', 'file', 'return', 'true', 'false', 'null', 'none', 'class', 'void', 'public', 'private', 'static', 'die', 'ich', 'sich', 'por', 'ir', 'si', 'di', 'sent', 'hzamiri', 'ui', 'ang', 'chelsea', 'hill', 'kraliyet', 'inc', 'plus', 'ibn', 'für', 'modal', 'dont', 'joka', 'jotka', 'nebo', 'ett', 'ith', 'na', 'dr', 'people', 'your', 'group', 'kumar', 'khan', 'au', 'habitat', 'team', 'part', 'run', 'kids', 'key', 'led', 'food', 'hit', 'cover', 'ovat', 'ein', 'um', 'zu', 'ni', 'oh', 'never', 'pour', 'il', 'les', 'du', 'te', 'et', 'vitamin', 'pentru', 'heavy', 'radar', 'element', 'oral', 'fast', 'role', 'reform', 'del', 'con', 'fit', 'talk', 'free', 'risk', 'plan', 'tip', 'ali', 'pro', 'soy', 'men'
}

start_clusters = ('tr', 'pr', 'dr', 'gr', 'cl', 'pl', 'st', 'sp', 'fl', 'fr', 'sk', 'sm', 'sn', 'sl', 'sc', 'str', 'spr', 'spl', 'br', 'cr')

pure_tr = set()
for w in raw_tr:
    wl = w.lower().strip()
    clean_w = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜ]', '', wl)
    if not clean_w or clean_w in foreign_exact:
        continue
    if any(c in 'xwqXWQ' for c in wl):
        continue
    if any(clean_w.startswith(c) for c in start_clusters):
        continue
    if any(p in clean_w for p in ('th', 'sh', 'ch', 'gh', 'ph', 'wh', 'ck', 'qu')):
        continue
    if len(clean_w) <= 3 and not any(c in "çğışöüÇĞIŞÖÜ" for c in wl) and clean_w not in {"bu", "de", "da", "ne", "ki", "en", "ve", "ya", "su", "ev", "at", "ip", "iç", "öz", "ek", "ön", "ad", "al", "ol", "et", "mi", "mı", "mu", "mü", "bir", "iki", "üç", "tek", "çok", "var", "yok", "son", "ilk", "her", "şey", "hem", "sen", "ben", "biz", "siz", "onn", "ona", "onu", "ile", "için", "gibi", "kadar", "daha", "biri", "diğer", "aynı", "diye", "göre"}:
        continue
    pure_tr.add(clean_w)

Path('artifacts/turkish_valid_words.json').write_text(
    json.dumps(sorted(list(pure_tr)), ensure_ascii=False), encoding='utf-8'
)
print(f"Filtered pure Turkish dictionary down to {len(pure_tr)} verified words.")
