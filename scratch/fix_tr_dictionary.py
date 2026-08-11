import json
import re
from pathlib import Path

raw_tr = json.loads(Path('artifacts/turkish_valid_words.json').read_text(encoding='utf-8'))

foreign_words = {
    'del', 'con', 'fit', 'talk', 'free', 'risk', 'plan', 'tip', 'ali', 'pro', 'au', 'soy', 'men', 'out', 'get', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'this', 'that', 'you', 'are', 'is', 'it', 'be', 'or', 'an', 'as', 'do', 'if', 'my', 'no', 'so', 'up', 'us', 'we', 'he', 'she', 'they', 'have', 'like', 'one', 'two', 'three', 'four', 'five', 'program', 'come', 'got', 'last', 'back', 'main', 'file', 'return', 'true', 'false', 'null', 'none', 'class', 'void', 'public', 'private', 'static', 'die', 'ich', 'sich', 'por', 'ir', 'si', 'di', 'sent', 'hzamiri', 'ui', 'ang', 'chelsea', 'hill', 'kraliyet', 'inc', 'plus', 'ibn', 'für', 'modal', 'dont', 'joka', 'jotka', 'nebo', 'ett', 'ith', 'na', 'dr', 'people', 'your', 'group', 'kumar', 'khan', 'habitat', 'team', 'part', 'run', 'kids', 'key', 'led', 'food', 'hit', 'cover', 'ovat', 'ein', 'um', 'zu', 'ni', 'oh', 'never', 'pour', 'il', 'les', 'du', 'te', 'et'
}

clean_tr = set()
for w in raw_tr:
    wl = w.lower().strip()
    if not wl or wl in foreign_words:
        continue
    if any(p in wl for p in ('th', 'sh', 'ch', 'gh', 'ph', 'wh', 'ck', 'qu', 'x', 'w', 'q')):
        continue
    if len(wl) <= 3 and not any(c in "çğışöüÇĞIŞÖÜ" for c in w) and wl not in {"bu", "de", "da", "ne", "ki", "en", "ve", "ya", "su", "ev", "at", "ip", "iç", "öz", "ek", "ön", "ad", "al", "ol", "et", "mi", "mı", "mu", "mü", "bir", "iki", "üç", "tek", "çok", "var", "yok", "son", "ilk", "her", "şey", "hem", "sen", "ben", "biz", "siz", "onn", "ona", "onu", "ile", "için", "gibi", "kadar", "daha", "biri", "diğer", "aynı", "diye", "göre"}:
        continue
    clean_tr.add(wl)

Path('artifacts/turkish_valid_words.json').write_text(
    json.dumps(sorted(list(clean_tr)), ensure_ascii=False), encoding='utf-8'
)
print(f"Definitively purged turkish_valid_words.json to {len(clean_tr)} pure Turkish words.")
