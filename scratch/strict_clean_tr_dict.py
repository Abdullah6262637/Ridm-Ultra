import json
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

def main():
    path = 'artifacts/turkish_valid_words.json'
    words = set(json.loads(open(path, encoding='utf-8').read()))
    print(f"Original size: {len(words)}")

    eng_bad = {
        'feet', 'fare', 'john', 'george', 'peter', 'paul', 'ad', 'her', 'home', 'love',
        'blues', 'magma', 'gastrointestinal', 'opera', 'liberal', 'ultra', 'oval',
        'pasta', 'karate', 'kilogram', 'yoga', 'bale', 'karma', 'soya', 'folk',
        'organ', 'motor', 'mineral', 'viral', 'moral', 'metal', 'normal', 'video'
    }

    clean = set()
    for w in words:
        wl = w.lower()
        if wl in eng_bad:
            continue
        if re.search(r'(th|sh|ch|gh|ph|wh|ck|qu|tion|sion|ment|ness|less|able|ible|ity|ive|ism|ist|ize|ous|ful)', wl):
            continue
        if any(c in 'xwqXWQ' for c in w):
            continue
        clean.add(w)

    print(f"Strict cleaned size: {len(clean)}")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(clean)), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
