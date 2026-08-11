import json
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

# Basic list of valid pure Turkish words (2 letters, 3 letters, common roots)
VALID_2LETTER_TR = {"bu", "de", "da", "ne", "ki", "en", "ve", "ya", "su", "ev", "at", "ip", "iç", "öz", "ek", "ön", "ad", "al", "ol", "et", "mi", "mı", "mu", "mü"}

# English/foreign words to remove
ENGLISH_LOANWORDS = {
    'rugby', 'hardcore', 'yard', 'test', 'meme', 'blog', 'poker', 'hamburger',
    'solo', 'bonus', 'vegan', 'golf', 'demo', 'patent', 'salon', 'atom', 'film',
    'modern', 'melanin', 'blues', 'magma', 'gastrointestinal', 'opera', 'liberal',
    'ultra', 'home', 'oval', 'feet', 'fare', 'pasta', 'her', 'Love', 'John', 'karate',
    'kilogram', 'yoga', 'bale', 'karma', 'soya', 'folk', 'organ', 'motor', 'mineral',
    'viral', 'moral', 'metal', 'normal', 'video', 'beta', 'amino', 'seven', 'fitness',
    'hemoglobin', 'paranormal', 'vintage', 'hormonal', 'format', 'model', 'minimal',
    'size', 'reuters', 'form', 'mark', 'cheshire', 'hitler', 'storm', 'hard', 'nasa',
    'soda', 'eine', 'melissa', 'zhang', 'serum', 'familyasından', 'iyelik', 'ünlem'
}

def main():
    path = 'artifacts/turkish_valid_words.json'
    words = set(json.loads(open(path, encoding='utf-8').read()))
    print(f"Current dict size: {len(words)}")

    cleaned = set()
    for w in words:
        wl = w.lower()
        if wl in ENGLISH_LOANWORDS:
            continue
        # Filter out obvious non-Turkish consonant clusters or endings
        if re.search(r'(th|sh|ch|gh|ph|wh|ck|qu|tion|sion|ment|ness|less|able|ible|ity|ive|ism|ist|ize|ous|ful|ed$|ing$|est$)', wl):
            continue
        if any(c in 'xwqXWQ' for c in w):
            continue
        cleaned.add(w)

    print(f"Purified dict size: {len(cleaned)}")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(cleaned)), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
