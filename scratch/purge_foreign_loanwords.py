import json
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

def is_pure_turkish(word: str) -> bool:
    w = word.lower()
    # Explicit English / foreign words to purge
    bad_words = {
        'hormonal', 'mineral', 'viral', 'vintage', 'beta', 'moral', 'fitness',
        'hemoglobin', 'metal', 'paranormal', 'motor', 'folk', 'normal', 'video',
        'seven', 'organ', 'amino', 'feminist', 'hitler', 'storm', 'hard', 'nasa',
        'soda', 'eine', 'melissa', 'zhang', 'serum', 'familyasından', 'iyelik', 'ünlem',
        'format', 'model', 'minimal', 'size', 'reuters', 'form', 'mark', 'cheshire'
    }
    if w in bad_words:
        return False
    if any(c in 'xwqXWQ' for c in word):
        return False
    # Check for non-Turkish letter combinations at start/end
    if re.search(r'(th|sh|ch|gh|ph|wh|ck|qu|tion|sion|ment|ness|less|able|ible|ity|ive|ism|ist|ize|ous|ful)', w):
        return False
    return True

def main():
    path = 'artifacts/turkish_valid_words.json'
    words = json.loads(open(path, encoding='utf-8').read())
    print(f"Initial dictionary count: {len(words)}")

    clean_words = [w for w in words if is_pure_turkish(w)]
    print(f"Cleaned dictionary count: {len(clean_words)}")

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean_words, f, ensure_ascii=False, indent=2)
    print("Successfully updated artifacts/turkish_valid_words.json!")

if __name__ == "__main__":
    main()
