import httpx
import sys

sys.stdout.reconfigure(encoding='utf-8')

def test_v8_pure_dynamic():
    print("=== RIDM ULTRA V8.0 PURE DYNAMIC CONVERSATIONAL TEST ===")
    with httpx.Client(timeout=120.0) as client:
        for q in ["selam", "nasılsın", "3 ile 5'i çarparsan kaç eder?", "biraz uzun bir hikaye olsun"]:
            print(f"--> USER: '{q}'")
            res = client.post('http://127.0.0.1:8000/api/v1/chat/stream', json={'message': q})
            tokens = []
            for chunk in res.text.split('\n\n'):
                if 'data: ' in chunk and '"delta": "' in chunk:
                    try:
                        delta = chunk.split('"delta": "')[1].split('"')[0]
                        tokens.append(delta)
                    except Exception:
                        pass
            print(f"--> SYSTEM RESPONSE:\n{''.join(tokens).replace('\\n', '\n')}\n")

if __name__ == "__main__":
    test_v8_pure_dynamic()
