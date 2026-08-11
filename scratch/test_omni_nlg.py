import asyncio
import sys
sys.stdout.reconfigure(encoding='utf-8')
from ridm_ultra.chat.engine import ChatEngine

async def run_omni_tests():
    engine = ChatEngine()
    
    questions = [
        "Selam, nasılsın?",
        "Bana bir hikaye anlat.",
        "Bana uzay hakkında bir teori ortaya at.",
        "Sence spor yapmanın faydaları nelerdir?",
        "Python'da asenkron programlama nedir?"
    ]
    
    for i, q in enumerate(questions):
        print(f"\n--- Soru {i+1}: {q} ---")
        try:
            print("RIDM Ultra: ", end="", flush=True)
            async for chunk in engine.chat_stream(q, "user_xyz"):
                print(chunk.delta, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\nHATA: {e}")

if __name__ == "__main__":
    asyncio.run(run_omni_tests())
