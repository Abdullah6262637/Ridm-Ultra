import asyncio
import sys
import time
from ridm_ultra.llm.native_decoder import NativeDecoder

sys.stdout.reconfigure(encoding='utf-8')

async def main():
    dec = NativeDecoder()
    prompt = "Türkiye'nin başkenti neresidir?"
    print(f"Testing NativeDecoder with prompt: '{prompt}'")
    
    t0 = time.perf_counter()
    tokens = []
    async for tok in dec.generate_stream(prompt, max_new_tokens=20):
        tokens.append(tok)
    
    elapsed = (time.perf_counter() - t0) * 1000.0
    print(f"\nGenerated Output: {''.join(tokens)}")
    print(f"Total stream generation time: {elapsed:.2f} ms")

if __name__ == "__main__":
    asyncio.run(main())
