"""Smoke test script for NativeDecoder zero-backprop generation."""
import asyncio
import time

from ridm_ultra.llm.native_decoder import NativeDecoder


async def main():
    print("=== RIDM ULTRA NATIVE ZERO-BACKPROP DECODER SMOKE TEST ===")
    decoder = NativeDecoder()

    prompts = [
        "selam",
        "nasılsın",
        "yapay zeka ve veri bilimi"
    ]


    for p in prompts:
        t0 = time.time()
        tokens = []
        async for token in decoder.generate_stream(prompt=p, max_new_tokens=20, temperature=0.7, top_k=8):
            tokens.append(token)

        gen_time = time.time() - t0
        full_response = "".join(tokens)
        tps = len(tokens) / gen_time if gen_time > 0 else 0

        print(f"\nPrompt          : '{p}'")
        print(f"Generated Tokens: {len(tokens)} tokens in {gen_time:.3f}s ({tps:.1f} tok/s)")
        print(f"Output Text     : '{full_response.strip()}'")

    print("\n[SUCCESS] Native Zero-Backprop Decoder Smoke Test Completed Successfully!")


if __name__ == "__main__":
    asyncio.run(main())
