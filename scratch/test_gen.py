import asyncio
import sys
from ridm_ultra.llm.native_decoder import NativeDecoder

sys.stdout.reconfigure(encoding='utf-8')

async def main():
    dec = NativeDecoder()
    prompt = "hayat nasil gidiyor"
    print(f"Prompt: '{prompt}'")
    tokens = []
    async for tok in dec.generate_stream(prompt, max_new_tokens=25):
        tokens.append(tok)
    print("\nGenerated full text:")
    print("".join(tokens))

if __name__ == "__main__":
    asyncio.run(main())
