import asyncio
import sys
import pandas as pd
from ridm_ultra.llm.native_decoder import NativeDecoder

sys.stdout.reconfigure(encoding='utf-8')

async def main():
    df = pd.read_parquet('data/raw/turkish_instructions.parquet')
    samples = df['text'].sample(5, random_state=123).tolist()
    
    dec = NativeDecoder()

    print("==================================================")
    print("RIDM ULTRA SVD CORE EVALUATION (5 GERÇEK VERİ SETİ ÖRNEĞİ)")
    print("==================================================\n")

    for i, text in enumerate(samples, 1):
        prompt = text.split('\n')[0][:80].strip()
        if not prompt:
            prompt = text[:80].strip()

        print(f"--- [ÖRNEK {i}] ---")
        print(f"📌 Girdi / Prompt         : {prompt}")
        
        # Native SVD Core Generation
        gen_tokens = []
        async for tok in dec.generate_stream(prompt, max_new_tokens=20):
            gen_tokens.append(tok)
        svd_text = "".join(gen_tokens).strip()

        print(f"⚡ Yerel SVD Üretimi          : {svd_text}")
        print(f"📚 Orijinal Veri Seti Metni  : {text[:150]}...\n")

if __name__ == "__main__":
    asyncio.run(main())
