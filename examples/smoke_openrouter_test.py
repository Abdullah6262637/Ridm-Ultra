"""Smoke test for OpenRouter Nemotron-3-Ultra streaming integration with ChatEngine."""
import asyncio
import time

from ridm_ultra.api.dependencies import get_chat_engine


async def main():
    print("=== RIDM ULTRA OPENROUTER NEMOTRON-3-ULTRA CHAT ENGINE SMOKE TEST ===")
    chat_engine = get_chat_engine()
    session = await chat_engine.get_or_create_session()

    queries = [
        "AFC Ajax (amatörler) takımı ve Ajax Gençlik Akademisi ev sahibi maçlarını hangi stadyumda oynamaktadır?",
        "Türkiye'nin en uzun nehri hangisidir?"
    ]


    for q in queries:
        t0 = time.time()
        first_token_time = None
        chunks = []

        async for chunk in chat_engine.chat_stream(q, session_id=session.session_id):
            if first_token_time is None and chunk.delta:
                first_token_time = time.time()
            chunks.append(chunk.delta)

        total_time = time.time() - t0
        ttft = (first_token_time - t0) * 1000 if first_token_time else 0.0
        full_text = "".join(chunks)

        print(f"\nUser Query   : '{q}'")
        print(f"TTFT Latency : {ttft:.2f} ms")
        print(f"Total Stream : {total_time:.2f} s")
        print(f"Response Text: '{full_text}'")

    print("\n[SUCCESS] OpenRouter Nemotron-3-Ultra Chat Engine Smoke Test Passed!")


if __name__ == "__main__":
    asyncio.run(main())
