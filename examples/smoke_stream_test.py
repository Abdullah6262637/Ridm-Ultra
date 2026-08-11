"""Smoke test script for verifying ChatEngine & SSE stream latency."""
import asyncio
import time

from ridm_ultra.api.dependencies import get_chat_engine


async def main():
    print("=== RIDM ULTRA SSE STREAM LATENCY SMOKE TEST ===")
    chat_engine = get_chat_engine()
    session = await chat_engine.get_or_create_session()

    queries = ["Selam", "Merhaba", "Explain quantum computing."]

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

        print(f"\nQuery        : '{q}'")
        print(f"TTFT Latency : {ttft:.2f} ms")
        print(f"Total Stream : {total_time * 1000:.2f} ms")
        print(f"Response Text: '{full_text}'")

        assert ttft < 500.0, f"TTFT latency exceeded 500ms limit! ({ttft:.2f} ms)"

    print("\n[SUCCESS] SSE Streaming Smoke Test Passed! (<500ms TTFT)")


if __name__ == "__main__":
    asyncio.run(main())
