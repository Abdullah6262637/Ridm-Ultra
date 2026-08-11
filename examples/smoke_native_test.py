import asyncio
import time

from ridm_ultra.api.dependencies import get_rag_engine
from ridm_ultra.chat import ChatEngine, SQLiteChatRepository


async def main():
    print("=== RIDM ULTRA 100% NATIVE C++ SVD ENGINE SMOKE TEST ===")
    repo = SQLiteChatRepository("artifacts/chat_sessions.db")
    rag = get_rag_engine()
    chat_engine = ChatEngine(repository=repo, rag_engine=rag)

    session = await chat_engine.get_or_create_session()

    queries = [
        "Selam",
        "Yapay zeka modelleri nasıl çalışır?",
        "Türkiye'nin başkenti neresidir?"
    ]

    for q in queries:
        t0 = time.time()
        print(f"\nUser Query   : '{q}'")
        full_text = ""
        async for chunk in chat_engine.chat_stream(session.session_id, q):
            full_text += chunk.delta

        total_time = time.time() - t0
        print(f"Total Time   : {total_time:.2f} s")
        print(f"Response Text:\n{full_text}")

if __name__ == "__main__":
    asyncio.run(main())
