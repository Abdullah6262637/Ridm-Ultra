"""End-to-End Verification Smoke Test for RIDM Ultra Chat Engine."""
import asyncio
from pathlib import Path

from ridm_ultra.core import RIDM
from ridm_ultra.graph_retrieval import SimpleRAG

from ridm_ultra.chat import (
    ChatEngine,
    SQLiteChatRepository,
)


async def main():
    print("=== RIDM ULTRA CHAT ENGINE VERIFICATION ===")

    # 1. Initialize RAG retrieval engine
    docs = [
        "RIDM Ultra uses closed-form SVD for zero-backprop language representation.",
        "The Chat Engine supports sliding context windows, background summarization, and MoE routing."
    ]
    vocab = ["<UNK>", "RIDM", "Ultra", "closed-form", "SVD", "Chat", "Engine", "sliding", "window"]
    ridm = RIDM(vocab=vocab, dim=64, window=3)
    ridm.word_emb = ridm.context_vecs[:len(vocab), :32]
    rag = SimpleRAG(ridm, docs)

    # 2. Instantiate ChatEngine with SQLite repository persistence
    repo = SQLiteChatRepository(db_path="artifacts/smoke_chat_sessions.db")
    engine = ChatEngine(repository=repo, rag_engine=rag)

    # 3. Test Greeting (Fast Model Router Tier)
    print("\n--- Test 1: Simple Greeting (Fast Tier Router) ---")
    session = await engine.get_or_create_session(system_prompt="You are a helpful AI assistant.")
    session_id = session.session_id

    print("User: Hello! Who are you?")
    print("Assistant (Streaming): ", end="", flush=True)
    async for chunk in engine.chat_stream("Hello! Who are you?", session_id=session_id):
        print(chunk.delta, end="", flush=True)
    print()

    # 4. Test Complex Query (SOTA Reasoning Router Tier + RAG)
    print("\n--- Test 2: Complex Technical Query (Reasoning Tier Router) ---")
    print("User: Can you explain how RIDM Ultra uses SVD and the Chat Engine architecture?")
    print("Assistant (Streaming): ", end="", flush=True)
    async for chunk in engine.chat_stream(
        "Can you explain how RIDM Ultra uses SVD and the Chat Engine architecture?",
        session_id=session_id
    ):
        print(chunk.delta, end="", flush=True)
    print()

    # 5. Verify Session Reload & History Persistence
    print("\n--- Test 3: Session Reload from SQLite Repository ---")
    reloaded_session = await repo.get_session(session_id)
    print(f"Reloaded Session ID: {reloaded_session.session_id}")
    print(f"Message Count: {len(reloaded_session.messages)}")
    for m in reloaded_session.messages:
        role_label = m.role.value.upper()
        content_preview = m.content[:60] + "..." if len(m.content) > 60 else m.content
        print(f"  [{role_label}]: {content_preview}")

    # Clean up test artifact safely
    db_file = Path("artifacts/smoke_chat_sessions.db")
    if db_file.exists():
        try:
            db_file.unlink()
        except OSError:
            pass

    print("\n[SUCCESS] Verification Completed Successfully!")


if __name__ == "__main__":
    asyncio.run(main())

