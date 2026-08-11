"""Unit & Integration tests for ridm_ultra.chat subsystem."""
from __future__ import annotations

import pytest

from ridm_ultra.chat import (
    ChatMessage,
    ChatSession,
    HierarchicalMemoryManager,
    MessageRole,
    ModelTier,
    SemanticRouter,
    SQLiteChatRepository,
)


@pytest.mark.asyncio
async def test_hierarchical_memory_manager():
    mem = HierarchicalMemoryManager(target_max_tokens=100, summarization_threshold_ratio=0.5)
    session = ChatSession(system_prompt="System prompt test.")

    # Add messages to exceed threshold
    for i in range(10):
        session.add_message(ChatMessage(role=MessageRole.USER, content=f"Message number {i} with some extra token text."))
        session.add_message(ChatMessage(role=MessageRole.ASSISTANT, content=f"Response number {i} with detailed text explanation."))

    context_msgs = mem.get_context_messages(session, max_token_budget=100)
    assert len(context_msgs) < len(session.messages)

    summary = await mem.summarize_if_needed(session, max_tokens=100)
    assert summary is not None
    assert "summary" in session.metadata


def test_semantic_router():
    router = SemanticRouter()

    # Fast tier patterns
    assert router.classify_intent("Hello!") == ModelTier.FAST
    assert router.classify_intent("selamlar") == ModelTier.FAST

    # Reasoning tier keywords & long queries
    assert router.classify_intent("Can you debug this python function algorithm?") == ModelTier.REASONING
    assert router.classify_intent("Write a mathematical proof for SVD convergence.") == ModelTier.REASONING


@pytest.mark.asyncio
async def test_sqlite_repository(temp_db_path):
    repo = SQLiteChatRepository(db_path=temp_db_path)

    session = ChatSession(title="Repository Test Session")
    session.add_message(ChatMessage(role=MessageRole.USER, content="Hello repository!"))
    await repo.save_session(session)

    loaded = await repo.get_session(session.session_id)
    assert loaded is not None
    assert loaded.title == "Repository Test Session"
    assert len(loaded.messages) == 1

    sessions_list = await repo.list_sessions()
    assert len(sessions_list) >= 1

    deleted = await repo.delete_session(session.session_id)
    assert deleted is True


@pytest.mark.asyncio
async def test_chat_engine_streaming(sample_chat_engine):
    session = await sample_chat_engine.get_or_create_session(system_prompt="You are a unit test assistant.")
    session_id = session.session_id

    chunks = []
    async for chunk in sample_chat_engine.chat_stream("Explain SVD in simple terms", session_id=session_id):
        chunks.append(chunk.delta)

    full_resp = "".join(chunks)
    assert len(full_resp) > 0

    reloaded_session = await sample_chat_engine.repository.get_session(session_id)
    assert len(reloaded_session.messages) == 2  # User + Assistant
