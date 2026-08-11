"""Pytest shared fixtures for RIDM Ultra test suite."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from ridm_ultra.core import RIDM
from ridm_ultra.graph_retrieval import SimpleRAG

from ridm_ultra.chat import ChatEngine, SQLiteChatRepository


@pytest.fixture
def sample_vocab():
    return ["<UNK>", "the", "cat", "sat", "on", "mat", "dog", "barked", "run"]


@pytest.fixture
def sample_token_ids():
    return [1, 2, 3, 4, 5, 1, 6, 7, 1, 2, 3, 4, 8]


@pytest.fixture
def sample_ridm(sample_vocab):
    counts = {w: 5 for w in sample_vocab}
    model = RIDM(vocab=sample_vocab, counts=counts, dim=32, window=3, seed=42)
    # Fit dummy tokens
    model.partial_fit([1, 2, 3, 4, 5, 1, 2, 3, 6, 7])
    model.finalize(k=16)
    return model


@pytest.fixture
def temp_db_path():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        yield Path(tmpdir) / "test_chat_sessions.db"



@pytest.fixture
def sample_chat_engine(temp_db_path, sample_ridm):
    repo = SQLiteChatRepository(db_path=temp_db_path)
    docs = ["the cat sat on the mat", "the dog barked at the cat"]
    rag = SimpleRAG(sample_ridm, docs)
    engine = ChatEngine(repository=repo, rag_engine=rag)
    return engine
