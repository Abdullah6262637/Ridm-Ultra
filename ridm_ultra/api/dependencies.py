"""Dependency Injection Container for FastAPI routes."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from ridm_ultra.chat import ChatEngine, SQLiteChatRepository
from ridm_ultra.core import RIDM
from ridm_ultra.rag import HybridRetriever, SemanticChunker

logger = logging.getLogger(__name__)
_rag_lock = threading.Lock()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Singletons
_repository: Optional[SQLiteChatRepository] = None
_rag_engine: Optional[HybridRetriever] = None
_chat_engine: Optional[ChatEngine] = None


def get_repository() -> SQLiteChatRepository:
    global _repository
    if _repository is None:
        _repository = SQLiteChatRepository("artifacts/chat_sessions.db")
    return _repository


import asyncio

def _load_rag_engine_sync():
    global _rag_engine
    try:
        import json
        from pathlib import Path
        import numpy as np

        emb_path = Path("artifacts/ridm_fineweb_embeddings.npz")
        vocab_path = Path("artifacts/ridm_fineweb_vocab.json")

        if emb_path.exists():
            npz = np.load(emb_path)
            word_emb = npz["word_emb"].astype(np.float32)

            if vocab_path.exists():
                vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            elif "vocab" in npz:
                vocab = list(npz["vocab"])
            else:
                vocab = ["<UNK>"] + [f"w_{i}" for i in range(word_emb.shape[0] - 1)]

            counts = {w: 1 for w in vocab}
            ridm = RIDM(vocab=vocab, counts=counts, dim=word_emb.shape[1], window=3)
            ridm.word_emb = word_emb

            docs = [
                "RIDM Ultra v6 uses closed-form SVD for zero-backprop language representation.",
                "The Chat Engine supports sliding context windows, background summarization, and MoE routing."
            ]

            # INJECT KNOWLEDGE BASE TEXT FOR FACTUAL QA
            with open(str(_PROJECT_ROOT / "ridm_ultra" / "data" / "knowledge_base.txt"), "r", encoding="utf-8") as f:
                docs.append(f.read())

            # Initialize UMA for Geometric Translation
            from ridm_ultra.llm.uma import UniversalManifoldAligner
            uma = UniversalManifoldAligner()
            try:
                uma.rotation_matrices["tr"] = np.load("artifacts/uma_tr.npy")
            except (FileNotFoundError, OSError) as e:
                logger.warning(f"UMA rotation matrix not found, using identity: {e}")
                uma.rotation_matrices["tr"] = np.eye(word_emb.shape[1])
            
            rag = HybridRetriever(ridm, uma_aligner=uma, lang_code="tr")
            chunker = SemanticChunker(window_size=40, overlap=15)
            chunks = []
            for i, doc in enumerate(docs):
                chunks.extend(chunker.chunk_document(doc, doc_id=str(i)))
            rag.add_chunks(chunks)
            with _rag_lock:
                _rag_engine = rag
            logger.info("[*] Background RAG Engine loading complete!")
    except Exception as e:
        logger.error(f"[!] RAG Engine Load Error: {e}")
        with _rag_lock:
            _rag_engine = None

_rag_load_task: Optional[asyncio.Task] = None
_indexer_task: Optional[asyncio.Task] = None

async def _autonomous_indexer_loop():
    import os
    import asyncio
    from pathlib import Path
    
    indexed_files = set()
    data_dir = _PROJECT_ROOT / "ridm_ultra" / "data"
    
    from ridm_ultra.rag import SemanticChunker
    chunker = SemanticChunker(window_size=40, overlap=15)
    
    while True:
        try:
            if _rag_engine is not None and data_dir.exists():
                for file_path in data_dir.glob("*.txt"):
                    if str(file_path) not in indexed_files:
                        logger.info(f"[*] Autonomous Indexing: Found new file {file_path.name}")
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        chunks = chunker.chunk_document(content, doc_id=file_path.name)
                        
                        # Assuming _rag_engine is thread-safe for adding chunks,
                        # but in python GIL protects basic appends mostly.
                        _rag_engine.add_chunks(chunks)
                        indexed_files.add(str(file_path))
                        logger.info(f"[*] Autonomous Indexing: Indexed {len(chunks)} chunks from {file_path.name}")
        except Exception as e:
            logger.error(f"[!] Autonomous Indexing Error: {e}")
            
        await asyncio.sleep(60)  # Check every 60 seconds

async def get_chat_engine() -> ChatEngine:
    global _chat_engine, _rag_load_task, _rag_engine, _indexer_task

    if _rag_load_task is None and _rag_engine is None:
        logger.info("[*] Starting Background RAG Engine Load...")
        loop = asyncio.get_running_loop()
        _rag_load_task = loop.run_in_executor(None, _load_rag_engine_sync)
        
    if _indexer_task is None:
        loop = asyncio.get_running_loop()
        _indexer_task = loop.create_task(_autonomous_indexer_loop())

    if _chat_engine is None:
        repo = get_repository()
        _chat_engine = ChatEngine(repository=repo, rag_engine=None)
    
    with _rag_lock:
        _chat_engine.rag_engine = _rag_engine
    return _chat_engine

