"""Session Storage and Persistence Repositories for RIDM Ultra Chat Engine."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from .interfaces import BaseChatRepository
from .types import ChatSession


class InMemoryChatRepository(BaseChatRepository):
    """Thread-safe, volatile in-memory chat session repository."""

    def __init__(self):
        self._sessions: Dict[str, ChatSession] = {}
        self._lock = asyncio.Lock()

    async def save_session(self, session: ChatSession) -> None:
        async with self._lock:
            self._sessions[session.session_id] = session

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def list_sessions(self, limit: int = 50) -> List[ChatSession]:
        async with self._lock:
            sorted_sessions = sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)
            return sorted_sessions[:limit]

    async def delete_session(self, session_id: str) -> bool:
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False


class SQLiteChatRepository(BaseChatRepository):
    """Zero-dependency disk-based SQLite chat session repository."""

    def __init__(self, db_path: str | Path = "artifacts/chat_sessions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    data TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    async def save_session(self, session: ChatSession) -> None:
        loop = asyncio.get_running_loop()
        def _write():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO chat_sessions (session_id, title, data, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        title = excluded.title,
                        data = excluded.data,
                        updated_at = excluded.updated_at
                    """,
                    (session.session_id, session.title, json.dumps(session.to_dict()), session.updated_at)
                )
                conn.commit()
        await loop.run_in_executor(None, _write)

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        loop = asyncio.get_running_loop()
        def _read():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data FROM chat_sessions WHERE session_id = ?", (session_id,)
                )
                row = cursor.fetchone()
                if row:
                    return ChatSession.from_dict(json.loads(row[0]))
                return None
        return await loop.run_in_executor(None, _read)

    async def list_sessions(self, limit: int = 50) -> List[ChatSession]:
        loop = asyncio.get_running_loop()
        def _list():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data FROM chat_sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
                )
                return [ChatSession.from_dict(json.loads(r[0])) for r in cursor.fetchall()]
        return await loop.run_in_executor(None, _list)

    async def delete_session(self, session_id: str) -> bool:
        loop = asyncio.get_running_loop()
        def _del():
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                return cursor.rowcount > 0
        return await loop.run_in_executor(None, _del)
