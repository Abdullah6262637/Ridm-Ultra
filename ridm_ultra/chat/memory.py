"""Hierarchical Context Window & Memory Management for RIDM Ultra Chat Engine."""
from __future__ import annotations

import math
from typing import List, Optional

from .interfaces import BaseMemoryManager
from .types import ChatMessage, ChatSession, MessageRole
CHARS_PER_TOKEN_ESTIMATE = 4.0


class HierarchicalMemoryManager(BaseMemoryManager):
    """Context window manager featuring sliding token window and background summarization."""

    def __init__(self, target_max_tokens: int = 8192, summarization_threshold_ratio: float = 0.8):
        self.target_max_tokens = target_max_tokens
        self.summarization_threshold_ratio = summarization_threshold_ratio

    def count_tokens(self, text: str) -> int:
        """Fast heuristic token counter (~4 characters per token)."""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE))

    def _message_tokens(self, msg: ChatMessage) -> int:
        return self.count_tokens(msg.content) + 4  # role overhead

    def get_context_messages(
        self,
        session: ChatSession,
        max_token_budget: int = 8192,
    ) -> List[ChatMessage]:
        """Assembles context messages guaranteeing compliance with max_token_budget."""
        result: List[ChatMessage] = []
        accumulated_tokens = 0

        # System prompt always gets top priority if present
        system_msg: Optional[ChatMessage] = None
        if session.system_prompt:
            system_msg = ChatMessage(role=MessageRole.SYSTEM, content=session.system_prompt)
            accumulated_tokens += self._message_tokens(system_msg)

        # Include summary message if present in session metadata
        summary_msg: Optional[ChatMessage] = None
        if "summary" in session.metadata and session.metadata["summary"]:
            summary_content = f"[Conversation Summary]: {session.metadata['summary']}"
            summary_msg = ChatMessage(role=MessageRole.SYSTEM, content=summary_content)
            accumulated_tokens += self._message_tokens(summary_msg)

        # Iterate backwards through recent messages (Sliding Window FIFO)
        recent_messages: List[ChatMessage] = []
        for msg in reversed(session.messages):
            t_cost = self._message_tokens(msg)
            if accumulated_tokens + t_cost > max_token_budget:
                break
            recent_messages.append(msg)
            accumulated_tokens += t_cost

        # Re-order back to chronological order
        recent_messages.reverse()

        # Construct final payload
        if system_msg:
            result.append(system_msg)
        if summary_msg:
            result.append(summary_msg)
        result.extend(recent_messages)

        return result

    async def summarize_if_needed(self, session: ChatSession, max_tokens: int = 8192) -> Optional[str]:
        """Triggers summarization when total token count exceeds threshold ratio."""
        total_tokens = sum(self._message_tokens(m) for m in session.messages)
        threshold = int(max_tokens * self.summarization_threshold_ratio)

        if total_tokens < threshold or len(session.messages) < 6:
            return None

        # Take older half of messages to form abstract summary
        cutoff_index = len(session.messages) // 2
        older_messages = session.messages[:cutoff_index]

        summary_topics = [m.content[:30] + "..." for m in older_messages if m.role == MessageRole.USER]
        new_summary = f"User discussed: {'; '.join(summary_topics)}"

        existing_summary = session.metadata.get("summary", "")
        updated_summary = f"{existing_summary} | {new_summary}".strip(" | ")

        session.metadata["summary"] = updated_summary
        # Retain only recent messages after summarization
        session.messages = session.messages[cutoff_index:]
        return updated_summary
