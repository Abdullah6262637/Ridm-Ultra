"""Abstract Base Classes & Component Contracts for RIDM Ultra Chat Engine."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, List, Optional

from .types import ChatMessage, ChatResponseChunk, ChatSession, ModelTier


class BaseModelAdapter(ABC):
    """Interface for LLM inference backends (Local PyTorch / Cloud APIs)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the identifier of the underlying model."""
        pass

    @property
    @abstractmethod
    def tier(self) -> ModelTier:
        """Return the performance/capability tier of this adapter."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[ChatResponseChunk, None]:
        """Asynchronously stream response chunks."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatMessage:
        """Asynchronously return a complete completion message."""
        pass


class BaseMemoryManager(ABC):
    """Interface for managing context windows, sliding windows, and background summarization."""

    @abstractmethod
    def get_context_messages(
        self,
        session: ChatSession,
        max_token_budget: int = 8192,
    ) -> List[ChatMessage]:
        """Assemble context messages fitting strictly within the token budget."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate or compute exact token count for given text."""
        pass

    @abstractmethod
    async def summarize_if_needed(self, session: ChatSession, max_tokens: int = 8192) -> Optional[str]:
        """Trigger background summarization if conversation length exceeds memory threshold."""
        pass


class BaseRouter(ABC):
    """Interface for dynamic query routing and expert selection."""

    @abstractmethod
    def route_query(
        self,
        query: str,
        context_messages: List[ChatMessage],
        available_adapters: Dict[ModelTier, BaseModelAdapter],
    ) -> BaseModelAdapter:
        """Determine which model adapter should handle the request."""
        pass


class BaseChatRepository(ABC):
    """Interface for chat session storage and persistence."""

    @abstractmethod
    async def save_session(self, session: ChatSession) -> None:
        """Save or update a chat session."""
        pass

    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Retrieve a session by ID."""
        pass

    @abstractmethod
    async def list_sessions(self, limit: int = 50) -> List[ChatSession]:
        """List active/historical sessions."""
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        pass
