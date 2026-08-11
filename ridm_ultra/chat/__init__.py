"""RIDM Ultra Chat Subsystem — Production-Grade Conversational Engine."""
from .adapters import AdapterFactory, LocalTransformerAdapter
from .engine import ChatEngine
from .interfaces import BaseChatRepository, BaseMemoryManager, BaseModelAdapter, BaseRouter
from .memory import HierarchicalMemoryManager
from .repository import InMemoryChatRepository, SQLiteChatRepository
from .router import SemanticRouter
from .types import ChatMessage, ChatResponseChunk, ChatSession, MessageRole, ModelTier, TokenUsage

__all__ = [
    "ChatEngine",
    "ChatMessage",
    "ChatSession",
    "ChatResponseChunk",
    "MessageRole",
    "ModelTier",
    "TokenUsage",
    "BaseModelAdapter",
    "BaseMemoryManager",
    "BaseRouter",
    "BaseChatRepository",
    "LocalTransformerAdapter",
    "AdapterFactory",
    "HierarchicalMemoryManager",
    "SemanticRouter",
    "InMemoryChatRepository",
    "SQLiteChatRepository",
]
