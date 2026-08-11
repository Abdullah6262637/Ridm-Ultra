"""Pydantic schemas for RIDM Ultra FastAPI Web & Streaming API."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    message: str = Field(..., description="User prompt message")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    system_prompt: Optional[str] = Field(None, description="System prompt override")
    forced_tier: Optional[str] = Field(None, description="Forced ModelTier (fast/balanced/reasoning)")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=8192)


class CreateSessionRequest(BaseModel):
    title: Optional[str] = Field("New Conversation", description="Session title")
    system_prompt: Optional[str] = Field(None, description="Initial system prompt")


class ChatMessageSchema(BaseModel):
    message_id: str
    role: str
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float


class ChatSessionSchema(BaseModel):
    session_id: str
    title: str
    messages: List[ChatMessageSchema]
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float
    updated_at: float


class SessionListResponse(BaseModel):
    sessions: List[ChatSessionSchema]
    total: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "6.0.0"
    timestamp: float = Field(default_factory=time.time)
    backend_info: Dict[str, Any] = Field(default_factory=dict)
