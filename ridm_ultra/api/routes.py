"""FastAPI routes for RIDM Ultra Chat API & SSE Streaming."""
from __future__ import annotations

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from ridm_ultra.chat import ChatEngine, ModelTier, SQLiteChatRepository

from .dependencies import get_chat_engine, get_repository
from .schemas import (
    ChatSessionSchema,
    ChatStreamRequest,
    CreateSessionRequest,
    HealthResponse,
    SessionListResponse,
)

router = APIRouter()


@router.get("/health")
async def health_check(chat_engine: ChatEngine = Depends(get_chat_engine)):
    return {
        "status": "ok",
        "is_loaded": chat_engine.rag_engine is not None,
        "backend_info": {
            "adapters": [a.model_name for a in chat_engine.factory.all_adapters().values()]
        }
    }


@router.post("/api/v1/chat/stream")
async def chat_stream_endpoint(
    req: ChatStreamRequest,
    chat_engine: ChatEngine = Depends(get_chat_engine),
):
    """Server-Sent Events (SSE) streaming endpoint."""
    forced_tier = None
    if req.forced_tier:
        try:
            forced_tier = ModelTier(req.forced_tier.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid forced_tier '{req.forced_tier}'. Must be one of fast, balanced, reasoning."
            )

    async def sse_event_generator() -> AsyncGenerator[str, None]:
        async for chunk in chat_engine.chat_stream(
            user_message=req.message,
            session_id=req.session_id,
            system_prompt=req.system_prompt,
            forced_tier=forced_tier,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        ):
            payload = {
                "delta": chunk.delta,
                "finish_reason": chunk.finish_reason,
                "model": chunk.model_name,
                "usage": chunk.usage.__dict__ if chunk.usage else None,
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/v1/sessions", response_model=SessionListResponse)
async def list_sessions(
    limit: int = 50,
    repository: SQLiteChatRepository = Depends(get_repository),
):
    sessions = await repository.list_sessions(limit=limit)
    schemas = [ChatSessionSchema.model_validate(s.to_dict()) for s in sessions]
    return SessionListResponse(sessions=schemas, total=len(schemas))


@router.get("/api/v1/sessions/{session_id}", response_model=ChatSessionSchema)
async def get_session(
    session_id: str,
    repository: SQLiteChatRepository = Depends(get_repository),
):
    session = await repository.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found."
        )
    return ChatSessionSchema.model_validate(session.to_dict())


@router.post("/api/v1/sessions/new", response_model=ChatSessionSchema)
async def create_session(
    req: CreateSessionRequest,
    chat_engine: ChatEngine = Depends(get_chat_engine),
):
    session = await chat_engine.get_or_create_session(system_prompt=req.system_prompt)
    if req.title:
        session.title = req.title
        await chat_engine.repository.save_session(session)
    return ChatSessionSchema.model_validate(session.to_dict())


@router.delete("/api/v1/sessions/{session_id}")
async def delete_session(
    session_id: str,
    repository: SQLiteChatRepository = Depends(get_repository),
):
    deleted = await repository.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found."
        )
    return {"status": "success", "deleted_session_id": session_id}
