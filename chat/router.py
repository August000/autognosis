"""FastAPI router for chat endpoints."""

from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chat.chat import ChatService, SUPPORTED_MODELS
from db import chat_repo

router = APIRouter(prefix="/api/chat", tags=["chat"])


class CreateConversationRequest(BaseModel):
    model: str = "gpt-4o-mini"


class SendMessageRequest(BaseModel):
    content: str
    model: str | None = None


@router.get("/models")
async def list_models():
    """Return the list of supported chat models."""
    return {"models": SUPPORTED_MODELS}


@router.post("/conversations")
async def create_conversation(request: Request, body: CreateConversationRequest):
    """Create a new conversation."""
    pool = request.app.state.pool
    conv = await chat_repo.create_conversation(pool, model=body.model)
    return conv


@router.get("/conversations")
async def list_conversations(request: Request, limit: int = 50):
    """List all conversations."""
    pool = request.app.state.pool
    convs = await chat_repo.list_conversations(pool, limit=limit)
    return {"conversations": convs}


@router.get("/conversations/{conversation_id}/messages")
async def get_messages(request: Request, conversation_id: UUID):
    """Get messages for a conversation."""
    pool = request.app.state.pool
    conv = await chat_repo.get_conversation(pool, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = await chat_repo.get_messages(pool, conversation_id)
    return {"conversation": conv, "messages": messages}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: UUID):
    """Delete a conversation and all its messages."""
    pool = request.app.state.pool
    deleted = await chat_repo.delete_conversation(pool, conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"success": True}


@router.post("/conversations/{conversation_id}/messages")
async def send_message(request: Request, conversation_id: UUID, body: SendMessageRequest):
    """Send a message and stream the assistant's response via SSE."""
    pool = request.app.state.pool

    conv = await chat_repo.get_conversation(pool, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    model = body.model or conv["model"]
    service = ChatService(pool)

    async def event_stream():
        async for chunk in service.send_message(conversation_id, body.content, model=model):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
