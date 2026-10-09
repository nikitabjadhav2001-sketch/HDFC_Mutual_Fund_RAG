import json
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.auth import require_api_key
from app.config import get_settings
from app.core.llm_provider import LLMError
from app.core.ratelimit import rate_limit
from app.core.telemetry import get_trace_id
from app.db import get_db
from app.models.tables import Conversation, Message
from app.rag.orchestrator import GenerationOrchestrator
from app.schemas import ConversationDetailOut, ConversationOut, MessageOut

router = APIRouter(
    dependencies=[Depends(require_api_key), Depends(rate_limit)], tags=["chat"]
)
orchestrator = GenerationOrchestrator()
logger = logging.getLogger(__name__)

TITLE_MAX_LENGTH = 80


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    history: Optional[list[dict[str, str]]] = Field(default_factory=list)


@router.post("/chat")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    conversation, history = _resolve_conversation(db, request)
    _save_message(db, conversation.id, "user", request.message)

    started_at = time.perf_counter()
    sources: list = []
    retrieved_chunk_ids: list = []
    answer_parts: list[str] = []

    async def event_generator():
        nonlocal sources, retrieved_chunk_ids
        try:
            async for event in orchestrator.stream_response(
                request.message, history, conversation_id=str(conversation.id)
            ):
                if event["type"] == "token":
                    answer_parts.append(event.get("content", ""))
                elif event["type"] == "sources":
                    sources = event.get("sources", [])
                elif event["type"] == "done":
                    retrieved_chunk_ids = event.get("retrieved_chunk_ids", [])
                    event = {**event, "conversation_id": str(conversation.id)}
                yield f"data: {json.dumps(event)}\n\n"
        except LLMError as exc:
            logger.error("chat stream failed", extra={"event": "llm_error"})
            error_event = {
                "type": "error",
                "message": exc.public_message,
                "conversation_id": str(conversation.id),
                "trace_id": get_trace_id(),
            }
            yield f"data: {json.dumps(error_event)}\n\n"
        except Exception:  # noqa: BLE001 - surface stream failures to the client
            logger.exception("chat stream failed")
            error_event = {
                "type": "error",
                "message": "internal error while generating the answer",
                "conversation_id": str(conversation.id),
                "trace_id": get_trace_id(),
            }
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            if answer_parts:
                _save_assistant_message(db, conversation.id, "".join(answer_parts), sources,
                                        retrieved_chunk_ids, started_at)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(db: Session = Depends(get_db)) -> list[ConversationOut]:
    last_message_at = (
        select(func.max(Message.created_at))
        .where(Message.conv_id == Conversation.id)
        .correlate(Conversation)
        .scalar_subquery()
    )
    conversations = db.execute(
        select(Conversation).order_by(
            func.coalesce(last_message_at, Conversation.created_at).desc()
        )
    ).scalars().all()
    return [_conversation_out(conversation) for conversation in conversations]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailOut)
def get_conversation(
    conversation_id: uuid.UUID, db: Session = Depends(get_db)
) -> ConversationDetailOut:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    rows = db.execute(
        select(Message)
        .where(Message.conv_id == conversation.id)
        .order_by(Message.created_at)
    ).scalars().all()
    return ConversationDetailOut(
        **_conversation_out(conversation).model_dump(),
        messages=[
            MessageOut(
                id=str(row.id),
                role=row.role,
                content=row.content,
                citations=row.citations,
                created_at=row.created_at.isoformat() if row.created_at else None,
            )
            for row in rows
        ],
    )


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: uuid.UUID, db: Session = Depends(get_db)
) -> Response:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    orchestrator.forget(str(conversation.id))
    db.delete(conversation)
    db.commit()
    return Response(status_code=204)


def _resolve_conversation(
    db: Session, request: ChatRequest
) -> tuple[Conversation, list[dict[str, str]]]:
    """Load history for an existing conversation, or start a new one.

    When a conversation is loaded from the database it is the source of truth
    for history; the client-sent `history` field is ignored.
    """
    if not request.conversation_id:
        conversation = Conversation(title=_title_from(request.message))
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation, list(request.history or [])

    try:
        conversation_id = uuid.UUID(request.conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="conversation not found") from exc
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    rows = db.execute(
        select(Message)
        .where(Message.conv_id == conversation.id)
        .order_by(Message.created_at)
    ).scalars().all()
    history = [{"role": row.role, "content": row.content} for row in rows]
    return conversation, history


def _conversation_out(conversation: Conversation) -> ConversationOut:
    return ConversationOut(
        id=str(conversation.id),
        title=conversation.title,
        created_at=conversation.created_at.isoformat() if conversation.created_at else None,
    )


def _title_from(message: str) -> str:
    text = " ".join(message.split())
    if len(text) <= TITLE_MAX_LENGTH:
        return text
    return text[: TITLE_MAX_LENGTH - 1] + "…"


def _save_message(
    db: Session,
    conversation_id: uuid.UUID,
    role: str,
    content: str,
    *,
    citations: list | None = None,
    retrieved_chunk_ids: list | None = None,
    latency_ms: int | None = None,
) -> None:
    db.add(
        Message(
            conv_id=conversation_id,
            role=role,
            content=content,
            citations=citations,
            retrieved_chunk_ids=retrieved_chunk_ids,
            model=get_settings().llm_model if role == "assistant" else None,
            latency_ms=latency_ms,
        )
    )
    db.commit()


def _save_assistant_message(
    db: Session,
    conversation_id: uuid.UUID,
    content: str,
    sources: list,
    retrieved_chunk_ids: list,
    started_at: float,
) -> None:
    try:
        _save_message(
            db,
            conversation_id,
            "assistant",
            content,
            citations=sources,
            retrieved_chunk_ids=retrieved_chunk_ids,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
    except Exception:  # noqa: BLE001 - never break the response while persisting
        db.rollback()
        logger.exception("failed to persist assistant message")
