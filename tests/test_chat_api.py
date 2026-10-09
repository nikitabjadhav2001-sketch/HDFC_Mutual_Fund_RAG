import json
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from app.db import SessionLocal
from app.main import app
from app.models.tables import Conversation, Message

client = TestClient(app)


def _db_ok() -> bool:
    from sqlalchemy import create_engine, text

    from app.config import get_settings

    try:
        engine = create_engine(
            get_settings().database_url, connect_args={"connect_timeout": 2}
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(not _db_ok(), reason="PostgreSQL not reachable")


def _wipe_conversations() -> None:
    db = SessionLocal()
    try:
        db.execute(delete(Message))
        db.execute(delete(Conversation))
        db.commit()
    finally:
        db.close()


@pytest.fixture(autouse=True)
def clean_conversations():
    _wipe_conversations()
    yield
    _wipe_conversations()


def _events(response_text: str) -> list[dict]:
    events = []
    for line in response_text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


async def _mock_stream(query, history=None, k=5, conversation_id=None):
    yield {"type": "token", "content": "Hello "}
    yield {"type": "token", "content": "world!"}
    yield {"type": "sources", "sources": []}
    yield {"type": "done", "retrieved_chunk_ids": []}


@requires_db
def test_chat_endpoint_streams_events():
    with patch("app.api.routes_chat.orchestrator.stream_response", side_effect=_mock_stream):
        response = client.post(
            "/api/v1/chat",
            json={"message": "Hi", "conversation_id": None},
        )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "Hello " in response.text
    assert "world!" in response.text


@requires_db
def test_chat_persists_messages_and_returns_conversation_id():
    with patch("app.api.routes_chat.orchestrator.stream_response", side_effect=_mock_stream):
        response = client.post(
            "/api/v1/chat",
            json={"message": "What is the exit load?", "conversation_id": None},
        )

    done = [event for event in _events(response.text) if event["type"] == "done"][0]
    conversation_id = done["conversation_id"]

    detail = client.get(f"/api/v1/conversations/{conversation_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["title"] == "What is the exit load?"
    assert [message["role"] for message in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["content"] == "What is the exit load?"
    assert body["messages"][1]["content"] == "Hello world!"


@requires_db
def test_chat_followup_loads_history_from_conversation():
    with patch("app.api.routes_chat.orchestrator.stream_response", side_effect=_mock_stream):
        first = client.post("/api/v1/chat", json={"message": "First question"})
    conversation_id = [
        event for event in _events(first.text) if event["type"] == "done"
    ][0]["conversation_id"]

    captured = {}

    async def spy(query, history=None, k=5, conversation_id=None):
        captured["history"] = history
        captured["conversation_id"] = conversation_id
        yield {"type": "token", "content": "ok"}
        yield {"type": "sources", "sources": []}
        yield {"type": "done", "retrieved_chunk_ids": []}

    with patch("app.api.routes_chat.orchestrator.stream_response", side_effect=spy):
        second = client.post(
            "/api/v1/chat",
            json={"message": "Follow-up", "conversation_id": conversation_id},
        )
    assert second.status_code == 200
    assert captured["history"] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "Hello world!"},
    ]


@requires_db
def test_chat_unknown_conversation_returns_404():
    response = client.post(
        "/api/v1/chat",
        json={"message": "hi", "conversation_id": "00000000-0000-0000-0000-000000000000"},
    )
    assert response.status_code == 404


@requires_db
def test_conversations_list_get_delete():
    with patch("app.api.routes_chat.orchestrator.stream_response", side_effect=_mock_stream):
        chat = client.post("/api/v1/chat", json={"message": "List me"})
    conversation_id = [
        event for event in _events(chat.text) if event["type"] == "done"
    ][0]["conversation_id"]

    listing = client.get("/api/v1/conversations")
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()] == [conversation_id]

    detail = client.get(f"/api/v1/conversations/{conversation_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == conversation_id

    deleted = client.delete(f"/api/v1/conversations/{conversation_id}")
    assert deleted.status_code == 204

    assert client.get(f"/api/v1/conversations/{conversation_id}").status_code == 404
    assert client.get("/api/v1/conversations").json() == []
    assert client.delete(f"/api/v1/conversations/{conversation_id}").status_code == 404


@requires_db
def test_delete_conversation_removes_messages():
    with patch("app.api.routes_chat.orchestrator.stream_response", side_effect=_mock_stream):
        chat = client.post("/api/v1/chat", json={"message": "Cascade check"})
    conversation_id = [
        event for event in _events(chat.text) if event["type"] == "done"
    ][0]["conversation_id"]

    client.delete(f"/api/v1/conversations/{conversation_id}")

    db = SessionLocal()
    try:
        count_stmt = (
            select(func.count())
            .select_from(Message)
            .where(Message.conv_id == uuid.UUID(conversation_id))
        )
        remaining = db.execute(count_stmt).scalar_one()
    finally:
        db.close()
    assert remaining == 0
