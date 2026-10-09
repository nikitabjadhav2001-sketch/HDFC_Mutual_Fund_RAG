"""Phase 6 tests: auth, rate limiting, observability, caching, upload limits."""

import json
import logging

import pytest
import redis
from fastapi.testclient import TestClient

from app.api import auth as auth_module
from app.api import routes_admin as routes_admin_module
from app.config import Settings, get_settings
from app.core import cache as cache_module
from app.core import ratelimit as ratelimit_module
from app.core.llm_provider import PUBLIC_ERROR_MESSAGE, LLMError
from app.core.logging import JsonFormatter
from app.main import app

client = TestClient(app)


def _db_ok() -> bool:
    from sqlalchemy import create_engine, text

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


def _patch_settings(monkeypatch, **overrides) -> Settings:
    base = get_settings()
    settings = Settings(
        database_url=base.database_url,
        redis_url=base.redis_url,
        storage_dir=base.storage_dir,
        openai_api_key=base.openai_api_key,
        openai_base_url=base.openai_base_url,
        **overrides,
    )
    monkeypatch.setattr(auth_module, "get_settings", lambda: settings)
    monkeypatch.setattr(routes_admin_module, "get_settings", lambda: settings)
    monkeypatch.setattr(ratelimit_module, "get_settings", lambda: settings)
    return settings


class TestAuth:
    @pytest.fixture(autouse=True)
    def keyed(self, monkeypatch):
        _patch_settings(monkeypatch, api_key="secret-key", admin_key="")

    def test_chat_rejects_missing_key(self):
        assert client.post("/api/v1/chat", json={"message": "hi"}).status_code == 401

    def test_chat_rejects_wrong_key(self):
        response = client.post(
            "/api/v1/chat",
            json={"message": "hi"},
            headers={"Authorization": "Bearer nope"},
        )
        assert response.status_code == 401

    @requires_db
    def test_chat_accepts_valid_key(self, monkeypatch):
        async def _mock(query, history=None, k=5, retrieval=None, use_cache=True,
                        conversation_id=None):
            yield {"type": "done", "retrieved_chunk_ids": []}

        monkeypatch.setattr("app.api.routes_chat.orchestrator.stream_response", _mock)
        response = client.post(
            "/api/v1/chat",
            json={"message": "hi"},
            headers={"Authorization": "Bearer secret-key"},
        )
        assert response.status_code == 200

    def test_admin_rejects_missing_key(self):
        assert client.get("/api/v1/admin/documents").status_code == 401

    @requires_db
    def test_admin_accepts_api_key_as_fallback(self):
        response = client.get(
            "/api/v1/admin/documents", headers={"Authorization": "Bearer secret-key"}
        )
        assert response.status_code == 200


class TestRateLimit:
    def test_limited_after_budget_exhausted(self, monkeypatch):
        _patch_settings(monkeypatch, rate_limit_per_minute=1, rate_limit_burst=1)
        first = client.get("/api/v1/conversations")
        second = client.get("/api/v1/conversations")
        assert first.status_code == 200
        assert second.status_code == 429
        assert "retry-after" in second.headers

    def test_disabled_with_zero_limit(self, monkeypatch):
        _patch_settings(monkeypatch, rate_limit_per_minute=0)
        assert client.get("/api/v1/conversations").status_code == 200

    def test_allow_request_fails_open_when_disabled(self, monkeypatch):
        _patch_settings(monkeypatch, rate_limit_per_minute=0)
        assert ratelimit_module.allow_request("k", capacity=1, refill_per_sec=1) is True


class TestObservability:
    def test_request_includes_trace_id_header(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.headers.get("x-request-id")

    def test_metrics_endpoint_exposes_http_metrics(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "http_requests_total" in response.text
        assert "http_request_duration_seconds" in response.text

    def test_json_formatter_emits_structured_record(self):
        record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "hello", (), None)
        payload = json.loads(JsonFormatter().format(record))
        assert payload["level"] == "INFO"
        assert payload["logger"] == "app.test"
        assert payload["message"] == "hello"
        assert payload["ts"]


class TestQueryCache:
    def test_normalize_query_collapses_whitespace_and_case(self):
        assert cache_module.normalize_query("  What   is RAG? ") == "what is rag?"

    def test_cache_fail_open_without_redis(self, monkeypatch):
        def _down(*args, **kwargs):
            raise redis.RedisError("no redis")

        monkeypatch.setattr(cache_module, "index_version", lambda: "v:1")
        monkeypatch.setattr(cache_module, "_client", _down)
        cached = cache_module.QueryCache("redis://none", ttl_seconds=60)
        assert cached.load("what is rag?", []) is None

    def test_cache_roundtrip_and_scoping(self, monkeypatch):
        storage: dict[str, str] = {}

        class _FakeRedis:
            def get(self, key):
                return storage.get(key)

            def setex(self, key, ttl, value):
                storage[key] = value

        monkeypatch.setattr(cache_module, "index_version", lambda: "v:1")
        monkeypatch.setattr(cache_module, "_client", lambda url: _FakeRedis())
        cached = cache_module.QueryCache("redis://fake", ttl_seconds=3600)

        assert cached.load(" What is RAG? ", []) is None
        cached.store(
            " What is RAG? ",
            [],
            {"answer": "A", "sources": [], "retrieved_chunk_ids": []},
        )

        assert cached.load("What is rag?", [])["answer"] == "A"
        with_history = cached.load("what is rag?", [{"role": "user", "content": "hi"}])
        assert with_history is None

    def test_cache_key_scopes_retrieval_memory(self, monkeypatch):
        storage: dict[str, str] = {}

        class _FakeRedis:
            def get(self, key):
                return storage.get(key)

            def setex(self, key, ttl, value):
                storage[key] = value

        monkeypatch.setattr(cache_module, "index_version", lambda: "v:1")
        monkeypatch.setattr(cache_module, "_client", lambda url: _FakeRedis())
        cached = cache_module.QueryCache("redis://fake", ttl_seconds=3600)

        cached.store("what is rag?", [], {"answer": "A"})
        assert cached.load("what is rag?", [])["answer"] == "A"
        # a conversation carrying retrieval memory must not reuse that entry
        assert cached.load("what is rag?", [], memory_ids=["chunk_1"]) is None

        cached.store("follow up?", [], {"answer": "B"}, memory_ids=["chunk_1"])
        assert cached.load("follow up?", [], memory_ids=["chunk_1"])["answer"] == "B"
        assert cached.load("follow up?", []) is None

    def test_disabled_cache_never_touches_redis(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("must not connect")

        monkeypatch.setattr(cache_module, "_client", _boom)
        cached = cache_module.QueryCache("redis://none", ttl_seconds=0)
        assert cached.enabled is False
        assert cached.load("what is rag?", []) is None
        cached.store("what is rag?", [], {"answer": "A"})


class TestUploadLimit:
    def test_upload_over_limit_returns_413(self, monkeypatch):
        _patch_settings(monkeypatch, api_key="sk", admin_key="sk", max_upload_bytes=64)
        response = client.post(
            "/api/v1/admin/documents",
            files={"file": ("big.txt", b"x" * 100)},
            headers={"Authorization": "Bearer sk"},
        )
        assert response.status_code == 413
        assert "limit" in response.json()["detail"]


class TestErrorMapping:
    @requires_db
    def test_llm_error_maps_to_public_message(self, monkeypatch):
        async def _boom(query, history=None, k=5, retrieval=None, use_cache=True,
                        conversation_id=None):
            yield {"type": "token", "content": "partial"}
            raise LLMError("secret provider detail")

        monkeypatch.setattr("app.api.routes_chat.orchestrator.stream_response", _boom)
        response = client.post("/api/v1/chat", json={"message": "hi"})
        events = [
            json.loads(line[len("data: "):])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        errors = [event for event in events if event["type"] == "error"]
        assert errors, f"expected an error event, got {events}"
        assert errors[0]["message"] == PUBLIC_ERROR_MESSAGE
        assert "secret provider detail" not in response.text
        assert errors[0]["trace_id"]
