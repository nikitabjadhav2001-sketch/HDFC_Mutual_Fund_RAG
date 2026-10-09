"""Exact-query answer cache (Phase 6).

Keys combine the normalized question + digest of the in-context history + digest
of the retrieval-memory chunk ids (when a conversation carries prior-turn
retrievals) with a short TTL and the current index version (chunk count + last
insert), so any ingestion or deletion invalidates cached answers. Redis outages
fail open.
"""

from __future__ import annotations

import hashlib
import json
import logging

import redis
from sqlalchemy import text

from app.config import get_settings
from app.db import SessionLocal

logger = logging.getLogger(__name__)

_CLIENTS: dict[str, redis.Redis] = {}


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def index_version() -> str:
    """Cheap fingerprint of the chunk index; changes on any insert/delete."""
    with SessionLocal() as db:
        row = db.execute(
            text(
                "SELECT count(*) AS n, coalesce(max(created_at), now()) AS ts FROM chunks"
            )
        ).one()
    return f"{row.n}:{int(row.ts.timestamp() * 1_000_000)}"


def _client(url: str) -> redis.Redis:
    client = _CLIENTS.get(url)
    if client is None:
        client = redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
        _CLIENTS[url] = client
    return client


class QueryCache:
    def __init__(self, redis_url: str, ttl_seconds: int) -> None:
        self.redis_url = redis_url
        self.ttl = ttl_seconds

    @property
    def enabled(self) -> bool:
        return self.ttl > 0

    def _key(
        self,
        version: str,
        query: str,
        history: list[dict],
        memory_ids: list[str] | None = None,
    ) -> str:
        history_json = json.dumps(history, sort_keys=True, ensure_ascii=False)
        key = (
            f"rag:answer:{version}:{_digest(normalize_query(query))}:"
            f"{_digest(history_json)}"
        )
        if memory_ids:
            key += f":{_digest(','.join(memory_ids))}"
        return key

    def load(
        self,
        query: str,
        history: list[dict],
        memory_ids: list[str] | None = None,
    ) -> dict | None:
        if not self.enabled:
            return None
        try:
            version = index_version()
            raw = _client(self.redis_url).get(self._key(version, query, history, memory_ids))
        except redis.RedisError as exc:
            logger.warning("query cache unavailable, failing open: %s", exc)
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def store(
        self,
        query: str,
        history: list[dict],
        payload: dict,
        memory_ids: list[str] | None = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            version = index_version()
            _client(self.redis_url).setex(
                self._key(version, query, history, memory_ids),
                self.ttl,
                json.dumps(payload, ensure_ascii=False),
            )
        except redis.RedisError as exc:
            logger.warning("query cache write failed: %s", exc)


def get_query_cache() -> QueryCache:
    settings = get_settings()
    return QueryCache(settings.redis_url, settings.query_cache_ttl)
