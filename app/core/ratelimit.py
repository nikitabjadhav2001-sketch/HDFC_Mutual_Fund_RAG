"""Redis token-bucket rate limiting (Phase 6).

Buckets are keyed per client credential (hashed API key, else IP). Redis
outages fail open — a cache miss must never take the API down.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time

import redis
from fastapi import HTTPException, Request

from app.config import get_settings

logger = logging.getLogger(__name__)

# Atomic token bucket: refill by elapsed time, allow when a token is available.
LUA_TOKEN_BUCKET = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_per_sec = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local bucket = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(bucket[1])
local ts = tonumber(bucket[2])
if tokens == nil then tokens = capacity; ts = now end
local elapsed = math.max(0, now - ts)
tokens = math.min(capacity, tokens + elapsed * refill_per_sec)
local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end
redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
local ttl = 120
if refill_per_sec > 0 then ttl = math.ceil(capacity / refill_per_sec) + 60 end
redis.call('EXPIRE', key, ttl)
return allowed
"""

_CLIENTS: dict[str, redis.Redis] = {}


def _client(url: str) -> redis.Redis:
    client = _CLIENTS.get(url)
    if client is None:
        client = redis.Redis.from_url(url, socket_connect_timeout=1, socket_timeout=1)
        _CLIENTS[url] = client
    return client


def allow_request(key: str, *, capacity: float, refill_per_sec: float, cost: float = 1.0) -> bool:
    """True when the bucket has budget for `cost`; fails open on Redis errors."""
    if capacity <= 0 or refill_per_sec <= 0:
        return True
    try:
        allowed = _client(get_settings().redis_url).eval(
            LUA_TOKEN_BUCKET,
            1,
            key,
            capacity,
            refill_per_sec,
            time.time(),
            cost,
        )
        return bool(allowed)
    except redis.RedisError as exc:
        logger.warning("rate limiter unavailable, failing open: %s", exc)
        return True


def identity_of(request: Request) -> str:
    """Stable per-client bucket key without storing raw credentials."""
    authorization = request.headers.get("authorization") or ""
    if authorization:
        material = authorization
    elif request.client:
        material = request.client.host
    else:
        material = "anon"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


async def rate_limit(request: Request) -> None:
    settings = get_settings()
    if settings.rate_limit_per_minute <= 0:
        return
    allowed = await asyncio.to_thread(
        allow_request,
        identity_of(request),
        capacity=float(settings.rate_limit_burst),
        refill_per_sec=settings.rate_limit_per_minute / 60.0,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded, slow down",
            headers={"Retry-After": "1"},
        )


def reset_clients() -> None:
    """Test hook: drop cached Redis clients after changing REDIS_URL."""
    _CLIENTS.clear()
