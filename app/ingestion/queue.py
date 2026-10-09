from arq import create_pool
from arq.connections import RedisSettings

from app.config import get_settings

JOB_NAME = "ingest_job"


async def enqueue_ingest(document_id: str) -> None:
    """Put an `ingest` job on the Redis queue; raises if Redis is unreachable."""
    settings = get_settings()
    pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    try:
        await pool.enqueue_job(JOB_NAME, document_id)
    finally:
        await pool.aclose()
