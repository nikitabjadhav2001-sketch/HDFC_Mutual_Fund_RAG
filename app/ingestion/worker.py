import asyncio
import uuid

from arq.connections import RedisSettings

from app.config import get_settings
from app.db import SessionLocal
from app.ingestion.pipeline import ingest


async def ingest_job(ctx: dict, document_id: str) -> None:
    session = SessionLocal()
    try:
        await asyncio.to_thread(ingest, uuid.UUID(document_id), session)
    finally:
        session.close()


class WorkerSettings:
    functions = [ingest_job]
    job_timeout = 600
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
