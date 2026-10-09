import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.embedding import EmbeddingProvider, build_embedder
from app.ingestion.chunker import chunk_document
from app.ingestion.parsers import parse
from app.models.tables import Chunk as ChunkRow
from app.models.tables import Document

logger = logging.getLogger(__name__)


def ingest(
    document_id: uuid.UUID | str,
    session: Session,
    *,
    embedder: EmbeddingProvider | None = None,
) -> Document:
    """parse → chunk → embed → upsert, with status tracking.

    Returns the refreshed Document; failures are recorded on the row
    (`status="failed"`, `error`) instead of raising.
    """
    document_id = uuid.UUID(str(document_id))
    doc = session.get(Document, document_id)
    if doc is None:
        raise KeyError(f"document {document_id} not found")

    doc.status = "processing"
    doc.updated_at = datetime.now(UTC)
    session.commit()

    try:
        _process(doc, session, embedder)
    except Exception as exc:  # noqa: BLE001 - failure is recorded, not raised
        logger.exception("ingestion failed for document %s", document_id)
        session.rollback()
        doc = session.get(Document, document_id)
        doc.status = "failed"
        doc.error = str(exc)
        doc.updated_at = datetime.now(UTC)
        session.commit()
    return doc


def _process(doc: Document, session: Session, embedder: EmbeddingProvider | None) -> None:
    if not doc.source_uri:
        raise ValueError("document has no source file")
    source = Path(doc.source_uri)
    if not source.is_file():
        raise FileNotFoundError(f"source file missing: {source}")

    parsed = parse(source.read_bytes(), doc.format, filename=f"{doc.title}.{doc.format}")
    chunks = chunk_document(parsed)
    if not chunks:
        raise ValueError("no chunks produced from document")

    existing = set(
        session.scalars(
            select(ChunkRow.content_hash).where(ChunkRow.doc_id == doc.id)
        )
    )
    new_chunks = [chunk for chunk in chunks if chunk.content_hash not in existing]

    now = datetime.now(UTC)
    if new_chunks:
        embedder = embedder or _default_embedder()
        vectors = embedder.embed_texts([chunk.text for chunk in new_chunks])
        offset = len(existing)
        for index, (chunk, vector) in enumerate(zip(new_chunks, vectors, strict=True)):
            session.add(
                ChunkRow(
                    doc_id=doc.id,
                    text=chunk.text,
                    embedding=vector,
                    section_path=chunk.section_path,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    token_count=chunk.token_count,
                    ordinal=offset + index,
                    content_hash=chunk.content_hash,
                )
            )

    doc.title = parsed.title or doc.title
    doc.status = "done"
    doc.error = None
    doc.updated_at = now
    session.commit()
    logger.info(
        "document %s ingested: %d chunks (%d new)",
        doc.id,
        len(chunks),
        len(new_chunks),
    )


def _default_embedder() -> EmbeddingProvider:
    return build_embedder()
