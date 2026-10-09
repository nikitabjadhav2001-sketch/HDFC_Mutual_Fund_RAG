import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import and_, or_, select

from app.db import SessionLocal
from app.models.tables import Chunk as ChunkRow
from app.models.tables import Document


@dataclass
class ScoredChunk:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    citation_index: int | None = None


def row_to_scored_chunk(
    row: ChunkRow, score: float = 0.0, doc_title: str | None = None
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=str(row.id),
        doc_id=str(row.doc_id),
        text=row.text,
        score=score,
        metadata={
            "section_path": row.section_path,
            "page_start": row.page_start,
            "page_end": row.page_end,
            "token_count": row.token_count,
            "ordinal": row.ordinal,
            "content_hash": row.content_hash,
            "document_title": doc_title,
        },
    )


class VectorStore(Protocol):
    def similarity_search(
        self, embedding: list[float], k: int, filters: dict | None = None
    ) -> list[ScoredChunk]: ...

    def get_chunks_by_ordinal(self, wanted: dict[str, set[int]]) -> list[ScoredChunk]: ...


class PgVectorStore:
    """pgvector similarity search implementation."""

    def similarity_search(
        self, embedding: list[float], k: int, filters: dict | None = None
    ) -> list[ScoredChunk]:
        with SessionLocal() as db:
            stmt = (
                select(
                    ChunkRow,
                    Document.title.label("doc_title"),
                    ChunkRow.embedding.cosine_distance(embedding).label("distance"),
                )
                .join(Document, Document.id == ChunkRow.doc_id, isouter=True)
                .order_by("distance")
                .limit(k)
            )

            if filters and "doc_ids" in filters and filters["doc_ids"]:
                stmt = stmt.where(ChunkRow.doc_id.in_(filters["doc_ids"]))

            results = db.execute(stmt).all()

            scored_chunks = []
            for row, doc_title, distance in results:
                # Cosine similarity = 1 - cosine distance
                score = 1.0 - float(distance)
                scored_chunks.append(row_to_scored_chunk(row, score=score, doc_title=doc_title))

            return scored_chunks

    def get_chunks_by_ordinal(self, wanted: dict[str, set[int]]) -> list[ScoredChunk]:
        """Fetch specific chunks by (doc_id, ordinal) pairs for neighbour expansion."""
        if not wanted:
            return []

        conditions = [
            and_(
                ChunkRow.doc_id == uuid.UUID(doc_id),
                ChunkRow.ordinal.in_(ordinals),
            )
            for doc_id, ordinals in wanted.items()
            if ordinals
        ]
        if not conditions:
            return []

        with SessionLocal() as db:
            rows = (
                db.execute(
                    select(ChunkRow, Document.title)
                    .join(Document, Document.id == ChunkRow.doc_id, isouter=True)
                    .where(or_(*conditions))
                )
                .all()
            )

        return [row_to_scored_chunk(row, doc_title=title) for row, title in rows]
