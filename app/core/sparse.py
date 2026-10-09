import re
import uuid
from typing import Protocol

from rank_bm25 import BM25Okapi
from sqlalchemy import select

from app.core.vectorstore import ScoredChunk, row_to_scored_chunk
from app.db import SessionLocal
from app.models.tables import Chunk as ChunkRow
from app.models.tables import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseSearch(Protocol):
    def search(self, query: str, k: int, filters: dict | None = None) -> list[ScoredChunk]: ...


class BM25SparseSearch:
    """MVP sparse retrieval: BM25 over chunks loaded per query."""

    def search(self, query: str, k: int, filters: dict | None = None) -> list[ScoredChunk]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        rows = self._load_rows(filters)
        if not rows:
            return []

        corpus = [tokenize(row.text) for row, _title in rows]
        if not any(corpus):
            return []

        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(
            ((item, float(score)) for item, score in zip(rows, scores, strict=True) if score > 0.0),
            key=lambda item: item[1],
            reverse=True,
        )[:k]

        return [
            row_to_scored_chunk(row, score=score, doc_title=title)
            for (row, title), score in ranked
        ]

    def _load_rows(self, filters: dict | None) -> list[tuple[ChunkRow, str | None]]:
        stmt = select(ChunkRow, Document.title)
        stmt = stmt.join(Document, Document.id == ChunkRow.doc_id, isouter=True)
        if filters and filters.get("doc_ids"):
            doc_ids = [uuid.UUID(str(doc_id)) for doc_id in filters["doc_ids"]]
            stmt = stmt.where(ChunkRow.doc_id.in_(doc_ids))
        with SessionLocal() as db:
            return list(db.execute(stmt).all())
