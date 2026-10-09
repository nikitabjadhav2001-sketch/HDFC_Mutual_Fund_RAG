import asyncio
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.config import get_settings
from app.core.embedding import EmbeddingProvider, build_embedder
from app.core.metrics import RETRIEVAL_DURATION
from app.core.sparse import BM25SparseSearch, SparseSearch
from app.core.telemetry import start_child_span
from app.core.vectorstore import PgVectorStore, ScoredChunk, VectorStore

logger = logging.getLogger(__name__)

RRF_K = 60


@dataclass
class RetrievalResult:
    """Fused retrieval output with assembled context and abstention gate."""

    query: str
    chunks: list[ScoredChunk]
    top_score: float
    is_sufficient: bool


class Retriever:
    """Hybrid retriever: dense (pgvector) + sparse (BM25) with RRF fusion."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        sparse_search: SparseSearch | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self.vector_store = vector_store or PgVectorStore()
        self.sparse_search = sparse_search or BM25SparseSearch()
        self.embedder = embedder or _default_embedder()
        self.settings = get_settings()

    async def retrieve(
        self,
        query: str,
        k: int = 8,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResult:
        """Embed the query, search dense + sparse in parallel, fuse, assemble context.

        Abstention gate: `is_sufficient=False` when the best dense score is below
        `score_threshold` (or non-positive when no threshold is configured).
        """
        if not query.strip():
            return RetrievalResult(query=query, chunks=[], top_score=0.0, is_sufficient=False)

        started = perf_counter()
        span = start_child_span(
            "retrieve",
            **{"db.system": "postgresql", "query": query, "k": k},
        )
        try:
            query_emb = await asyncio.to_thread(self.embedder.embed_texts, [query])

            dense, sparse = await asyncio.gather(
                asyncio.to_thread(
                    self.vector_store.similarity_search,
                    embedding=query_emb[0],
                    k=k,
                    filters=filters,
                ),
                asyncio.to_thread(self.sparse_search.search, query, k=k, filters=filters),
            )

            top_score = max((chunk.score for chunk in dense), default=0.0)
            fused = rrf_fuse(dense, sparse, k_rrf=RRF_K, top_k=k)
            context = self._assemble_context(fused)

            threshold = (
                score_threshold if score_threshold is not None else self.settings.score_threshold
            )
            is_sufficient = bool(context) and top_score > 0.0 and top_score >= threshold

            result = RetrievalResult(
                query=query,
                chunks=context,
                top_score=top_score,
                is_sufficient=is_sufficient,
            )
            span.set_attribute("chunks", len(context))
            span.set_attribute("is_sufficient", is_sufficient)
            return result
        finally:
            span.end()
            RETRIEVAL_DURATION.observe(perf_counter() - started)

    def _assemble_context(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """Expand with ±1 ordinal neighbours, dedupe, truncate to token budget,
        and assign citation indices (1-based)."""
        if not chunks:
            return []

        neighbours = self._fetch_neighbours(chunks)
        fused_ids = {chunk.chunk_id for chunk in chunks}

        expanded: list[ScoredChunk] = []
        seen: set[str] = set()
        for parent in chunks:
            for candidate in self._with_neighbours(parent, neighbours):
                if candidate.chunk_id in seen:
                    continue
                if candidate.chunk_id not in fused_ids:
                    candidate.score = parent.score
                seen.add(candidate.chunk_id)
                expanded.append(candidate)

        budget = self.settings.context_token_budget
        context: list[ScoredChunk] = []
        used_tokens = 0
        for chunk in expanded:
            tokens = _token_count(chunk)
            if context and used_tokens + tokens > budget:
                break
            context.append(chunk)
            used_tokens += tokens

        for index, chunk in enumerate(context, start=1):
            chunk.citation_index = index

        return context

    def _fetch_neighbours(self, chunks: list[ScoredChunk]) -> dict[tuple[str, int], ScoredChunk]:
        wanted: dict[str, set[int]] = {}
        for chunk in chunks:
            ordinal = chunk.metadata.get("ordinal")
            if ordinal is None:
                continue
            wanted.setdefault(chunk.doc_id, set()).update({ordinal - 1, ordinal + 1})

        neighbours: dict[tuple[str, int], ScoredChunk] = {}
        for chunk in self.vector_store.get_chunks_by_ordinal(wanted):
            ordinal = chunk.metadata.get("ordinal")
            if ordinal is not None:
                neighbours[(chunk.doc_id, ordinal)] = chunk
        return neighbours

    @staticmethod
    def _with_neighbours(
        parent: ScoredChunk,
        neighbours: dict[tuple[str, int], ScoredChunk],
    ) -> list[ScoredChunk]:
        ordered = [parent]
        ordinal = parent.metadata.get("ordinal")
        if ordinal is not None:
            for candidate_ordinal in (ordinal - 1, ordinal + 1):
                neighbour = neighbours.get((parent.doc_id, candidate_ordinal))
                if neighbour is not None and neighbour.chunk_id != parent.chunk_id:
                    ordered.append(neighbour)
        return ordered


def _token_count(chunk: ScoredChunk) -> int:
    count = chunk.metadata.get("token_count")
    if not count:
        count = max(1, len(chunk.text) // 4)
    return int(count)


def _default_embedder() -> EmbeddingProvider:
    return build_embedder()


def rrf_fuse(
    dense_results: list[ScoredChunk],
    sparse_results: list[ScoredChunk],
    k_rrf: int = 60,
    top_k: int = 8,
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion (RRF) combining dense and sparse rankings.

    Produces new ScoredChunk objects so input lists keep their raw scores.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, ScoredChunk] = {}

    # Accumulate RRF score for dense rankings
    for rank, item in enumerate(dense_results, start=1):
        chunk_map[item.chunk_id] = item
        rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + (1.0 / (k_rrf + rank))

    # Accumulate RRF score for sparse rankings
    for rank, item in enumerate(sparse_results, start=1):
        chunk_map[item.chunk_id] = item
        rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + (1.0 / (k_rrf + rank))

    # Sort candidates by combined RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    fused_chunks: list[ScoredChunk] = []
    for cid in sorted_ids:
        source = chunk_map[cid]
        fused_chunks.append(
            ScoredChunk(
                chunk_id=source.chunk_id,
                doc_id=source.doc_id,
                text=source.text,
                score=rrf_scores[cid],
                metadata=dict(source.metadata),
                citation_index=source.citation_index,
            )
        )

    return fused_chunks
