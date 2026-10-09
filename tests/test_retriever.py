import asyncio
import hashlib
import json
import math
import time
import uuid
import zlib
from pathlib import Path

import pytest
from sqlalchemy import delete

from app.config import get_settings
from app.core.sparse import tokenize
from app.core.vectorstore import ScoredChunk
from app.db import SessionLocal
from app.models.tables import EMBEDDING_DIM, Document
from app.models.tables import Chunk as ChunkRow
from app.rag.retriever import Retriever, rrf_fuse

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "retrieval_corpus.json"


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


def _load_corpus() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


# --- fakes (no DB needed) -----------------------------------------------------


def chunk(
    chunk_id: str,
    doc_id: str = "doc-1",
    score: float = 0.5,
    text: str = "chunk text",
    ordinal: int | None = None,
    tokens: int | None = None,
) -> ScoredChunk:
    metadata: dict = {}
    if ordinal is not None:
        metadata["ordinal"] = ordinal
    if tokens is not None:
        metadata["token_count"] = tokens
    return ScoredChunk(chunk_id=chunk_id, doc_id=doc_id, text=text, score=score, metadata=metadata)


class FakeStore:
    def __init__(self, dense: list[ScoredChunk], neighbours: dict | None = None) -> None:
        self.dense = dense
        self.neighbours = neighbours or {}
        self.search_calls: list[dict] = []
        self.neighbour_calls: list[dict] = []

    def similarity_search(self, embedding: list[float], k: int, filters: dict | None = None):
        self.search_calls.append({"embedding": embedding, "k": k, "filters": filters})
        return self.dense[:k]

    def get_chunks_by_ordinal(self, wanted: dict[str, set[int]]) -> list[ScoredChunk]:
        self.neighbour_calls.append(wanted)
        found: list[ScoredChunk] = []
        for doc_id, ordinals in wanted.items():
            for ordinal in ordinals:
                found.extend(self.neighbours.get((doc_id, ordinal), []))
        return found


class FakeSparse:
    def __init__(self, results: list[ScoredChunk]) -> None:
        self.results = results
        self.search_calls: list[dict] = []

    def search(self, query: str, k: int, filters: dict | None = None) -> list[ScoredChunk]:
        self.search_calls.append({"query": query, "k": k, "filters": filters})
        return self.results[:k]


class StaticEmbedder:
    model = "static"

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [list(self.vector) for _ in texts]


def make_retriever(
    dense: list[ScoredChunk],
    sparse: list[ScoredChunk],
    neighbours: dict | None = None,
    vector: list[float] | None = None,
) -> tuple[Retriever, FakeStore, FakeSparse, StaticEmbedder]:
    store = FakeStore(dense, neighbours)
    sparse_search = FakeSparse(sparse)
    embedder = StaticEmbedder(vector if vector is not None else [1.0])
    retriever = Retriever(
        vector_store=store, sparse_search=sparse_search, embedder=embedder
    )
    return retriever, store, sparse_search, embedder


# --- fusion -------------------------------------------------------------------


def test_rrf_fuse_beats_pure_dense_on_lexical_query() -> None:
    golden = chunk("golden", score=0.05)
    dense = [chunk(f"d{i}", score=0.9 - 0.01 * i) for i in range(7)] + [golden]
    sparse = [golden] + [chunk(f"s{i}") for i in range(1, 8)]

    assert dense.index(golden) == 7, "golden must rank last under pure dense"

    fused = rrf_fuse(dense, sparse, k_rrf=60, top_k=8)

    assert fused[0].chunk_id == "golden"
    assert [c.chunk_id for c in fused].index("golden") < [c.chunk_id for c in fused].index("d0")


def test_rrf_fuse_limits_to_top_k() -> None:
    dense = [chunk(f"d{i}") for i in range(5)]
    sparse = [chunk(f"s{i}") for i in range(5)]
    fused = rrf_fuse(dense, sparse, top_k=3)
    assert len(fused) == 3


def test_rrf_fuse_does_not_mutate_input_scores() -> None:
    dense = [chunk("d0", score=0.9)]
    sparse = [chunk("s0", score=4.0)]
    fused = rrf_fuse(dense, sparse, top_k=8)
    assert dense[0].score == 0.9
    assert sparse[0].score == 4.0
    assert fused[0].score < 1.0


def test_rrf_fuse_empty_inputs() -> None:
    assert rrf_fuse([], []) == []


# --- retrieval pipeline (no DB) ----------------------------------------------


def test_retrieve_empty_query_returns_insufficient() -> None:
    retriever, store, sparse_search, embedder = make_retriever([chunk("d0")], [])
    result = asyncio.run(retriever.retrieve("   "))
    assert result.chunks == []
    assert result.top_score == 0.0
    assert result.is_sufficient is False
    assert store.search_calls == []
    assert sparse_search.search_calls == []
    assert embedder.calls == []


def test_retrieve_runs_both_searches_and_fuses() -> None:
    dense = [chunk("d1", score=0.9), chunk("g", score=0.4)]
    sparse = [chunk("g", score=5.0), chunk("s1", score=2.0)]
    retriever, store, sparse_search, embedder = make_retriever(
        dense, sparse, vector=[1.0, 2.0]
    )

    result = asyncio.run(
        retriever.retrieve("lexical query", k=2, filters={"doc_ids": ["x"]})
    )

    assert embedder.calls == [["lexical query"]]
    assert store.search_calls == [
        {"embedding": [1.0, 2.0], "k": 2, "filters": {"doc_ids": ["x"]}}
    ]
    assert sparse_search.search_calls == [
        {"query": "lexical query", "k": 2, "filters": {"doc_ids": ["x"]}}
    ]
    assert [c.chunk_id for c in result.chunks] == ["g", "d1"]
    assert [c.citation_index for c in result.chunks] == [1, 2]
    assert result.top_score == pytest.approx(0.9)
    assert result.is_sufficient is True


def test_abstention_gate_below_threshold() -> None:
    retriever, _, _, _ = make_retriever([chunk("d1", score=0.1)], [chunk("s1")])
    result = asyncio.run(retriever.retrieve("anything", score_threshold=0.5))
    assert result.chunks, "context is still returned when gating"
    assert result.top_score == pytest.approx(0.1)
    assert result.is_sufficient is False


def test_gate_insufficient_without_dense_signal() -> None:
    retriever, _, _, _ = make_retriever([], [chunk("s1", score=9.0)])
    result = asyncio.run(retriever.retrieve("keyword"))
    assert [c.chunk_id for c in result.chunks] == ["s1"]
    assert result.top_score == 0.0
    assert result.is_sufficient is False


# --- context assembly ---------------------------------------------------------


def test_context_expands_neighbours_and_assigns_citations() -> None:
    parent = chunk("p", ordinal=2, score=0.7, tokens=10)
    neighbours = {
        ("doc-1", 1): [chunk("n-prev", ordinal=1, score=0.1, tokens=10)],
        ("doc-1", 3): [chunk("n-next", ordinal=3, score=0.1, tokens=10)],
    }
    retriever, store, _, _ = make_retriever([parent], [], neighbours=neighbours)

    result = asyncio.run(retriever.retrieve("q"))

    assert [c.chunk_id for c in result.chunks] == ["p", "n-prev", "n-next"]
    assert [c.citation_index for c in result.chunks] == [1, 2, 3]
    assert result.chunks[1].score == result.chunks[0].score
    assert result.chunks[2].score == result.chunks[0].score
    assert len(store.neighbour_calls) == 1


def test_context_dedupes_overlapping_neighbours() -> None:
    parent_a = chunk("a", ordinal=2, score=0.7, tokens=10)
    parent_b = chunk("b", ordinal=4, score=0.6, tokens=10)
    neighbours = {
        ("doc-1", 1): [chunk("n1", ordinal=1, tokens=10)],
        ("doc-1", 3): [chunk("n3", ordinal=3, tokens=10)],
        ("doc-1", 5): [chunk("n5", ordinal=5, tokens=10)],
    }
    retriever, _, _, _ = make_retriever([parent_a, parent_b], [], neighbours=neighbours)

    result = asyncio.run(retriever.retrieve("q"))

    assert [c.chunk_id for c in result.chunks] == ["a", "n1", "n3", "b", "n5"]
    assert [c.citation_index for c in result.chunks] == [1, 2, 3, 4, 5]


def test_context_respects_token_budget(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "context_token_budget", 900)
    parent = chunk("p", ordinal=2, tokens=400)
    neighbours = {
        ("doc-1", 1): [chunk("n1", ordinal=1, tokens=400)],
        ("doc-1", 3): [chunk("n3", ordinal=3, tokens=400)],
    }
    retriever, _, _, _ = make_retriever([parent], [], neighbours=neighbours)

    result = asyncio.run(retriever.retrieve("q"))

    assert [c.chunk_id for c in result.chunks] == ["p", "n1"]


def test_context_always_keeps_first_chunk(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "context_token_budget", 50)
    parent = chunk("p", ordinal=2, tokens=400)
    neighbours = {("doc-1", 1): [chunk("n1", ordinal=1, tokens=10)]}
    retriever, _, _, _ = make_retriever([parent], [], neighbours=neighbours)

    result = asyncio.run(retriever.retrieve("q"))

    assert [c.chunk_id for c in result.chunks] == ["p"]


# --- DB-backed hybrid retrieval (require PostgreSQL) --------------------------


def bow_vector(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIM
    for token in tokenize(text):
        vector[zlib.crc32(token.encode("utf-8")) % EMBEDDING_DIM] += 1.0
    return vector


class BowEmbedder:
    model = "bow-hash"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [bow_vector(text) for text in texts]


@pytest.fixture()
def db():
    session = SessionLocal()
    session.execute(delete(ChunkRow))
    session.execute(delete(Document))
    session.commit()
    yield session
    session.execute(delete(ChunkRow))
    session.execute(delete(Document))
    session.commit()
    session.close()


@pytest.fixture()
def seeded_db(db) -> dict[str, str]:
    doc_ids: dict[str, str] = {}
    for doc in _load_corpus()["documents"]:
        document = Document(
            id=uuid.uuid4(),
            title=doc["title"],
            format="md",
            content_hash=hashlib.sha256(doc["title"].encode()).hexdigest(),
            status="done",
        )
        db.add(document)
        db.flush()
        for ordinal, text in enumerate(doc["chunks"]):
            db.add(
                ChunkRow(
                    doc_id=document.id,
                    text=text,
                    embedding=bow_vector(text),
                    token_count=len(tokenize(text)),
                    ordinal=ordinal,
                    content_hash=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
        doc_ids[doc["title"]] = str(document.id)
    db.commit()
    return doc_ids


@requires_db
def test_golden_queries_hit_expected_doc_in_top5(seeded_db) -> None:
    queries = _load_corpus()["queries"]
    retriever = Retriever(embedder=BowEmbedder())
    hits = 0
    for query in queries:
        result = asyncio.run(retriever.retrieve(query["question"], k=8))
        expected_doc = seeded_db[query["expected_doc"]]
        assert result.is_sufficient, f"expected sufficient context for: {query['question']}"
        if any(c.doc_id == expected_doc for c in result.chunks[:5]):
            hits += 1
    assert hits >= math.ceil(0.8 * len(queries)), f"only {hits}/{len(queries)} queries hit top-5"


@requires_db
def test_unanswerable_query_triggers_abstention(seeded_db) -> None:
    retriever = Retriever(embedder=BowEmbedder())
    result = asyncio.run(
        retriever.retrieve(
            "flibbertigibbet zagzig woobler quantum",
            k=8,
            score_threshold=0.15,
        )
    )
    assert result.is_sufficient is False
    assert result.top_score < 0.15


@requires_db
def test_retrieval_p95_latency_under_300ms(seeded_db) -> None:
    queries = [q["question"] for q in _load_corpus()["queries"]]
    retriever = Retriever(embedder=BowEmbedder())
    asyncio.run(retriever.retrieve(queries[0], k=8))

    latencies = []
    for index in range(30):
        query = queries[index % len(queries)]
        start = time.perf_counter()
        asyncio.run(retriever.retrieve(query, k=8))
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    p95 = latencies[math.ceil(0.95 * len(latencies)) - 1]
    assert p95 < 300, f"retrieval p95 was {p95:.1f}ms"
