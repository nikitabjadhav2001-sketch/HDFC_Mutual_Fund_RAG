from unittest.mock import AsyncMock, patch

import anyio

from app.core.vectorstore import ScoredChunk
from app.rag.orchestrator import GenerationOrchestrator
from app.rag.retriever import RetrievalResult


def _result(chunks, top_score=0.0, sufficient=False):
    return RetrievalResult(
        query="q", chunks=chunks, top_score=top_score, is_sufficient=sufficient
    )


def test_orchestrator_refusal_when_insufficient_context():
    async def run_test():
        orchestrator = GenerationOrchestrator()

        with patch.object(
            orchestrator.retriever, "retrieve", new_callable=AsyncMock
        ) as mock_retrieve:
            mock_retrieve.return_value = _result([])

            events = []
            async for event in orchestrator.stream_response("Unanswerable question?"):
                events.append(event)

            assert len(events) == 3
            assert events[0]["type"] == "token"
            assert "don't have enough information" in events[0]["content"]
            assert events[1]["type"] == "sources"
            assert events[1]["sources"] == []
            assert events[2]["type"] == "done"
            assert events[2]["retrieved_chunk_ids"] == []

    anyio.run(run_test)


def test_orchestrator_refusal_when_gate_is_closed():
    async def run_test():
        orchestrator = GenerationOrchestrator()
        weak_chunk = ScoredChunk(
            chunk_id="weak", doc_id="doc-1", text="tangential", score=0.05, metadata={}
        )

        with patch.object(
            orchestrator.retriever, "retrieve", new_callable=AsyncMock
        ) as mock_retrieve:
            mock_retrieve.return_value = _result([weak_chunk], top_score=0.05, sufficient=False)

            events = []
            async for event in orchestrator.stream_response("Anything?"):
                events.append(event)

            assert [e["type"] for e in events] == ["token", "sources", "done"]
            assert "don't have enough information" in events[0]["content"]

    anyio.run(run_test)


def test_orchestrator_streams_tokens_and_extracts_citations():
    async def run_test():
        orchestrator = GenerationOrchestrator()

        chunk = ScoredChunk(
            chunk_id="chunk_123",
            doc_id="doc-1",
            text="This is test context for chunk 1.",
            score=0.9,
            metadata={"document_title": "Test Doc", "page_start": 1},
        )

        with patch.object(
            orchestrator.retriever, "retrieve", new_callable=AsyncMock
        ) as mock_retrieve, patch.object(orchestrator.llm, "stream") as mock_stream:

            mock_retrieve.return_value = _result([chunk], top_score=0.9, sufficient=True)

            async def mock_llm_stream(messages):
                yield "According to "
                yield "the doc [1]."

            mock_stream.side_effect = mock_llm_stream

            events = []
            async for event in orchestrator.stream_response("What is the context?"):
                events.append(event)

            tokens = [e["content"] for e in events if e["type"] == "token"]
            assert "".join(tokens) == "According to the doc [1]."

            sources = [e["sources"] for e in events if e["type"] == "sources"][0]
            assert len(sources) == 1
            assert sources[0]["chunk_id"] == "chunk_123"
            assert sources[0]["doc_title"] == "Test Doc"
            assert sources[0]["page"] == 1

            done = [e for e in events if e["type"] == "done"][0]
            assert done["retrieved_chunk_ids"] == ["chunk_123"]

            mock_retrieve.assert_awaited_once_with(query="What is the context?", k=5)

    anyio.run(run_test)


def _chunk(i: int, text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=f"chunk_{i}",
        doc_id="doc-1",
        text=text,
        score=0.9,
        metadata={"document_title": f"Doc {i}"},
    )


def _capture_llm(orchestrator):
    """Patch llm.stream so every call's messages are recorded; reply 'ok'."""
    captured: list[list[dict]] = []

    def fake_stream(messages):
        captured.append(messages)

        async def tokens():
            yield "ok"

        return tokens()

    return captured, patch.object(orchestrator.llm, "stream", side_effect=fake_stream)


def test_memory_not_present_on_first_turn_and_injected_on_followup():
    async def run_test():
        orchestrator = GenerationOrchestrator()
        captured, llm_patch = _capture_llm(orchestrator)

        with patch.object(
            orchestrator.retriever, "retrieve", new_callable=AsyncMock
        ) as mock_retrieve, llm_patch:
            mock_retrieve.return_value = _result(
                [_chunk(1, "memory text alpha")], top_score=0.9, sufficient=True
            )
            async for _ in orchestrator.stream_response(
                "q1", conversation_id="conv-a", use_cache=False
            ):
                pass

            mock_retrieve.return_value = _result(
                [_chunk(2, "current text beta")], top_score=0.9, sufficient=True
            )
            async for _ in orchestrator.stream_response(
                "q2",
                history=[{"role": "user", "content": "q1"}],
                conversation_id="conv-a",
                use_cache=False,
            ):
                pass

        first_system = captured[0][0]["content"]
        assert "EARLIER RETRIEVED CONTEXT" not in first_system

        second_system = captured[1][0]["content"]
        assert "EARLIER RETRIEVED CONTEXT" in second_system
        assert "memory text alpha" in second_system
        # memory block must not repeat the current turn's chunks
        memory_section = second_system.split("EARLIER RETRIEVED CONTEXT", 1)[1]
        assert "current text beta" not in memory_section

    anyio.run(run_test)


def test_retrieval_memory_window_caps_at_10_chunks():
    async def run_test():
        orchestrator = GenerationOrchestrator()
        captured, llm_patch = _capture_llm(orchestrator)

        with patch.object(
            orchestrator.retriever, "retrieve", new_callable=AsyncMock
        ) as mock_retrieve, llm_patch:
            for turn in range(3):
                chunks = [_chunk(turn * 5 + i, f"t{turn} c{i}") for i in range(1, 6)]
                mock_retrieve.return_value = _result(chunks, top_score=0.9, sufficient=True)
                async for _ in orchestrator.stream_response(
                    f"q{turn}", conversation_id="conv-a", use_cache=False
                ):
                    pass

        remembered_ids = [
            chunk.chunk_id for chunk in orchestrator.retrieval_memory("conv-a")
        ]
        assert len(remembered_ids) == 10
        # oldest turn (chunk_1..chunk_5) evicted, newest retained in order
        assert remembered_ids == [f"chunk_{i}" for i in range(6, 16)]

        third_system = captured[2][0]["content"]
        memory_section = third_system.split("EARLIER RETRIEVED CONTEXT", 1)[1]
        assert "Doc 15" not in memory_section  # current turn not in memory
        assert "t0" not in memory_section  # evicted first turn
        assert "t1 c1" in memory_section

    anyio.run(run_test)


def test_retrieval_memory_is_scoped_per_conversation():
    async def run_test():
        orchestrator = GenerationOrchestrator()
        captured, llm_patch = _capture_llm(orchestrator)

        with patch.object(
            orchestrator.retriever, "retrieve", new_callable=AsyncMock
        ) as mock_retrieve, llm_patch:
            mock_retrieve.return_value = _result(
                [_chunk(1, "conv a fact")], top_score=0.9, sufficient=True
            )
            async for _ in orchestrator.stream_response(
                "q", conversation_id="conv-a", use_cache=False
            ):
                pass

            mock_retrieve.return_value = _result(
                [_chunk(2, "conv b fact")], top_score=0.9, sufficient=True
            )
            async for _ in orchestrator.stream_response(
                "q", conversation_id="conv-b", use_cache=False
            ):
                pass

            # no conversation: nothing remembered, nothing injected
            async for _ in orchestrator.stream_response("q", use_cache=False):
                pass
            async for _ in orchestrator.stream_response("q", use_cache=False):
                pass

        assert [c.chunk_id for c in orchestrator.retrieval_memory("conv-a")] == ["chunk_1"]
        assert [c.chunk_id for c in orchestrator.retrieval_memory("conv-b")] == ["chunk_2"]
        assert orchestrator.retrieval_memory(None) == []

        assert "conv a fact" not in captured[1][0]["content"]  # conv-b prompt
        assert "EARLIER RETRIEVED CONTEXT" not in captured[3][0]["content"]  # stateless

        orchestrator.forget("conv-a")
        assert orchestrator.retrieval_memory("conv-a") == []

    anyio.run(run_test)
