import logging
import re
from collections import OrderedDict
from time import perf_counter
from typing import Any, AsyncIterator, Dict, List

from app.core.cache import get_query_cache
from app.core.llm_provider import LLMProvider
from app.rag.prompts import (
    SYSTEM_PROMPT,
    build_chat_messages,
    format_context_block,
    format_memory_block,
)
from app.rag.retriever import RetrievalResult, Retriever

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = (
    "I don't have enough information in the provided context to answer this question."
)

HISTORY_WINDOW_MESSAGES = 12  # last 6 turns
RETRIEVAL_MEMORY_WINDOW = 10  # chunks remembered across turns per conversation
MAX_MEMORY_CONVERSATIONS = 200  # oldest conversations are evicted first


class GenerationOrchestrator:
    def __init__(self):
        self.llm = LLMProvider()
        self.retriever = Retriever()
        # conversation_id -> retrieved chunks from earlier turns (recency ordered)
        self._retrieval_memory: OrderedDict[str, list] = OrderedDict()

    def retrieval_memory(self, conversation_id: str | None) -> list:
        """Chunks retrieved in earlier turns of this conversation (newest last)."""
        if not conversation_id:
            return []
        return list(self._retrieval_memory.get(conversation_id, ()))

    def forget(self, conversation_id: str | None) -> None:
        """Drop remembered retrievals for a conversation (e.g. on delete)."""
        if conversation_id:
            self._retrieval_memory.pop(conversation_id, None)

    def _remember(self, conversation_id: str | None, chunks: list) -> None:
        """Record this turn's retrieval, deduped, capped at the memory window."""
        if not conversation_id or not chunks:
            return
        current_ids = {chunk.chunk_id for chunk in chunks}
        remembered = [
            chunk
            for chunk in self._retrieval_memory.pop(conversation_id, [])
            if chunk.chunk_id not in current_ids
        ]
        remembered.extend(chunks)
        self._retrieval_memory[conversation_id] = remembered[-RETRIEVAL_MEMORY_WINDOW:]
        while len(self._retrieval_memory) > MAX_MEMORY_CONVERSATIONS:
            self._retrieval_memory.popitem(last=False)

    async def stream_response(
        self,
        query: str,
        history: List[Dict[str, str]] = None,
        k: int = 5,
        retrieval: RetrievalResult | None = None,
        use_cache: bool = True,
        conversation_id: str | None = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream chat events; pass `retrieval` to reuse an already-computed result.

        `conversation_id` scopes the retrieval memory: chunks retrieved in
        earlier turns are re-injected as background context for follow-ups.
        """
        history = history or []
        # Memory state before this turn's retrieval — key for cache load/store.
        memory_ids = self._memory_ids(conversation_id)

        # 0. Exact-query cache bypasses retrieval + generation (Redis, fail open)
        cache = get_query_cache() if use_cache else None
        if cache is not None and retrieval is None:
            cached = cache.load(query, history, memory_ids=memory_ids)
            if cached is not None:
                yield {"type": "token", "content": cached.get("answer", "")}
                yield {"type": "sources", "sources": cached.get("sources", [])}
                yield {
                    "type": "done",
                    "retrieved_chunk_ids": cached.get("retrieved_chunk_ids", []),
                    "cached": True,
                }
                return

        # 1. Hybrid retrieval (dense + sparse + RRF fusion + context assembly)
        started = perf_counter()
        result = retrieval if retrieval is not None else await self.retriever.retrieve(
            query=query, k=k
        )
        chunks = result.chunks
        logger.info(
            "retrieval_snapshot",
            extra={
                "event": "retrieval_snapshot",
                "query": query,
                "retrieved_chunk_ids": [chunk.chunk_id for chunk in chunks],
                "top_score": result.top_score,
                "is_sufficient": result.is_sufficient,
                "retrieval_ms": round((perf_counter() - started) * 1000, 1),
            },
        )

        # 1b. Remember this turn's retrieval for follow-up questions
        self._remember(conversation_id, chunks)

        # 2. Refusal gate: short-circuit when retrieval is not sufficient
        if not result.is_sufficient or not chunks:
            yield {"type": "token", "content": REFUSAL_MESSAGE}
            yield {"type": "sources", "sources": []}
            yield {"type": "done", "retrieved_chunk_ids": []}
            return

        context_block = format_context_block(chunks)

        # 2b. Prior-turn retrievals as background memory (current turn excluded)
        current_ids = {chunk.chunk_id for chunk in chunks}
        remembered = [
            chunk
            for chunk in self.retrieval_memory(conversation_id)
            if chunk.chunk_id not in current_ids
        ]
        memory_block = format_memory_block(remembered)

        # 3. Sliding history window (last 6 turns)
        windowed_history = history[-HISTORY_WINDOW_MESSAGES:]
        messages = build_chat_messages(
            SYSTEM_PROMPT, context_block, windowed_history, query, memory_block
        )

        # 4. Stream tokens from LLM
        full_response = ""
        async for token in self.llm.stream(messages):
            full_response += token
            yield {"type": "token", "content": token}

        # 5. Extract and map citations [1], [2] to source objects
        cited_indices = list(
            dict.fromkeys(int(idx) for idx in re.findall(r"\[(\d+)\]", full_response))
        )
        valid_sources = []
        for idx in cited_indices:
            if 1 <= idx <= len(chunks):
                chunk = chunks[idx - 1]
                valid_sources.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_title": chunk.metadata.get("document_title") or "Document",
                    "page": chunk.metadata.get("page_start"),
                    "snippet": chunk.text[:200],
                })

        retrieved_ids = [chunk.chunk_id for chunk in chunks]
        yield {"type": "sources", "sources": valid_sources}
        yield {"type": "done", "retrieved_chunk_ids": retrieved_ids}

        # 6. Warm the cache with the computed answer (only complete generations)
        if cache is not None:
            cache.store(
                query,
                history,
                {
                    "answer": full_response,
                    "sources": valid_sources,
                    "retrieved_chunk_ids": retrieved_ids,
                },
                memory_ids=memory_ids,
            )

    def _memory_ids(self, conversation_id: str | None) -> list[str] | None:
        ids = [chunk.chunk_id for chunk in self.retrieval_memory(conversation_id)]
        return ids or None
