# Implementation Plan — RAG Chatbot

| | |
|---|---|
| **Status** | Draft |
| **Version** | 0.1 |
| **Date** | 2026-10-07 |
| **Related** | `Docs/PRD.md`, `Docs/architecture.md` |

**How to use this document:** implement phase by phase in Cursor. Each phase is self-contained: complete all tasks, satisfy the acceptance criteria, run the verification commands, then move to the next phase. Do not start a phase until the previous phase's acceptance criteria pass.

**Locked stack (change only if explicitly told):**
- Python 3.11+, FastAPI, Pydantic v2
- PostgreSQL 16 + pgvector (vector + relational data)
- Redis + ARQ (ingestion queue) — Phase 1+
- OpenAI API (embeddings: `text-embedding-3-small`, chat: configurable model)
- Simple BM25 (`rank-bm25`) over Postgres full-text or in-process for MVP
- UI: minimal React (Vite) — or Streamlit if told to go faster
- Testing: pytest; lint: ruff; package: uv/pip + docker-compose

---

## Phase 0 — Project Scaffold

**Goal:** runnable skeleton with health check, config, and DB migrations.

### Tasks
1. **Repo layout**
   ```
   RAG_7thOCT/
   ├── app/
   │   ├── main.py              # FastAPI app factory, router mounting
   │   ├── config.py            # pydantic-settings (env vars)
   │   ├── api/
   │   │   ├── routes_chat.py
   │   │   ├── routes_admin.py
   │   │   └── routes_health.py
   │   ├── core/
   │   │   ├── llm_provider.py   # interface + OpenAI impl (stub in P0)
   │   │   ├── embedding.py
   │   │   └── vectorstore.py    # interface + pgvector impl
   │   ├── ingestion/
   │   │   ├── parsers.py
   │   │   ├── chunker.py
   │   │   └── pipeline.py
   │   ├── rag/
   │   │   ├── retriever.py
   │   │   ├── reranker.py
   │   │   ├── prompts.py
   │   │   └── orchestrator.py
   │   ├── models/               # SQLAlchemy models
   │   ├── schemas/              # Pydantic request/response
   │   └── db.py                 # engine/session
   ├── migrations/               # Alembic
   ├── tests/
   ├── docker-compose.yml        # postgres+pgvector, redis, api, worker, ui
   ├── Dockerfile
   ├── requirements.txt / pyproject.toml
   ├── .env.example
   └── README.md
   ```
2. `docker-compose.yml`: `postgres` (image `pgvector/pgvector:pg16`), `redis`, `api`.
3. Alembic init; migration creating `documents`, `chunks` (with `vector(1536)` + HNSW index), `conversations`, `messages` tables.
4. `GET /api/v1/health` → `{status, db_ok, index_count}`.
5. `.env.example`: `DATABASE_URL`, `REDIS_URL`, `OPENAI_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL`.

### Acceptance criteria
- `docker compose up` starts cleanly; `curl localhost:8000/api/v1/health` returns `db_ok: true`.
- `pytest` runs (even if only 1 smoke test) and passes.

### Verify
```bash
docker compose up -d && sleep 10
curl -s localhost:8000/api/v1/health
pytest -q
```

---

## Phase 1 — Ingestion Pipeline

**Goal:** document in → chunks embedded and stored, with status tracking.

### Tasks
1. **Parsers** (`app/ingestion/parsers.py`): `parse(file_bytes, format) -> ParsedDoc(title, sections[{heading, text, page}])` supporting TXT, MD, PDF (pypdf), DOCX (python-docx).
2. **Chunker** (`chunker.py`): heading-aware, `max_tokens≈400`, `overlap≈80` (configurable); output `Chunk(text, ordinal, section_path, page_start, page_end, content_hash)`.
3. **Embedding** (`core/embedding.py`): `EmbeddingProvider.embed_texts(list[str]) -> list[list[float]]`, batched (≤100), retries with backoff, usage logged.
4. **Pipeline** (`pipeline.py`): `ingest(document_id)` — parse → chunk → embed → upsert into `chunks` + set `documents.status` (`queued/processing/done/failed`, store error). Idempotent via content hash (skip unchanged).
5. **Models + migrations**: `documents`, `chunks` (FK, metadata columns, embedding, HNSW index).
6. **Admin endpoints** (`routes_admin.py`): upload (multipart), list, get status, delete (removes chunks + doc row).
7. **Worker**: ARQ consumer processing an `ingest` queue; `POST /documents` enqueues.
8. **CLI fallback**: `python -m app.ingest path/to/file.pdf` (sync path for tests).

### Acceptance criteria
- Upload a PDF and an MD file → status reaches `done`; `SELECT count(*) FROM chunks` > 0.
- Re-upload identical file → no duplicate chunks (hash skip).
- Delete document → its chunks removed.
- Parse failure marks doc `failed` with error message, API stays healthy.

### Verify
```bash
pytest tests/test_chunker.py tests/test_ingestion.py -q
curl -F "file=@sample.pdf" localhost:8000/api/v1/admin/documents
curl localhost:8000/api/v1/admin/documents/<id>/status   # → done
```

---

## Phase 2 — Retrieval

**Goal:** query in → best chunks out, with fusion and score.

### Tasks
1. **Dense retrieval** (`vectorstore.py`): `similarity_search(embedding, k, filters) -> List[ScoredChunk]` using pgvector cosine.
2. **Sparse retrieval**: BM25 via `rank-bm25` over chunks loaded per query (MVP) **or** Postgres FTS (`ts_rank`); return same `ScoredChunk` shape.
3. **Fusion** (`retriever.py`): RRF (`k=60`) merging dense + sparse lists; final score exposed.
4. **`Retriever.retrieve(query, k=8, filters=None)`** — embeds query, runs both searches in parallel (`asyncio.gather`), fuses.
5. **Context assembly**: expand top chunks with ±1 neighbor, dedupe by `doc_id`, truncate to `context_token_budget` (configurable, default ~3000 tokens); assign citation indices.
6. **Abstention gate**: `top_score < RERANK/SCORE_THRESHOLD` → `RetrievalResult.is_sufficient=False`.
7. **Tests**: golden queries over a seeded fixture corpus; assert expected doc appears in top-k; assert fusion beats pure-dense on a lexical-only query.

### Acceptance criteria
- Seeded corpus (≥10 docs) test: expected chunk in top-5 for ≥80% of fixture queries.
- Unanswerable query → `is_sufficient=False`.
- Retrieval p95 < 300ms locally on the fixture corpus.

### Verify
```bash
pytest tests/test_retriever.py -q
python -m app.debug_query "your question"   # prints top-k + scores
```

---

## Phase 3 — Generation + Chat API (streaming, citations)

**Goal:** end-to-end cited answers via streaming API.

### Tasks
1. **`core/llm_provider.py`**: `LLMProvider.stream(messages) -> AsyncIterator[str]`; OpenAI impl; model from config.
2. **Prompts** (`prompts.py`): versioned templates — system grounding rules ("answer only from context", "cite [n]", "say I don't know"), context block with numbered chunks + source metadata, history, question.
3. **Orchestrator** (`orchestrator.py`):
   - retrieve → gate (refusal short-circuit) → build prompt (history: last N turns verbatim, summarize older when over budget) → stream LLM → post-process citations.
   - Citation validator: strip `[n]` markers not present in context; map to source objects `{chunk_id, doc_title, page, snippet}`.
4. **Chat endpoints** (`routes_chat.py`): `POST /api/v1/chat` (SSE events: `token`, `sources`, `done`, `error`), conversations CRUD.
5. **Persistence**: save user + assistant messages with `retrieved_chunk_ids`, citations, tokens, latency, `trace_id`.
6. **History management**: in-context window = last 6 turns; older compressed to a one-paragraph summary (LLM call, cacheable).

### Acceptance criteria
- `curl -N POST /chat` streams tokens; final event contains `sources` with title/page for grounded claims.
- Follow-up "what about the second one?" resolves using history (fixture test).
- KB-unanswerable question → refusal answer, no hallucinated content.
- Answer + citations persisted; retrievable via `GET /conversations/{id}`.

### Verify
```bash
pytest tests/test_orchestrator.py tests/test_chat_api.py -q
curl -N -X POST localhost:8000/api/v1/chat -H 'Content-Type: application/json' \
  -d '{"conversation_id": null, "message": "What does the doc say about X?"}'
```


---

## Phase 4 — Chat UI

**Goal:** usable web frontend.

### Tasks
1. Vite + React app in `ui/`; docker-compose service.
2. Views: chat (streaming via SSE/fetch-reader), sources panel expandable per message, conversation sidebar (list, new, delete), upload/status page (admin).
3. States: loading/thinking, error banner, empty state.
4. Wire to API with base-URL config; no auth yet (localhost only).


### Acceptance criteria
- Ask a question in browser → tokens stream live → sources visible with titles/snippets.
- New chat clears context; old conversations load from sidebar.
- Upload a doc from the UI → status reaches `done` → subsequent answers use it.

### Verify
Manual walkthrough of the four use cases (PRD §3) in the browser.

---

## Phase 5 — Evaluation Harness

**Goal:** measurable quality gates (PRD §6–7).

### Tasks
1. `eval/dataset.jsonl`: golden set `{question, expected_doc_ids[], expected_keywords[], acceptable_answer?}` — start with ≥20 entries over the fixture corpus.
2. `eval/run_eval.py`: runs full pipeline per question → outputs retrieval metrics (hit@k, MRR, context precision/recall) + generation metrics (RAGAS: faithfulness, answer relevancy, citation correctness if supported) → `eval/results/{timestamp}.json`.
3. CI/Makefile target: `make eval`; non-zero exit if metrics below thresholds (faithfulness ≥ 0.9, hit@5 ≥ 0.8 initially — tune).
4. Prompt version tag recorded in results (`git_sha` + prompt version).

### Acceptance criteria
- `make eval` produces a results file and exits 0 on current thresholds.
- Deliberately breaking the prompt (e.g., "answer creatively") makes eval fail.

### Verify
```bash
make eval && cat eval/results/latest.json
```

---

## Phase 6 — Hardening (production pilot)

**Goal:** security, observability, performance.

### Tasks
1. **Auth**: API key (chat) + admin key (admin routes) via `Authorization` header; simple dependency check. (Swap for JWT/SSO if required.)
2. **Rate limiting**: Redis token bucket per key.
3. **Observability**: structured JSON logs with `trace_id`; OpenTelemetry spans (api → retrieve → llm); Prometheus `/metrics` (latency histograms, token counters, errors); log retrieval snapshot per answer.
4. **Caching**: cache exact normalized queries (key includes index version) with short TTL.
5. **Robustness**: upload size/type limits, LLM retry/timeout, graceful provider-error messages.
6. **Load test**: `locust`/`k6` script — ≥50 concurrent sessions, verify p95 first token < 3s.
7. **Deploy**: prod compose or K8s manifests; secrets via env; backup note for Postgres.

### Acceptance criteria
- All endpoints require keys; wrong key → 401.
- Dashboards/logs show per-request trace with retrieval + latency.
- Load test meets p95 target; eval still passes.
- `docker compose --profile full up` runs the complete system from a clean machine.

---

## Cross-phase Rules (for Cursor)

1. **Interfaces first:** implement `LLMProvider`, `EmbeddingProvider`, `VectorStore`, `Retriever` as explicit protocols/classes; all config via `app/config.py` env vars — no hardcoded keys/models/paths.
2. **No dead code, no extra features:** build only the phase's tasks; leave TODOs for deferred items.
3. **Every phase ends green:** `ruff check . && pytest -q` must pass before moving on.
4. **Tests follow tasks:** unit tests for chunker/fusion/citation-validator/parsers; API tests with `httpx.AsyncClient` + test DB; fixture corpus under `tests/fixtures/`.
5. **Secrets:** only `.env` (git-ignored), `.env.example` committed.
6. **Commit granularity:** one commit per task group with message `P{phase}: {scope}` (only when asked to commit).

## Definition of Done (overall)

- [ ] Phases 0–6 acceptance criteria all pass
- [ ] PRD metrics validated by `make eval` (faithfulness ≥ 0.9, hit@5 ≥ 0.8, first-token p95 < 3s)
- [ ] README: quickstart (compose up), env vars, architecture summary link
- [ ] `ruff` clean, test suite green in CI
