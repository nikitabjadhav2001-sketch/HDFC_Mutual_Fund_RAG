# Architecture Document — RAG Chatbot

| | |
|---|---|
| **Status** | Draft |
| **Version** | 0.1 |
| **Date** | 2026-10-07 |
| **Related** | `Docs/PRD.md` |

---

## 1. Overview

The system is a Retrieval-Augmented Generation (RAG) chatbot: documents are ingested, chunked, embedded, and indexed; at query time the system retrieves relevant chunks, assembles a grounded prompt, and streams an LLM answer with citations.

**Design principles**
- Provider-agnostic: LLM, embedding, and vector-store behind interfaces (swap without touching core logic).
- Retrieval and generation are separate services with a clear contract.
- Everything observable: every request carries a trace ID; retrieval results are logged alongside answers.
- Local-first: full stack runs via docker-compose; cloud deployable without code changes.

---

## 2. High-Level Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │                  Admin / CLI                │
                         │   upload · delete · status · index stats    │
                         └──────────────────┬──────────────────────────┘
                                            │
                                            ▼
┌──────────────┐   ┌───────────────────────────────────────────┐   ┌──────────────────┐
│  Documents   │──▶│           Ingestion Pipeline              │──▶│   Vector Index   │
│ pdf/docx/md  │   │  parse → clean → chunk → embed → write    │   │  (+ metadata)    │
└──────────────┘   └───────────────────────────────────────────┘   └────────┬─────────┘
                                                                            │
┌──────────────┐   ┌──────────────┐   ┌────────────────────────────┐        │
│   Chat UI    │──▶│  Chat API    │──▶│      RAG Orchestrator      │◀───────┘
│ (streaming)  │◀──│  (SSE/WS)    │   │  1. rewrite/query expand   │
└──────────────┘   └──────────────┘   │  2. hybrid retrieve        │
                                      │  3. rerank                 │
                                      │  4. context assembly       │
                                      │  5. generate (stream)      │
                                      │  6. cite + persist         │
                                      └──────────┬─────────────────┘
                                                 │
                                      ┌──────────▼─────────────────┐
                                      │   LLM Provider (abstract)  │
                                      │ OpenAI/Anthropic/vLLM/...  │
                                      └─────────────────────────────┘

              Cross-cutting: Auth · Config · Logging/Tracing · Eval hooks
```

---

## 3. Components

### 3.1 Ingestion Pipeline
**Responsibility:** turn documents into retrievable, cited chunks.

| Stage | Details |
|---|---|
| **Acquire** | Upload via admin API/UI, or watch a folder / connector (v2). |
| **Parse** | Format-specific parsers (PDF, DOCX, HTML, MD, TXT) → normalized document tree (headings, paragraphs, tables). |
| **Chunk** | Heading-aware, token-budgeted splitting with overlap; each chunk gets a stable ID, doc ID, title, section path, page numbers, timestamps. |
| **Embed** | Batch embedding via `EmbeddingProvider`; retries with backoff; cost/usage logged. |
| **Write** | Upsert vectors + metadata into index; status row updated (`queued → processing → done/failed`). |

- Runs as a **separate worker** (queue-based: Redis/RQ, Celery, or ARQ) so uploads never block chat.
- **Idempotent:** content hash per document/chunk → re-ingestion skips unchanged chunks.
- Supports delete/re-ingest by `doc_id` without full index rebuild.

### 3.2 Retrieval Service
**Responsibility:** return the best context for a query.

1. **Query understanding:** detect language, optional LLM-based query rewrite (resolve pronouns from chat history), extract filters.
2. **Parallel retrieval:**
   - **Dense:** vector similarity search (top-k) on the vector index.
   - **Sparse:** BM25/keyword search (lexical store or index-side hybrid support).
3. **Fusion:** Reciprocal Rank Fusion (RRF) → merged candidate list.
4. **Rerank (optional, config):** cross-encoder / LLM reranker over top-N (e.g., top 50 → top 5–8).
5. **Context assembly:** expand selected chunks with neighbors, dedupe by doc, truncate to token budget; each retained chunk keeps an index used for citation `[1]`, `[2]`.

**Interfaces:** `Retriever.retrieve(query, filters, k) → List[ScoredChunk]`, `Reranker.rerank(query, chunks, k) → List[ScoredChunk]`.

### 3.3 Generation Service
**Responsibility:** grounded, streamed, cited answers.

- **Prompt template (versioned):**
  - System: answer only from context, cite with `[n]`, say "I don't know" if not in context.
  - Blocks: `CONTEXT` (numbered chunks + source metadata), `HISTORY` (summarized/truncated turns), `QUESTION`.
- **Abstention:** if fused top score < threshold (configurable, tuned via eval) or reranker scores are poor → short refusal answer without calling the LLM (cheap + safe).
- **Citation formatting:** model outputs `[n]` markers; post-processor validates markers against provided context and maps them to source objects (title, page, snippet, link); invalid markers stripped.
- **Streaming:** SSE from Chat API to UI; first token target < 3s (PRD §5).
- **History management:** keep last N turns verbatim; summarize older turns when exceeding budget.

### 3.4 Chat API
- `POST /api/v1/chat` — send message, returns SSE stream (`token`, `sources`, `done`, `error` events).
- `GET /api/v1/conversations`, `GET /api/v1/conversations/{id}` — history.
- `DELETE /api/v1/conversations/{id}` — new chat.
- `GET /api/v1/health`, `GET /api/v1/index/stats`.

### 3.5 Admin API
- `POST /api/v1/admin/documents` (multipart upload), `GET .../documents`, `DELETE .../documents/{id}`, `GET .../documents/{id}/status`.
- Protected by admin role / API key.

### 3.6 Chat UI
- Streaming message view, loading/thinking state, collapsible **Sources** panel per answer, conversation list, new-chat button.
- Suggested: Next.js/React or simple server-rendered app for MVP.

### 3.7 Eval Harness
- CLI/CI job: runs golden-set questions through the full pipeline, scores retrieval (precision/recall, MRR) and generation (faithfulness, answer relevance, citation correctness) — e.g., RAGAS.
- Gates prompt/model/index changes; results stored as JSON artifacts for trend tracking.

---

## 4. Data Model (logical)

```
Document  { doc_id, title, source_uri, content_hash, format, status,
            uploaded_at, updated_at, tags[] }

Chunk     { chunk_id, doc_id, text, embedding, section_path, page_range,
            token_count, ordinal, content_hash }

Conversation { conv_id, user_id, created_at, title }

Message   { msg_id, conv_id, role, content, citations[], retrieved_chunk_ids[],
            model, prompt_tokens, completion_tokens, latency_ms, created_at }

EvalRun   { run_id, git_sha, config_hash, scores{}, created_at }
```

**Storage**
- **Vector index:** Qdrant / pgvector / FAISS (choose per scale; pgvector preferred if already on Postgres for MVP simplicity).
- **Relational:** Postgres for documents, conversations, messages, eval runs.
- **Queue/cache:** Redis (ingestion queue, optional response cache, rate limiting).

---

## 5. Request Lifecycle (happy path)

1. UI `POST /chat` → API authenticates, creates/loads conversation, generates `trace_id`.
2. Orchestrator rewrites query using history (optional, ≤1 extra LLM call or heuristics).
3. Dense + sparse retrieval run in parallel (p95 < 300ms), fusion, rerank.
4. Score gate: below threshold → stream refusal, log, stop.
5. Assemble prompt within token budget → LLM stream.
6. Each chunk delta forwarded as SSE; on completion, citations validated and attached; message + retrieval snapshot persisted.
7. Response logged with latency/tokens for observability (PRD FR-20).

**Failure handling:** retrieval failure → friendly error; LLM timeout/retry → bounded retries then error event; never return uncited partials silently.

---

## 6. Technology Choices (indicative)

| Concern | MVP choice | Alternatives |
|---|---|---|
| Language / API | Python (FastAPI) | Node (NestJS/Express) |
| Vector store | pgvector | Qdrant, Chroma, FAISS |
| Sparse search | Postgres FTS / Tantivy | Qdrant hybrid, Elasticsearch |
| Embeddings | OpenAI `text-embedding-3-small` | Cohere, local (bge/gte via TEI) |
| LLM | GPT-4o-mini / GPT-4o | Claude, Llama via vLLM/Ollama |
| Reranker | `bge-reranker` or Cohere Rerank | LLM-based rerank |
| Queue | Redis + ARQ/Celery | RQ, sidekiq-style workers |
| UI | React/Next.js | Streamlit (fastest demo) |
| Eval | RAGAS + pytest | DeepEval, custom |
| Packaging | docker-compose | K8s (later) |

*All swaps are contained behind `LLMProvider`, `EmbeddingProvider`, `VectorStore`, `Retriever` interfaces.*

---

## 7. Deployment

**Local / dev**
```
docker-compose up
  ├── api        (FastAPI: chat + admin)
  ├── worker     (ingestion consumer)
  ├── postgres   (+ pgvector)
  ├── redis
  └── ui
```

**Production (target):** same containers behind a reverse proxy (HTTPS), managed Postgres/Redis, workers scaled horizontally; LLM calls egress via allowlist. Stateless API → horizontal scale; ingestion workers scale by queue depth.

---

## 8. Security & Privacy

- **AuthN/AuthZ:** API keys or JWT for chat; separate admin role for document endpoints.
- **Secrets:** environment variables / secret manager; never in repo.
- **Data flow:** documents only reach configured LLM/embedding providers; no third-party logging of content; prompts/responses logged redacted.
- **Upload safety:** file-type allowlist, size limits, malware scan hook (v2), parse in sandboxed worker.
- **Prompt-injection defense:** retrieved text injected as clearly delimited data, system prompt instructs ignoring instructions found in context; admin-only ingestion reduces untrusted input.
- **Retention:** configurable conversation/document TTL (open question PRD §11.7).

---

## 9. Observability & SLOs

| Signal | What | Tooling |
|---|---|---|
| **Traces** | trace_id across API → retrieval → LLM spans | OpenTelemetry |
| **Metrics** | p95 latency (retrieval, first token), QPS, error rate, token spend, index freshness | Prometheus/Grafana |
| **Logs** | structured JSON: query, top chunks, answer ref, model, tokens | stdout → Loki/Cloud logs |
| **Quality** | eval scores per run; sampled live-answer review | Eval artifacts |

**SLOs (from PRD §5/§6):** first token p95 < 3s; retrieval p95 < 300ms; availability 99.5%; eval faithfulness ≥ 0.9.

---

## 10. Scalability & Performance Notes

- 100k+ chunks: pgvector with HNSW + metadata filtering is sufficient; migrate to Qdrant if QPS/latency demands.
- Embedding batches are async; ingestion never on the request path.
- Prompt/response caching for repeated questions (cache key = normalized query + index version).
- Parallel dense/sparse calls; rerank only top-N to bound cost/latency.
- Conversation history summarization caps context growth.

---

## 11. Risks (architecture view)

| Risk | Mitigation in design |
|---|---|
| Hallucination | Context-only prompts, score gate/abstention, citation validation, eval gate |
| Index staleness | Versioned index, freshness SLA, admin status + `index.stats` |
| Provider outage/lock-in | Provider interfaces, per-call fallback config, retries with backoff |
| Cost spikes | Token budgets per request, usage metrics, model tiering (small model default) |
| Prompt injection via docs | Delimited context, ingestion restricted to admins, injection eval cases |

---

## 12. Milestone Mapping

| Phase | Architecture scope |
|---|---|
| **M1 Spike** | Scripted pipeline: parse → chunk → embed → naive top-k → single-shot answer (CLI) |
| **M2 MVP** | Full components above: worker, hybrid retrieval, SSE API, basic UI, citations |
| **M3 Quality** | Reranker, abstention threshold, eval harness in CI, prompt versioning |
| **M4 Hardening** | Auth, OTel tracing, metrics/dashboards, load test, compose → cloud deploy |

---

## 13. Open Decisions

1. Vector store: **pgvector vs. Qdrant** (recommend pgvector for MVP).
2. LLM/embedding provider and region/privacy constraints.
3. Reranker in-path (latency cost) vs. retrieval-only quality.
4. UI: React app vs. Streamlit for M2.
5. Auth mechanism (SSO / API keys / anonymous).
