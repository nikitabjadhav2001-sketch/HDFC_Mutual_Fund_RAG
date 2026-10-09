# RAG Chatbot

Retrieval-Augmented Generation chatbot with grounded, cited answers.

Docs: `Docs/PRD.md` · `Docs/architecture.md` · `Docs/implementation.md`

## Quickstart (Docker)

```bash
cp .env.example .env        # add OPENAI_API_KEY (LLM), API_KEY/ADMIN_KEY for prod
docker compose up --build
curl localhost:8000/api/v1/health   # → {"status":"ok","db_ok":true,"index_count":0}

# complete system incl. Prometheus:
docker compose --profile full up --build   # API :8000, UI :5173, Prometheus :9090
```

Migrations run automatically on API startup (`alembic upgrade head`).

The chat UI is served at `http://localhost:5173` (nginx proxies `/api` to the API service).

Production compose (restart policies, no exposed DB/Redis ports, backup note in
the file header): `docker compose -f deploy/docker-compose.prod.yml up --build -d`

## Quickstart (local venv)

Requires a local PostgreSQL 16 with pgvector and Redis.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # or: source .venv/bin/activate
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload
```

## Development

```bash
make lint           # ruff check .
make test           # pytest -q
make eval           # Phase 5 quality gate (faithfulness ≥ 0.9, hit@5 ≥ 0.8)
make migrate        # alembic upgrade head
```

### Embeddings

Default: `EMBEDDING_PROVIDER=local` — `sentence-transformers/all-MiniLM-L6-v2`
(384-dim) running on-device via the `sentence-transformers` package; first use
downloads the model (~90 MB), then everything is offline and free.

Alternatives: `EMBEDDING_PROVIDER=huggingface` (HF Inference API — needs HF
credits and `HF_TOKEN` in `.env`) or `openai` (any OpenAI-compatible
`/embeddings` endpoint) or `hash` (offline, tests/CI only).

The embedding model/dimension is baked into `chunks.embedding`
(`vector(384)`); changing it needs a matching migration plus full
re-ingestion (`python -m app.ingest <source_uri>`).

### UI (Phase 4)

```bash
cd ui
npm install
npm run dev         # http://localhost:5173 (proxies /api → localhost:8000)
npm run build       # production build → ui/dist
```

Base URL is configurable via `VITE_API_BASE_URL` (default: same-origin; the dev
server proxies `/api` to `http://localhost:8000`, override with
`VITE_API_PROXY_TARGET`). API keys are read at build/runtime from
`VITE_API_KEY` / `VITE_ADMIN_KEY` (empty = auth disabled, localhost dev).

## Conversation memory

Each conversation keeps a rolling window of the **10 most recently retrieved
chunks** (`RETRIEVAL_MEMORY_WINDOW` in `app/rag/orchestrator.py`). Follow-up
turns re-inject them as an "EARLIER RETRIEVED CONTEXT" block (background only,
not citable with `[n]`), so questions like "what about the second one?" resolve
even when fresh retrieval misses. Memory is scoped per conversation, evicted on
conversation delete, and included in the answer-cache key. Eval runs are
stateless (no conversation → no memory).

## Security & observability (Phase 6)

- **Auth**: `Authorization: Bearer <key>`; `API_KEY` guards chat routes,
  `ADMIN_KEY` guards `/admin/*` (falls back to `API_KEY`). Health and
  `/metrics` stay open for healthchecks/scraping. Wrong key → 401.
- **Rate limiting**: Redis token bucket per client, `RATE_LIMIT_PER_MINUTE` /
  `RATE_LIMIT_BURST` (429 + `Retry-After` when exceeded; fails open if Redis is
  down).
- **Logs**: JSON on stdout with `trace_id` — one id across the access line,
  `retrieval_snapshot` log (retrieved chunk ids + top score + latency) and LLM
  errors; also echoed as the `X-Request-ID` response header.
- **Tracing**: OpenTelemetry spans (http → retrieve → llm); set
  `OTEL_EXPORTER_OTLP_ENDPOINT` (e.g. `http://otel-collector:4318`) to export.
- **Metrics**: `GET /metrics` (Prometheus) — request latency histogram, token
  counters, LLM failures, retrieval latency; scraped by the `full` profile.
- **Caching**: exact normalized question (+ history digest) in Redis,
  `QUERY_CACHE_TTL` seconds, invalidated by index version; disabled with `0`.

## Load test (Phase 6)

```bash
API_KEY=... locust -f loadtest/locustfile.py --host http://localhost:8000 \
    -u 50 -r 5 -t 120s --headless
```

Acceptance: 50 concurrent sessions, `chat_first_token` p95 < 3s in the Locust
summary.

## Layout

- `app/api` — FastAPI routes (health, chat, conversations, admin, metrics)
- `app/core` — provider interfaces (LLM, embeddings, vector store), security,
  caching, logging/telemetry
- `app/ingestion` — parse → chunk → embed pipeline (Phase 1)
- `app/rag` — retrieval, prompts, orchestrator (Phases 2–3)
- `app/models` — SQLAlchemy models
- `migrations` — Alembic migrations
- `ui` — React (Vite) chat interface (Phase 4)
- `eval` — Phase 5 golden set + quality gate (`make eval`)
- `loadtest` — Locust script (Phase 6)
- `deploy` — Prometheus config + production compose
- `tests` — pytest suite
