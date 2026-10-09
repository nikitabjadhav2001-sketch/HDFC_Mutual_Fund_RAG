# Product Requirements Document (PRD)
## RAG Chatbot

| | |
|---|---|
| **Status** | Draft |
| **Version** | 0.1 |
| **Date** | 2026-10-07 |
| **Source** | `Docs/ProblemStatement.txt` (empty at time of writing — content based on stated goal: "building a RAG chatbot") |

---

## 1. Problem Statement

Users need accurate, trustworthy answers that are grounded in a specific, evolving knowledge base (documents, FAQs, policies, etc.). General-purpose LLM chatbots hallucinate and cannot cite organization-specific or up-to-date information.

**Goal:** Build a Retrieval-Augmented Generation (RAG) chatbot that retrieves relevant content from a curated knowledge base and generates answers grounded in that content, with citations.

---

## 2. Goals & Non-Goals

### Goals
- Ingest a corpus of documents and make it searchable.
- Answer user questions using only retrieved context where possible.
- Cite sources for every answer so users can verify.
- Provide a conversational interface (multi-turn chat with memory).
- Keep the knowledge base updateable without retraining a model.

### Non-Goals (v1)
- Fine-tuning or training custom LLMs.
- Agentic multi-step tool use beyond retrieval.
- Voice / multimodal (image, video) input.
- Multi-tenant SaaS billing and permissions (unless required).

---

## 3. Users & Use Cases

| Persona | Need |
|---|---|
| **End user** | Ask questions in natural language; get fast, cited answers. |
| **Knowledge admin** | Upload/remove documents; monitor ingestion and index health. |
| **Developer** | Integrate the chatbot via API/SDK into an existing product. |

**Primary use cases**
1. Ask a question → get a grounded answer with source links/snippets.
2. Follow-up questions in the same conversation (coreference, context carry-over).
3. "I don't know" behavior when the KB has no relevant information.
4. Admin uploads new documents → answers reflect them within a defined SLA.

---

## 4. Functional Requirements

### 4.1 Ingestion
- **FR-1:** Support upload of PDF, DOCX, TXT, Markdown, and HTML (extendable).
- **FR-2:** Parse documents into structured chunks (heading-aware, size- and overlap-configurable).
- **FR-3:** Generate embeddings per chunk and store in a vector index, alongside metadata (source, title, page/section, timestamp, tags).
- **FR-4:** Support re-ingestion, incremental updates, and deletion of documents without full rebuild.
- **FR-5:** Ingestion pipeline is idempotent and reports per-document status (queued / processing / done / failed).

### 4.2 Retrieval
- **FR-6:** Hybrid retrieval: dense (vector) + sparse (keyword/BM25) search, with score fusion (e.g., reciprocal rank fusion).
- **FR-7:** Optional re-ranking of top-k candidates with a cross-encoder or LLM reranker.
- **FR-8:** Metadata filters (date, source, tags) applied at query time.
- **FR-9:** Configurable `top_k`, score threshold, and chunk-merging/windowing (include neighboring chunks for context).

### 4.3 Generation
- **FR-10:** Prompt construction: system instructions + retrieved context + conversation history + user question.
- **FR-11:** Every factual claim traceable to retrieved context; answers include inline citations and a source list.
- **FR-12:** Refusal/abstention: if retrieval confidence is below threshold, respond that the KB has no answer.
- **FR-13:** Streaming token-by-token responses to the UI.
- **FR-14:** Configurable model (provider-agnostic abstraction: OpenAI, Anthropic, open-weights via vLLM/Ollama, etc.).

### 4.4 Chat Experience
- **FR-15:** Multi-turn conversations with history summarization/truncation to fit context limits.
- **FR-16:** Show sources (title, snippet, link) expandable under each answer.
- **FR-17:** Explicit "thinking"/loading states; graceful error messages.
- **FR-18:** Conversation history persisted per user session; ability to start new chat.

### 4.5 Admin & Operations
- **FR-19:** Admin UI or CLI to upload, list, delete documents and view ingestion status.
- **FR-20:** Observability: log queries, retrieved chunks, latency, tokens, and errors for each request.
- **FR-21:** Health endpoint and index statistics (doc count, chunk count, last update).

---

## 5. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Latency** | p95 end-to-end first token < 3s (excluding cold start); retrieval p95 < 300ms. |
| **Accuracy** | Measured on an evaluation set: answer correctness, context precision/recall, citation faithfulness (see §7). |
| **Scalability** | Support ≥ 100k chunks; handle concurrent users (target configurable, default ≥ 50 concurrent sessions). |
| **Availability** | 99.5% for chat API (v1). |
| **Security** | Authentication for chat and admin endpoints; secrets via environment/secret manager; no logging of sensitive content by default. |
| **Privacy** | Document content never sent to third parties beyond the configured LLM provider; data retention policy defined. |
| **Compliance** | Respect provider data-processing terms; support PII redaction hooks if required. |
| **Portability** | Containerized; runs locally (docker-compose) and on a cloud target. |

---

## 6. Success Metrics

- **Retrieval:** context precision ≥ 0.8, context recall ≥ 0.85 on the eval set.
- **Answer quality:** faithfulness/groundedness ≥ 0.9; hallucination rate < 5% on eval set.
- **Deflection:** ≥ 70% of eval questions answered correctly without human escalation.
- **Performance:** p95 first-token latency < 3s.
- **Adoption:** weekly active users, messages/user (tracked once launched).

---

## 7. Evaluation & Testing

- **Golden set:** curated question → expected answer/source pairs, versioned in the repo.
- **Automated eval pipeline:** run retrieval + generation metrics (groundedness, answer relevance, citation correctness) on every ingestion or prompt change. Framework: e.g., RAGAS or equivalent.
- **Regression:** prompt/model changes must not degrade eval scores beyond an agreed tolerance.
- **Manual review:** sampled weekly review of live conversations.

---

## 8. Technical Approach (indicative architecture)

```
[Documents] → Ingestion (parse → chunk → embed) → Vector Index (+ metadata store)
                                                        │
User → Chat UI → API → Query (hybrid retrieve → rerank) │
                       ↓                                │
                 Prompt assembly ←──────────────────────┘
                       ↓
              LLM (streaming) → Answer + citations → UI
```

**Components**
- **Ingestion worker:** parsers, chunker, embedding client, index writer.
- **Retrieval service:** vector store (e.g., Qdrant/pgvector/FAISS), BM25/keyword search, fusion, reranker.
- **Generation service:** provider-agnostic LLM client, prompt templates, citation formatting.
- **API:** chat endpoint (SSE/streaming), conversation store, admin endpoints.
- **UI:** chat interface with source panels.
- **Eval harness:** golden-set runner integrated in CI.

*Final stack choices (vector DB, LLM provider, framework such as LangChain/LlamaIndex or custom) to be decided in Technical Design.*

---

## 9. Milestones

| Phase | Scope | Outcome |
|---|---|---|
| **M1 — Spike** | End-to-end hello-world: ingest 5 docs, ask 1 question, cited answer (CLI) | Validates feasibility |
| **M2 — MVP** | Full ingestion pipeline, hybrid retrieval, streaming chat UI, citations | Usable demo on internal KB |
| **M3 — Quality** | Reranker, abstention, eval harness, prompt tuning, ingestion SLA | Meets §6 metrics |
| **M4 — Hardening** | Auth, observability, admin UI, deployment, load test | Production pilot |

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Hallucination despite RAG | Trust loss | Strict grounding prompts, abstention threshold, citation checks, eval gate |
| Poor retrieval on messy docs | Wrong answers | Better parsers/chunking, hybrid search, reranker, doc-quality guidelines |
| Stale index after doc updates | Outdated answers | Incremental ingestion, index freshness SLA, admin status view |
| Latency from rerank/multi-step | UX degradation | Parallel retrieval, small/fast reranker, streaming, cache frequent queries |
| LLM cost | Budget overrun | Token budgets, prompt compression, caching, model tiering |
| Vendor lock-in | Flexibility | Provider-abstraction layer for LLM and vector store |

---

## 11. Open Questions

1. **Knowledge base scope:** which document sets, formats, and approximate volume?
2. **LLM provider/model** preference and data-privacy constraints?
3. **Auth model:** anonymous, SSO, API keys, or embedded in an existing app?
4. **Deployment target:** local, cloud (which provider), or on-prem?
5. **Language:** English only, or multilingual?
6. **Citation UX:** inline footnotes, side panel, or both?
7. **Retention:** how long are conversations and documents stored?

---

## 12. Appendix

- **Glossary:** *chunk* — a retrievable unit of text; *grounded answer* — an answer supported only by retrieved context; *abstention* — explicit refusal when context is insufficient.
- **Related docs:** `Docs/ProblemStatement.txt` (to be populated), Technical Design (TBD).
