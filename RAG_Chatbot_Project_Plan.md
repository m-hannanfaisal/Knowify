# Advanced RAG Chatbot — Full Project Plan

**Goal:** Build a production-grade, agentic RAG chatbot for your GitHub portfolio that demonstrates senior-level skill in retrieval, evaluation, observability, and deployment — not just "LangChain + OpenAI + Streamlit."

---

## 1. Architecture Overview

Base flow (from your diagram, extended):

```
User Query
   │
Load Conversation History (short-term, current session)
   │
Retrieve Relevant Long-Term Memories (vector search on user_id + query)
   │
Query Rewriter (LLM Call 1) ── uses conversation context + long-term memory
   │
Orchestrator (LLM Call 2) ── does this need RAG?
   │
   ├── No ──────────────────────────────► Main LLM Call (direct answer)
   │
   └── Yes
       │
       Hybrid Retrieval (Dense + BM25, ChromaDB/Qdrant)
       │
       Cross-Encoder Rerank (+ RRF fusion)
       │
       Relevance Evaluator (LLM Call 3) ── sufficient & relevant?
       │
       ├── No → Retry Limit Check ──► Query Rewriter (LLM Call, feedback-driven) ──► loop back to Retrieval
       │              │
       │              └── retries exhausted → Safe Response ("not enough information")
       │
       └── Yes → Main LLM Call (grounded generation + citations)
                       │
                Return Response to User (streamed)
                       │
                Save Query + Response (Conversation Memory, this session)
                       │
                Memory Extraction (LLM Call) ── pull durable facts worth remembering
                       │
                Write to Long-Term Memory Store (async, non-blocking)
                       │
                Log trace + metrics (async, non-blocking)
```

**Single orchestrator, multi-step** — one LLM/agent performing sequential specialized calls (rewrite → route → evaluate → generate → extract memory), not separate handoff-based agents. This keeps the system easier to trace, eval, and debug while still being genuinely agentic in its control flow.

Every box above is a **traced span** — this is what makes the system debuggable and demo-able.

---

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Backend API | FastAPI | async, streaming support, industry standard |
| Orchestration | LangGraph (or custom state machine) | shows agentic control-flow understanding, not just chains |
| Vector DB | Qdrant or ChromaDB | hybrid search support, easy self-host or cloud; also hosts a separate namespace/collection for long-term memory |
| Sparse search | BM25 (rank_bm25 or Elasticsearch) | hybrid retrieval |
| Reranker | bge-reranker-v2 or Cohere Rerank | real relevance scoring beyond RRF |
| LLM | Claude/GPT-4o via API, swappable via config | flexibility, cost control |
| Eval framework | RAGAS | faithfulness, relevancy, context precision/recall |
| Observability | LangSmith or Arize Phoenix + custom structured logs | tracing, debugging, metrics |
| Cache | Redis | repeated-query caching, session store |
| Frontend | Next.js/React (streaming UI) or Streamlit for MVP | React reads more "product," Streamlit is faster to ship |
| Auth | JWT / Clerk / Supabase Auth | multi-user, rate limiting |
| Deployment | Docker + Render/Fly.io/Railway | live demo link, non-negotiable |
| CI/CD | GitHub Actions | run eval suite + tests on every PR |

---

## 3. Logging & Observability (detailed)

**a) Structured application logs**
- Use `structlog` or Python `logging` with JSON formatter.
- Every request gets a `trace_id`. Log at each pipeline stage:
  - stage name, duration_ms, input/output size, model used, token counts, cost estimate.
- Ship logs to stdout (captured by Docker/host) — optionally forward to a lightweight log store (e.g. Loki, or just a Postgres table for the portfolio version).

**b) LLM tracing**
- Instrument each LLM call and retrieval step with LangSmith or Arize Phoenix (OpenTelemetry-based).
- This gives you a visual trace tree per conversation: rewrite → retrieve → rerank → evaluate → generate, each with latency and content.
- Use this in your demo video — "here's a trace showing the retry loop triggering because the first retrieval was insufficient" is a very strong portfolio moment.

**c) Metrics dashboard**
- Track: retrieval hit rate, average faithfulness score (RAGAS), retry rate, p50/p95 latency, cost per query, cache hit rate.
- Simple version: a `/metrics` endpoint + a small React/Streamlit dashboard page. Advanced version: Prometheus + Grafana.

**d) Error logging**
- Catch and log LLM API failures, retrieval timeouts, malformed outputs — with retry/backoff logic, not silent failures.

---

## 4. Evaluation Strategy

- Build a **golden test set**: 30–50 question/answer pairs with known correct sources, covering easy, ambiguous, and out-of-scope queries.
- Run RAGAS metrics: faithfulness, answer relevancy, context precision, context recall, answer correctness.
- Add these as automated checks in CI — a PR that drops faithfulness below a threshold should fail the build (with tolerance bands, since LLM outputs are non-deterministic).
- Log eval scores over time so you can show a "quality improved as I iterated" chart — great portfolio narrative.

---

## 5. Full Functionality List (priority-tiered)

### 🟢 Must-Have (core differentiators, build these first)
- Long-term persistent memory: LLM-based fact extraction per session, stored per-user in a vector namespace, retrieved and injected on future queries (remembers across sessions, not just within one chat)
- Multi-format ingestion: PDF, DOCX, HTML, CSV, Excel, JSON, Markdown/TXT
- Image understanding (OCR + vision captioning) for uploaded screenshots/charts/scanned docs
- Web search fallback tool (agent decides: internal KB vs. web)
- Query rewriting with conversation memory
- Adaptive routing (RAG vs. no-RAG vs. web vs. tool call)
- Hybrid retrieval (dense + BM25) + cross-encoder rerank
- Self-correcting retry loop (Corrective RAG)
- Source citations (clickable, chunk/page-level)
- RAGAS evaluation + golden test set
- Structured logging + LLM tracing (LangSmith/Phoenix)
- Streaming responses
- Live deployed demo

### 🟡 Should-Have (strong signal, second wave)
- Memory consolidation/pruning (dedupe and merge overlapping long-term memories periodically)
- MCP client support (agent calls external MCP servers as tools instead of hardcoded integrations)
- MCP server mode (expose your RAG pipeline as an MCP tool others can plug into, e.g. Claude Desktop/Code)
- Multi-hop retrieval (combine info across multiple docs)
- Guardrails: prompt-injection resistance, PII redaction, refusal on out-of-scope queries
- Metrics dashboard (faithfulness trend, retrieval hit rate, latency, cost, cache hit rate)
- Redis caching (repeated queries + session store)
- Document management UI (upload, re-index, delete, chunk counts)
- Auth + per-user document libraries (multi-tenancy)

### 🔵 Nice-to-Have / Roadmap Items (mention in README even if unbuilt)
- Audio input (Whisper transcription → feeds into existing text pipeline, thin wrapper not core logic)
- Conversation export (PDF/Markdown)
- Feedback buttons (👍👎) logged into eval pipeline
- Conversation history browser/search
- API tool-calling beyond MCP (weather, calculator, stock price) if not already covered by MCP servers

---

## 6. Phased Roadmap

### Phase 0 — Setup (2–3 days)
- Repo scaffolding, Docker Compose (API + vector DB + Redis), CI skeleton.
- Pick and provision LLM + embedding provider.
- Basic structured logging in place from day one.

### Phase 1 — MVP RAG (1 week)
- Document ingestion pipeline: PDF/DOCX/CSV/HTML/JSON/Excel → chunk → embed → store.
- Naive retrieval → LLM generation, no rerank/eval yet.
- Simple frontend (Streamlit acceptable here) to prove the pipeline end-to-end.
- Basic logs per request.

### Phase 2 — Agentic/Advanced Retrieval + Long-Term Memory (2 weeks)
- Add query rewriter (conversation-aware).
- Add orchestrator (RAG vs. no-RAG vs. web search routing).
- Add hybrid search (dense + BM25) + cross-encoder rerank + RRF fusion.
- Add relevance evaluator + retry loop with feedback-driven rewriting.
- Add citations (source chunk + page/section linked in response).
- Add image understanding (OCR + vision captioning) for uploaded images.
- Add web search tool as a fallback source.
- Build long-term memory subsystem: per-user memory namespace, memory extraction LLM call after each session, memory retrieval step injected into the query pipeline.

### Phase 3 — Evaluation & Observability (1 week)
- Integrate RAGAS + golden test set.
- Integrate LangSmith/Phoenix tracing.
- Build metrics endpoint/dashboard.
- Add CI job that runs eval suite on every push.

### Phase 4 — MCP Integration (3–5 days)
- Build an MCP client so the orchestrator can call external MCP servers as tools (e.g. filesystem, web search, GitHub) instead of hardcoded integrations.
- Wrap your own RAG pipeline as an MCP server, so it can be plugged into Claude Desktop/Claude Code as an external tool.
- Document both directions clearly in the README — this is a rare, high-signal feature for 2026 portfolios.

### Phase 5 — Production Hardening (1–1.5 weeks)
- Streaming responses (SSE).
- Redis caching for repeated queries + session memory.
- Memory consolidation job (periodic dedupe/merge of long-term memory entries).
- Guardrails: prompt-injection resistance, PII redaction, refusal handling for out-of-scope questions.
- Auth + basic rate limiting + multi-tenancy.
- Real frontend polish (React/Next.js) if not already done.

### Phase 6 — Stretch Features (optional, time-permitting)
- Audio input: Whisper transcription step feeding into existing text pipeline (no new RAG logic needed).
- Conversation export, feedback buttons, conversation history browser.

### Phase 7 — Deployment & Portfolio Polish (3–5 days)
- Dockerize fully, deploy to Render/Fly.io/Railway with a live demo URL.
- Write README: architecture diagram, design decisions ("why RRF + reranker + retry loop over naive RAG", "why MCP"), setup instructions, eval results.
- Record a 2–3 min demo video/GIF showing a retry-loop trace, MCP tool call, and a citation-backed answer.
- Add badges (CI passing, license, live demo link) to repo top.

**Total estimated time: ~8–9 weeks part-time, ~4–5 weeks full-time (including long-term memory and MCP integration).**

---

## 7. Suggested Repo Structure

```
rag-chatbot/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routes
│   │   ├── orchestrator/    # LangGraph state machine
│   │   ├── retrieval/       # hybrid search, rerank
│   │   ├── evaluation/      # relevance evaluator, retry logic
│   │   ├── memory/          # short-term conversation history + long-term memory (extraction, storage, retrieval, pruning)
│   │   ├── logging/         # structured logging, tracing hooks
│   │   └── prompts/
│   ├── tests/
│   └── Dockerfile
├── eval/
│   ├── golden_set.json
│   └── run_ragas.py
├── frontend/
├── .github/workflows/ci.yml
├── docker-compose.yml
└── README.md
```

---

## 8. Portfolio Presentation Checklist

- [ ] Live deployed demo link at top of README
- [ ] Architecture diagram (your flowchart, cleaned up)
- [ ] "Why these design choices" section (RRF, reranker, retry loop, hybrid search, long-term memory, single-orchestrator vs. multi-agent)
- [ ] RAGAS eval results table/chart
- [ ] Demo video/GIF showing a trace + retry loop + citation + MCP tool call
- [ ] CI badge (passing eval + tests)
- [ ] Clear setup instructions (docker-compose up)
- [ ] License file
- [ ] "Roadmap" section in README listing unbuilt stretch items (audio input, export, feedback loop) to show forward thinking
