---
title: Zania RAG Bot
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Zania RAG Bot

Question-answering API over PDF, JSON, or XLSX documents. FastAPI + LangChain + Chroma + OpenAI `gpt-4o-mini` (or any OpenAI-compatible provider — Groq, Ollama, vLLM).

Built for the Zania coding challenge.

## 🚀 Live demo

**https://indraniinapakolla-zania-rag-bot.hf.space** — chat UI with `samples/spec_kb.json` already attached, so you can ask the spec's sample questions in one click. Currently powered by Llama 3.1 8B via Groq (free tier, OpenAI-compatible) because the spec's OpenAI key was rate-limited; the canonical default in the codebase is still `gpt-4o-mini`.

GitHub: https://github.com/Indrani-19/zania-rag-bot

## Spec compliance — at a glance

| Spec line | Status | Where |
|---|---|---|
| QA bot powered by an LLM | ✅ | `app/core/qa.py` |
| LangChain framework | ✅ | `langchain-core`, `langchain-text-splitters` |
| FastAPI / Flask / Django | ✅ | FastAPI, `app/main.py` |
| Input: JSON questions file + PDF/JSON doc | ✅ | `POST /qa` accepts both as multipart |
| Output: JSON pairing each question with its answer | ✅ | List of `{question, answer}` — spec example was malformed JSON; the list form preserves duplicate questions and is documented in [Decisions](#decisions) |
| OpenAI `gpt-4o-mini` | ✅ | Default at `app/config.py:19` |
| VectorDB | ✅ | Chroma, persisted to `./chroma_db` |
| Production-quality code: README + tests + deps | ✅ | This README · **90 unit tests** · eval harness · structured errors · cost tracking |
| ≤ $5 budget | ✅ | `COST_HARD_CAP_USD=4.0`, hard-aborts past it |
| No keys committed | ✅ | `.env` is gitignored |
| `gpt-4o-mini` only (no GPT-4 / 16K models) | ✅ | Never references larger models |

## Beyond spec — what I added

The spec is fully satisfied by `POST /qa`. The items below are additive — they make the bot easier to demo, easier to operate, and harder to misuse, without changing the spec contract.

- **Browser chat UI at `/`** — single-page, no build step, ships in the same Docker image. Sidebar persists chats in `localStorage`; per-chat delete on hover.
- **`.xlsx` ingestion** — the spec's "Sample JSON file" link is actually a Google Sheet; handling xlsx natively skips the manual export step. Each sheet → "page", each row → coherent `[Sheet, row N]` key/value block so questions and their answers stay in the same chunk.
- **Stateful endpoints** — `POST /documents` + `POST /documents/{id}/questions` let a client re-query a doc without re-embedding (saves cost and latency on the second question onward).
- **Server-side intent routing** — greetings / help / low-signal input answer in <1 ms with canned responses (no LLM call). Summary and listing requests use broad-fetch retrieval (~30 chunks across the doc) with synthesis-friendly prompts. Factual Q&A stays on the original strict path. See `app/core/qa.py`.
- **Two-layer hallucination guard** — strict system prompt that says "answer only from context" *plus* a similarity-floor short-circuit (LLM is never called when the best chunk's similarity is below the floor — refusals stay consistent and cost stays predictable).
- **Eval harness with LLM-as-judge** — `eval/` runs a labeled set against the same code path the API serves, scoring deterministic substring/refusal checks plus faithfulness, relevance, and refusal precision/recall. Captured scorecard: **faithfulness 90%, relevance 90%, refusal recall 100%** on the SOC2 PDF. Exits non-zero on regression — ready to gate CI.
- **Provider-swap via `OPENAI_BASE_URL`** — same code path runs on OpenAI, Ollama (local dev), Groq (the live demo), or any other OpenAI-compatible API. One env var, no code change.
- **Local sentence-transformers embedding fallback** — for providers like Groq that don't host embeddings. Set `EMBEDDING_PROVIDER=local` and the bot embeds in-process with `sentence-transformers/all-MiniLM-L6-v2` (~90 MB model, $0/call). Activated automatically on the live demo.
- **Demo preload** — `DEMO_PRELOAD=true` indexes `samples/spec_kb.json` at startup under a stable `document_id`; the chat UI auto-attaches it on first visit. Reviewer hits the URL → can ask questions immediately, zero friction.
- **Hugging Face Spaces deploy config** — repo + Dockerfile + `DEPLOY.md` recipe goes from `git push` to public URL in ~5 minutes on free infrastructure (Groq for chat + sentence-transformers for embeddings).
- **Structured problem-JSON errors** with stable `type` codes (`llm_quota_exhausted`, `scanned_pdf`, `ingestion_error`, `llm_unreachable`, ...) — clients branch on `type`, not English error strings.
- **Cost tracker** — every LLM/embedding call is logged to `cost_log.jsonl` with model, token counts, and USD. Hard cap aborts requests that would push spend past `COST_HARD_CAP_USD`.

## Quickstart

```bash
git clone https://github.com/Indrani-19/zania-rag-bot.git
cd zania-rag-bot
cp .env.example .env       # paste your OpenAI key into OPENAI_API_KEY
```

Then **either** Docker:

```bash
docker compose up --build
```

**or** local Python 3.12:

```bash
make install               # create venv + install deps
make run                   # uvicorn on :8000
```

Service is at `http://localhost:8000` — chat UI at `/`, Swagger UI at `/docs`.

### Run on Llama via Ollama (no OpenAI key)

If your OpenAI key is rate-limited, the same code path runs on local Llama. `OPENAI_BASE_URL` swaps the provider for both chat and embeddings.

```bash
brew install ollama && ollama serve &
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

In `.env`:

```bash
OPENAI_API_KEY=ollama                     # ignored by Ollama, but the client requires it set
OPENAI_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text
```

Then `rm -rf chroma_db` (embedding dim changes 1536 → 768) and `make run`. Cost tracker reports `$0.00` for non-OpenAI base URLs by design.

### Public demo (Hugging Face Spaces + Groq)

See [`DEPLOY.md`](DEPLOY.md). TL;DR: chat = Groq Llama 3.1 (free, OpenAI-compatible), embeddings = sentence-transformers in-process, app = Docker Space. Set `DEMO_PRELOAD=true` so the spec's sample KB is pre-indexed.

### Smoke test against the spec's sample inputs

```bash
curl -L -o /tmp/soc2.pdf https://productfruits.com/docs/soc2-type2.pdf

# PDF run
curl -X POST "http://localhost:8000/qa?verbose=true" \
  -F "document=@/tmp/soc2.pdf" \
  -F "questions=@samples/spec_questions.json"

# JSON KB run (same questions, different doc → different answers)
curl -X POST "http://localhost:8000/qa?verbose=true" \
  -F "document=@samples/spec_kb.json" \
  -F "questions=@samples/spec_questions.json"
```

Captured output for both at [`samples/example_outputs.md`](samples/example_outputs.md). The PDF answers AWS / Europe; the JSON KB answers GCP / US — same bot, same questions, different docs. That's grounded retrieval working, not training-data recall.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /qa` | One-shot: upload a document + questions, get answers |
| `POST /documents` | Index a document, return a `document_id` |
| `POST /documents/{id}/questions` | Re-query an indexed document (no re-embedding cost) |
| `DELETE /documents/{id}` | Remove a document and its embeddings |
| `GET /health` | Liveness probe |
| `GET /` | Browser chat UI (not in OpenAPI schema) |
| `GET /demo` | Returns the preloaded `document_id` when `DEMO_PRELOAD=true`; 404 otherwise |

### Response shape

`POST /qa` and `POST /documents/{id}/questions` return a list of question/answer pairs. Add `?verbose=true` for source page snippets and the retrieval similarity score:

```json
[
  {
    "question": "Which cloud providers do you rely on?",
    "answer": "AWS.",
    "sources": [{"page": 12, "snippet": "...hosted on Amazon Web Services..."}],
    "retrieval_score": 0.81
  }
]
```

`POST /documents` returns `{document_id, chunk_count, estimated_cost_usd}`. `DELETE` returns `{document_id, deleted_chunks}`. Full schemas at `/docs`.

### Errors

All failures return a structured problem response: `{type, title, status, detail}`.

| Status | `type` | When |
| --- | --- | --- |
| 402 | `llm_quota_exhausted` / `budget_exceeded` | Provider quota / local cap reached |
| 422 | `validation_error` / `ingestion_error` / `scanned_pdf` / `empty_pdf` | Bad payload, unsupported file, image-only PDF, zero-page PDF |
| 429 | `llm_rate_limited` | Provider per-minute rate limit |
| 502 / 503 | `llm_upstream_error` / `llm_unreachable` / `llm_auth_failed` | Upstream HTTP error / connection failure / bad API key |

## Architecture

```
PDF / JSON / XLSX  →  Loader  →  Chunker  →  Embeddings  →  Chroma (persisted)
                                                                ↓
                             Question  →  Top-k retrieval  →  Similarity floor
                                                                ↓
                                  (above)  →  intent classify  →  LLM  →  Answer
                                  (below)  →  refusal sentence  (LLM never called)
```

Every LLM/embedding call is cost-tracked and budget-capped. Each answer is cited back to the chunk(s) it came from when `verbose=true`.

## Decisions

| Decision | Why |
| --- | --- |
| Stateful endpoints + `/qa` convenience | Stateful is right for RAG; `/qa` matches the spec's single-bundle example |
| Output is a list of `{question, answer}` objects | Spec example was malformed JSON; a list preserves duplicate questions |
| `retrieval_score`, not `confidence` | Cosine similarity measures *retrieval relevance*, not *answer correctness* |
| JSON docs flattened to `key.path: value` lines, then chunked | Preserves structure; arrays past 50 elements are summarized to bound chunk count |
| XLSX rows kept whole as `[Sheet, row N]` blocks | Question + answer columns stay in the same chunk → retrieval lands on the full Q&A pair |
| Chroma, persisted to disk | Zero-setup, no cloud account, survives restarts |
| `uvicorn --workers 1` | Chroma's persistent client isn't safe for multi-process writes |
| Scanned PDFs detected and rejected with 422 | `pypdf` can't OCR; failing loudly beats mysterious empty answers |
| LLM mocked in tests by default | Avoids burning the budget on every commit |
| Provider via `OPENAI_BASE_URL` | One env var routes the whole pipeline through any OpenAI-compatible provider (Ollama, Groq, vLLM, etc.) |
| Intent routing for greetings/summary/listing | Strict factual prompt + canned-reply fast paths give the right behavior for each question shape without compromising refusal quality |
| No auth | Coding-challenge scope — documented, not half-implemented |

## Evaluation harness

A QA bot you can't measure is a QA bot you can't trust. `eval/` ships a labeled set + LLM-as-judge that runs the same code path the API serves.

```bash
make eval          # runs eval/datasets/soc2.json against /tmp/soc2.pdf
```

Scores deterministic substring/refusal checks, faithfulness (claim → context grounding), relevance (on-topic), and refusal precision/recall. Configurable thresholds; CLI exits non-zero on regression. Captured scorecard at [`samples/eval_scorecard.md`](samples/eval_scorecard.md): **faithfulness 90%, relevance 90%, refusal recall 100%** on the SOC2 PDF.

## Cost

Every LLM/embedding call is logged to `cost_log.jsonl`. `COST_HARD_CAP_USD` (default `$4`) aborts any request that would push spend over the cap. Typical end-to-end run on `gpt-4o-mini`: ~$0.005.

## Testing

```bash
make test          # 90 unit tests, all mocked, no API calls
make lint          # ruff
```

Coverage includes the refusal logic, intent classifiers, ingestion edge cases (scanned PDFs, malformed JSON, empty xlsx rows, missing headers), the full error-mapping table, and the chat UI route. CI runs both on every push (`.github/workflows/ci.yml`).

## Project layout

```
app/
  main.py        # FastAPI app, exception handlers, demo-preload startup hook
  config.py      # Pydantic settings (incl. embedding_provider, demo_preload)
  api/           # HTTP routes
  core/          # Ingestion, embedding (OpenAI + local), retrieval, QA, vector store
  models/        # Request/response schemas
  utils/         # Cost tracking
  static/        # index.html — chat UI mounted at GET /
eval/            # Labeled questions + scoring harness
samples/         # Spec sample inputs + captured outputs + eval scorecard
tests/           # 90 mocked unit tests
DEPLOY.md        # HF Spaces + Groq deploy recipe
```

## Extending to new file types

Currently supported: `.pdf`, `.json`, `.xlsx`. The pipeline is `bytes → text/pages → chunks → embeddings → Chroma` — adding a format is one loader.

**Pattern:**

1. Write `load_<format>(content: bytes) -> list[tuple[int, str]] | str` in `app/core/ingestion.py`. Return `(page_num, text)` tuples if the format has page semantics; otherwise a flat string.
2. Register the extension in the `ingest()` dispatcher.
3. Add a parametrized test in `tests/test_ingestion.py`.

**Library map for the common formats:**

| Format | Library | Notes |
| --- | --- | --- |
| `.docx` | `python-docx` | Iterate paragraphs; tables need explicit handling |
| `.csv` | stdlib `csv` | Same flatten-as-key:value treatment |
| `.html` | `beautifulsoup4` | `.get_text(separator='\n')` after stripping `<nav>`/`<footer>` |
| `.md` | none needed | Read as text; consider `MarkdownTextSplitter` for header-aware chunking |
| `.pptx` | `python-pptx` | One slide per "page" |
| `.eml` / `.mbox` | stdlib `email` | Subject + headers + body |
| Code (`.py`, `.ts`, ...) | none | Use `RecursiveCharacterTextSplitter.from_language(...)` for syntax-aware chunks |

**For scanned PDFs and images** (currently rejected with HTTP 422):

- Local: `pytesseract` + `pdf2image` for PDFs, `pytesseract` + `Pillow` for images
- Cloud: AWS Textract / Azure Document Intelligence / Google Document AI — much higher accuracy on real-world layouts

**For production at scale:** [`unstructured.io`](https://unstructured.io) — one library, 20+ formats, layout-aware extraction, table parsing, optional OCR. Replaces most of `app/core/ingestion.py` with a single `partition()` call.

## Scaling to production — what comes next

Coding-challenge submission, not a production deployment. Concrete gaps, ranked by what breaks first.

**Tier 1 — breaks first under load**

- `uvicorn --workers 1` ceiling — Chroma's persistent client isn't multi-process safe. Fix: Chroma server mode (or Qdrant / pgvector), then scale workers.
- Sync ingestion blocks the request thread on big files. Fix: background job queue; `POST /documents` returns a `job_id`.
- Top-k cosine misses cover-page facts and collapses on near-identical array items. Fix: hybrid BM25 + dense, optional cross-encoder rerank.
- Upload size check runs *after* the full read into memory. Fix: stream + check incrementally.

**Tier 2 — hardening before external customers**

- No auth → cost-bomb. Fix: per-tenant API keys + `slowapi` rate-limit middleware.
- All docs in one Chroma collection → one filter bug = cross-tenant leak. Fix: collection-per-tenant or row-level security.
- In-process cost tracker → multi-instance = N × budget. Fix: Redis/Postgres atomic counter.
- No data retention → GDPR risk. Fix: TTL + tenant-cascade delete.
- Pricing table goes stale silently. Fix: pull from versioned config or billing API.
- Each chat turn is independent → "tell me more" can't follow up. Fix: server-side LLM rewrite of the last 2–3 turns into a self-contained query, then run RAG.

**Tier 3 — operational scaffolding**

- No structured logs / metrics. Fix: JSON logs with `contextvars` `request_id`, Prometheus middleware.
- No LLM circuit breaker. Fix: `circuitbreaker` around chat/embed calls.
- No prompt versioning. Fix: prompt registry (Langfuse / PromptLayer / versioned YAML).
- Eval harness doesn't gate CI. Fix: CI step running `make eval` against a fixed doc.
- HF Spaces free tier loses Chroma between cold starts. Fix: persistent storage at `/data/chroma_db`, or move to a host with real volumes.

**Out of scope (deliberate)**

Streaming responses · multi-document corpus queries · OCR for scanned PDFs.

## License

MIT.
