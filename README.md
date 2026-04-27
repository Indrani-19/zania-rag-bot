# Zania RAG Bot

Question-answering API over PDF or JSON documents. FastAPI + LangChain + Chroma + OpenAI `gpt-4o-mini`.

Built for the Zania coding challenge.

## Quickstart

```bash
git clone https://github.com/Indrani-19/zania-rag-bot.git
cd zania-rag-bot
cp .env.example .env       # paste your OpenAI key into OPENAI_API_KEY
```

Then **either** Docker (one command):

```bash
docker compose up --build
```

**or** local Python 3.12:

```bash
make install               # create venv + install deps
make run                   # uvicorn on :8000
```

Service is at `http://localhost:8000` — Swagger UI at `/docs`.

### Smoke test against the spec's sample inputs

The challenge doc references a sample PDF and a sample JSON KB. Both are wired up:

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
| 402 | `llm_quota_exhausted` / `budget_exceeded` | OpenAI returned `insufficient_quota` / local cap reached |
| 422 | `validation_error` / `ingestion_error` / `scanned_pdf` / `empty_pdf` | Bad payload, unsupported file, image-only PDF, zero-page PDF |
| 429 | `llm_rate_limited` | OpenAI per-minute rate limit |
| 502 / 503 | `llm_upstream_error` / `llm_unreachable` / `llm_auth_failed` | Upstream HTTP error / connection failure / bad API key |

## Architecture

```
PDF / JSON  →  Loader  →  Chunker  →  Embeddings  →  Chroma (persisted)
                                                          ↓
                       Question  →  Top-k retrieval  →  Similarity floor
                                                          ↓
                            (above)  →  gpt-4o-mini  →  Answer
                            (below)  →  refusal sentence  (LLM never called)
```

Two-layer hallucination guard: strict system prompt + similarity-floor short-circuit. Every LLM/embedding call is cost-tracked and budget-capped.

## Decisions

| Decision | Why |
| --- | --- |
| Stateful endpoints + `/qa` convenience | Stateful is right for RAG; `/qa` matches the spec's single-bundle example |
| Output is a list of `{question, answer}` objects | Spec example was malformed JSON; a list preserves duplicate questions |
| `retrieval_score`, not `confidence` | Cosine similarity measures *retrieval relevance*, not *answer correctness* |
| JSON docs flattened to `key.path: value` lines, then chunked | Preserves structure; arrays past 50 elements are summarized to bound chunk count |
| Chroma, persisted to disk | Zero-setup, no cloud account, survives restarts |
| `uvicorn --workers 1` | Chroma's persistent client isn't safe for multi-process writes |
| Scanned PDFs detected and rejected with 422 | `pypdf` can't OCR; failing loudly beats mysterious empty answers |
| LLM mocked in tests by default | Avoids burning the budget on every commit |
| Provider via `OPENAI_BASE_URL` | One env var routes the whole pipeline through any OpenAI-compatible provider (Ollama, Groq, vLLM, etc.) |
| No auth | Coding-challenge scope — documented, not half-implemented |

## Evaluation harness

A QA bot you can't measure is a QA bot you can't trust. `eval/` ships a labeled set + LLM-as-judge that runs the same code path the API serves.

```bash
make eval          # runs eval/datasets/soc2.json against /tmp/soc2.pdf
```

Scores deterministic substring/refusal checks, faithfulness (claim → context grounding), relevance (on-topic), and refusal precision/recall. Configurable thresholds; CLI exits non-zero on regression. Captured scorecard at [`samples/eval_scorecard.md`](samples/eval_scorecard.md): faithfulness 90%, relevance 90%, refusal recall 100% on the SOC2 PDF.

## Cost

Every LLM/embedding call is logged to `cost_log.jsonl`. `COST_HARD_CAP_USD` (default `$4`) aborts any request that would push spend over the cap. Typical end-to-end run: ~$0.005.

## Testing

```bash
make test          # 34 unit tests, all mocked, no API calls
make lint          # ruff
```

CI runs both on every push (`.github/workflows/ci.yml`).

## Project layout

```
app/
  main.py        # FastAPI app + exception handlers
  config.py      # Pydantic settings
  api/           # HTTP routes
  core/          # Ingestion, embedding, retrieval, QA, vector store
  models/        # Request/response schemas
  utils/         # Cost tracking
eval/            # Labeled questions + scoring harness
samples/         # Spec sample inputs + captured outputs
tests/           # Mocked unit tests
```

## Extending to new file types

Currently supported: `.pdf` and `.json`. The pipeline is `bytes → text/pages → chunks → embeddings → Chroma` — adding a format is one loader.

**Pattern:**

1. Write `load_<format>(content: bytes) -> list[tuple[int, str]] | str` in `app/core/ingestion.py`. Return `(page_num, text)` tuples if the format has page semantics; otherwise a flat string.
2. Register the extension in the `ingest()` dispatcher.
3. Add a parametrized test in `tests/test_ingestion.py`.

**Library map for the common formats:**

| Format | Library | Notes |
| --- | --- | --- |
| `.docx` | `python-docx` | Iterate paragraphs; tables need explicit handling |
| `.xlsx` | `openpyxl` | One sheet per "page"; rows → flat key:value lines |
| `.csv` | stdlib `csv` | Same flatten-as-key:value treatment |
| `.html` | `beautifulsoup4` | `.get_text(separator='\n')` after stripping `<nav>`/`<footer>` |
| `.md` | none needed | Read as text; consider `MarkdownTextSplitter` for header-aware chunking |
| `.pptx` | `python-pptx` | One slide per "page" |
| `.eml` / `.mbox` | stdlib `email` | Subject + headers + body |
| Code (`.py`, `.ts`, ...) | none | Use `RecursiveCharacterTextSplitter.from_language(...)` for syntax-aware chunks |

**For scanned PDFs and images** (currently rejected with HTTP 422):

- Local: `pytesseract` + `pdf2image` for PDFs, `pytesseract` + `Pillow` for images
- Cloud: AWS Textract / Azure Document Intelligence / Google Document AI — much higher accuracy on real-world layouts

**For production at scale:** [`unstructured.io`](https://unstructured.io) is the right answer — one library, 20+ formats, layout-aware extraction, table parsing, optional OCR. It would replace most of `app/core/ingestion.py` with a single `partition()` call.

## Scaling to production

This is a coding-challenge submission, not a production deployment. Code patterns are right; operational scaffolding isn't. Concrete gaps, ranked by what breaks first:

**Tier 1 — breaks first under load**

- `uvicorn --workers 1` is a hard ceiling. Chroma's persistent client isn't multi-process safe. Fix: Chroma server mode (or Qdrant / pgvector), then scale workers.
- Sync ingestion blocks the request thread on big PDFs. Fix: background job queue (Celery / RQ / ARQ); `POST /documents` returns a `job_id`.
- Top-k cosine misses cover-page facts and collapses on near-identical array items. Fix: hybrid BM25 + dense, optionally re-rank with a cross-encoder.
- Upload size check runs *after* the full file is read into memory. Fix: stream + check incrementally.

**Tier 2 — hardening before external customers**

- No auth → trivial cost-bomb. Fix: API keys per tenant + `slowapi` rate-limit middleware.
- All docs in one Chroma collection → one filter bug = cross-tenant leak. Fix: collection-per-tenant or row-level security.
- Cost tracker is in-process → multi-instance = N × budget. Fix: Redis/Postgres counter with atomic increments.
- No data retention → uploaded docs persist forever. GDPR risk. Fix: TTL + `DELETE /tenants/{id}` cascade.
- Pricing table goes stale silently. Fix: pull from versioned config or OpenAI billing API.

**Tier 3 — operational scaffolding**

- No structured logs / metrics. Fix: JSON logs with `contextvars`-propagated `request_id`, Prometheus middleware.
- No circuit breaker on the LLM provider. Fix: `circuitbreaker` lib around `chat_completion` / `embed_texts`.
- No prompt versioning. Fix: prompt registry (Langfuse / PromptLayer / versioned YAML).
- Eval harness exists but isn't gating CI. Fix: CI step running `make eval` against a fixed doc.

**Out of scope (deliberate, not scaling)**

Streaming responses · multi-document corpus queries · conversation history · OCR fallback for scanned PDFs.

**Effort estimate:** ~1 week for Tier 1, ~2 weeks for Tier 2, ~1 week minimum for Tier 3. Roughly **2-3 weeks for an internal beta, ~6 weeks for external customers**.

## License

MIT.
