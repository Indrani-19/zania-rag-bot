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

A question-answering API over PDF, JSON, or XLSX documents. FastAPI + LangChain + Chroma, defaulting to OpenAI `gpt-4o-mini` (the spec model) but happy to run on any OpenAI-compatible provider — Groq, Ollama, vLLM — by changing one env var.

Built for the Zania coding challenge.

## Live demo

**https://indraniinapakolla-zania-rag-bot.hf.space**

The chat opens with `samples/spec_kb.json` already attached, so you can ask the spec's sample questions in one click. The live URL runs on Llama 3.1 via Groq because the OpenAI key was rate-limited; the default in the codebase is still `gpt-4o-mini`.

## Quickstart

```bash
git clone https://github.com/Indrani-19/zania-rag-bot.git
cd zania-rag-bot
cp .env.example .env        # paste your OpenAI key
docker compose up --build   # or: make install && make run
```

Service comes up on `http://localhost:8000` — chat UI at `/`, Swagger at `/docs`.

### Smoke test against the spec inputs

```bash
curl -L -o /tmp/soc2.pdf https://productfruits.com/docs/soc2-type2.pdf

curl -X POST "http://localhost:8000/qa?verbose=true" \
  -F "document=@/tmp/soc2.pdf" \
  -F "questions=@samples/spec_questions.json"
```

Captured output for both the spec PDF and the JSON KB at [`samples/example_outputs.md`](samples/example_outputs.md). The PDF answers AWS / Europe; the JSON answers GCP / US — same bot, same questions, different docs. That's grounded retrieval working, not training-data recall.

### Other providers

`OPENAI_BASE_URL` swaps the provider for both chat and embeddings (any OpenAI-compatible API works). [`DEPLOY.md`](DEPLOY.md) has the Hugging Face Spaces + Groq recipe used for the live demo.

## What I built

The spec asks for `POST /qa`: document + questions in, JSON pairing each question to its answer out. That works. A few things felt worth adding beyond the spec:

- **Browser chat UI at `/`** — single-page, no build step, ships in the same Docker image. Sidebar persists chats in localStorage.
- **Stateful endpoints** alongside `/qa`. `POST /documents` returns a `document_id`; `POST /documents/{id}/questions` re-queries without re-embedding.
- **Intent routing.** Greetings, help, and low-signal input answer in <1ms with no LLM call. Summary and listing requests broaden retrieval (~30 chunks) and use synthesis-friendly prompts. Factual questions stay on the strict path.
- **Two-layer hallucination guard** — strict system prompt that says "answer only from context", plus a similarity-floor short-circuit that skips the LLM entirely when the best chunk's similarity is too low.
- **Eval harness** with LLM-as-judge scoring. Captured scorecard at [`samples/eval_scorecard.md`](samples/eval_scorecard.md): 90% faithfulness, 100% refusal recall on the SOC2 PDF.
- **`.xlsx` support** — the spec's "Sample JSON file" is actually a Google Sheet; handling xlsx directly saves the export step.
- **Cost tracking** with a hard cap. Every call is logged to `cost_log.jsonl`; `COST_HARD_CAP_USD` aborts requests that would push spend past the cap.
- **Structured problem-JSON errors** with stable `type` codes — clients branch on `type`, not English error strings.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /qa` | Spec-shaped one-shot: document + questions in, answers out |
| `POST /documents` | Index a doc, return a `document_id` |
| `POST /documents/{id}/questions` | Re-query an indexed doc |
| `DELETE /documents/{id}` | Remove a doc |
| `GET /health` | Liveness |
| `GET /` | Chat UI |

Both Q&A endpoints return a list of `{question, answer}` objects. Add `?verbose=true` for source page snippets and the retrieval similarity score:

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

Errors are `{type, title, status, detail}` with stable type codes (`scanned_pdf`, `llm_quota_exhausted`, `ingestion_error`, …). Full schemas at `/docs`.

## Architecture

```
PDF / JSON / XLSX → Loader → Chunker → Embeddings → Chroma (persisted)
                                                       ↓
                       Question → Top-k retrieval → Similarity floor
                                                       ↓
                            (above) → intent classify → LLM → Answer
                            (below) → fixed refusal sentence
```

## Decisions worth calling out

- **Output is a list of `{question, answer}`** objects, not a dict. The spec's example was malformed JSON; a list preserves duplicate questions.
- **`retrieval_score`, not `confidence`** — cosine similarity tells you about retrieval relevance, not answer correctness.
- **JSON flattened to `key.path: value` lines** before chunking. Arrays over 50 elements are summarized so one giant array can't blow up into 1000 chunks.
- **XLSX rows kept whole** as `[Sheet, row N]` blocks — question and answer columns stay in the same chunk.
- **Scanned PDFs rejected with 422.** `pypdf` can't OCR; failing loudly beats silently empty answers.
- **Provider via `OPENAI_BASE_URL`.** One env var, no code change, swap to Groq / Ollama / vLLM.
- **No auth.** Coding-challenge scope; documented, not half-implemented.

## Tests, eval, cost

```bash
make test                  # full suite — unit + integration + regression, mocked, no API calls
make regression            # offline pipeline regression suite only
make eval                  # eval harness against /tmp/soc2.pdf (absolute thresholds)
make eval-check            # eval + compare to eval/baseline.json (regression gate)
make eval-update-baseline  # accept current run as the new baseline
make lint                  # ruff
```

**Three layers:**

1. **Unit + integration** (~100 tests, 82% line coverage on `app/`). Unit tests cover ingestion, intent classifiers, refusal logic, cost tracker, eval harness, observability. Integration tests hit the full FastAPI stack via `TestClient` — happy path, the full error-mapping table (402/422/503/504/404), oversized uploads, deterministic `document_id`.
2. **Pipeline regression** (`tests/test_regression_pipeline.py`, 14 cases). Frozen contracts on `answer_question()` over the full real stack — ingestion → Chroma → intent → retrieval → similarity floor → prompt routing — with mocked LLM and a deterministic 256-dim hashed bag-of-words embedder. Catches anyone who breaks intent routing, the floor short-circuit, or the listing/summary prompts. Runs in CI; no OpenAI calls.
3. **Eval regression gate** (`make eval-check`). Runs the labeled question set against real OpenAI, then compares per-metric scores (deterministic / faithfulness / relevance / refusal precision / refusal recall) to `eval/baseline.json` with a configurable tolerance (default ±5pp). Catches the case where a prompt or model change leaves absolute thresholds intact but quietly drops 10 points off faithfulness. Run after intentional changes with `make eval-update-baseline` to accept the new scores.

A typical eval run on `gpt-4o-mini` costs ~$0.005.

## Project layout

```
app/
  main.py        # FastAPI app + demo-preload startup
  config.py      # Pydantic settings
  api/           # HTTP routes
  core/          # Ingestion, embeddings, retrieval, QA, vector store
  models/        # Request/response schemas
  utils/         # Cost tracking
  static/        # index.html — chat UI mounted at GET /
eval/            # Labeled questions + scoring harness
samples/         # Spec sample inputs + captured outputs + eval scorecard
tests/           # Mocked unit tests
DEPLOY.md        # HF Spaces + Groq deploy recipe
```

## What's deliberately not done

Production scaffolding isn't here yet. The biggest gaps, ranked by what breaks first:

**Tier 1 — breaks first under load**
- `uvicorn --workers 1` is a ceiling; Chroma's persistent client isn't multi-process safe. Fix: Chroma server mode (or Qdrant / pgvector), then scale workers.
- Sync ingestion blocks the request thread. Fix: background job queue; `POST /documents` returns a `job_id`.
- Top-k cosine misses cover-page facts and collapses on near-identical array items. Fix: hybrid BM25 + dense, optional cross-encoder rerank.

**Tier 2 — hardening before external customers**
- No auth → cost-bomb. Fix: per-tenant API keys + `slowapi` rate-limit middleware.
- All docs in one Chroma collection. Fix: collection-per-tenant or row-level security.
- In-process cost tracker → multi-instance = N × budget. Fix: Redis/Postgres atomic counter.
- No data retention → GDPR risk. Fix: TTL + tenant-cascade delete.
- Each chat turn is independent — "tell me more" can't follow up. Fix: server-side LLM rewrite of the last 2–3 turns into a self-contained query, then run RAG.

**Tier 3 — operational scaffolding**
- No structured logs / metrics. Fix: JSON logs with `contextvars` `request_id`, Prometheus middleware.
- No LLM circuit breaker. Fix: `circuitbreaker` around chat/embed calls.
- No prompt versioning. Fix: prompt registry (Langfuse / PromptLayer / versioned YAML).
- Eval harness doesn't gate CI. Fix: CI step running `make eval` against a fixed doc.
- HF Spaces free tier loses Chroma between cold starts; demo preload re-indexes the sample doc but user uploads don't survive a restart.

**Out of scope** — streaming responses · multi-document corpus queries · OCR for scanned PDFs.

## License

MIT.
