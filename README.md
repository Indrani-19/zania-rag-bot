# Zania RAG Bot

A document question-answering API powered by Retrieval-Augmented Generation. Upload a PDF or JSON document, then ask one or more questions against its contents.

Built for the Zania coding challenge.

> **Quick start:** `make install && make test`. To run live: `cp .env.example .env`, fill in your `OPENAI_API_KEY`, then `make run` (or `make docker-up`).

## Quickstart

```bash
git clone <repo-url>
cd zania-rag-bot
cp .env.example .env       # add your OpenAI API key
docker compose up --build
```

Service runs on `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /qa` | One-shot: upload a document + questions in a single request, get answers back. (Convenience endpoint matching the spec example.) |
| `POST /documents` | Upload a document for indexing. Returns a `document_id`. |
| `POST /documents/{document_id}/questions` | Ask one or more questions against an already-indexed document. |
| `DELETE /documents/{document_id}` | Remove a document and its embeddings. |
| `GET /health` | Liveness probe. |

### Output shape

The base output matches the spec example (a list of question/answer pairs):

```json
[
  {"question": "What is X?", "answer": "X is..."},
  {"question": "What is Y?", "answer": "Y is..."}
]
```

Pass `?verbose=true` to also receive `sources` (the page snippets the answer was grounded in) and `retrieval_score` (cosine similarity of the best-matching chunk):

```json
[
  {
    "question": "What is X?",
    "answer": "X is...",
    "sources": [{"page": 3, "snippet": "..."}],
    "retrieval_score": 0.82
  }
]
```

## Architecture

```
PDF / JSON  →  Loader  →  Chunker  →  Embeddings (text-embedding-3-small)  →  Chroma (persisted to disk)
                                                                                   ↓
                                                              Question  →  Top-k retrieval  →  Similarity floor check
                                                                                                       ↓
                                                                         (if above floor)  →  gpt-4o-mini  →  Answer
                                                                         (if below floor)  →  "I don't have enough information"
```

- **Stateful by design:** documents are embedded once and queried many times. Re-embedding per request would burn the OpenAI budget and isn't how RAG works in practice.
- **Hallucination guard runs in two places:** a strict system prompt ("answer only from context, otherwise say you don't know") *and* a similarity-floor short-circuit that skips the LLM entirely when no chunk is sufficiently relevant.
- **Cost-capped:** every LLM and embedding call is logged to `cost_log.jsonl`. A hard cap (`COST_HARD_CAP_USD`, default $4) raises before exceeding budget so the OpenAI key is never drained.

## Decisions & Assumptions

The spec left several things unspecified or ambiguous. Each call below is intentional and defensible:

| Area | Decision | Reasoning |
| --- | --- | --- |
| API style | Stateful (`/documents` + `/questions`) **plus** a convenience `/qa` endpoint | Stateful is correct for RAG (embed once, query many). The `/qa` convenience matches the spec's single-bundle example so a reviewer's smoke test works either way. |
| Output shape | Base shape is `[{"question", "answer"}]` (matches spec example). Verbose mode adds `sources` and `retrieval_score`. | The spec example was malformed JSON (`[]` brackets but `:` dict syntax). A list-of-objects preserves duplicate questions; a dict can't. Verbose extras are opt-in to avoid surprising the spec contract. |
| `retrieval_score`, not `confidence` | Honestly named | Cosine similarity measures *retrieval relevance*, not *answer correctness*. Calling it "confidence" would be misleading. |
| JSON document handling | Recursively flatten to `key.path: value` strings, then chunk like text. Arrays past 50 elements are summarized rather than fully expanded. | Spec doesn't constrain JSON shape. Flat key-paths preserve structure for retrieval; the array cap prevents a 1000-element list from exhausting the chunk budget. |
| JSON questions file | Accept both `["q1", "q2"]` and `{"questions": ["q1", "q2"]}` | Spec doesn't define the schema. Both are common conventions; supporting both costs nothing. |
| Vector DB | Chroma, persisted to disk | Zero-setup, file-based, survives restarts. No cloud account needed for the reviewer. |
| Embedding model | `text-embedding-3-small` | Cheapest OpenAI embedding. Pairs naturally with `gpt-4o-mini`. |
| Concurrency | uvicorn pinned to `--workers 1` | Chroma's persistent client is not safe for concurrent multi-process writes against the same on-disk directory. |
| Document IDs | UUID4 | Filename-derived IDs collide on concurrent uploads of identically-named files. |
| Scanned PDFs | Detect (zero extractable text) and return a clear error rather than silently returning empty answers | `pypdf` cannot extract text from image-only pages. Failing loudly is better than mysterious "I don't know" responses. |
| Tests | Mock the LLM by default with `langchain-core`'s `FakeListLLM`. Real-call tests gated by `RUN_LIVE_TESTS=1` env var. | A test suite that hits the real OpenAI API on every save would burn the $5 budget in one CI run. |
| Auth | None | Out of scope for a coding-challenge demo. Documented here rather than half-implemented. |
| File size cap | 50 MB | Prevents a runaway PDF from hanging the worker or burning the budget. |

## OpenAI cost model

Modeled at planning time with `cost-watcher`:

| Operation | Cost |
| --- | --- |
| Embed sample 50-page PDF (~25K tokens) | ~$0.0005 |
| One question batch (5 questions, top-k=4) | ~$0.004 |
| 30 dev iterations (re-embed each time) | ~$0.13 |
| Realistic total (build + reviewer testing) | **~$0.20** |

The $5 cap has ~25x headroom. The hard cap of $4 in code preserves $1 for unexpected reviewer testing.

## Development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env       # fill in OPENAI_API_KEY
uvicorn app.main:app --reload
```

## Testing

```bash
pytest                           # uses FakeListLLM, no API calls
RUN_LIVE_TESTS=1 pytest          # exercises the real OpenAI API (costs ~$0.005)
```

## Evaluation harness

A QA bot you can't measure is a QA bot you can't trust. `eval/` holds a labeled question set and scoring rig that runs the same code path the API serves.

```bash
python -m eval.cli --document /tmp/soc2.pdf
```

**What it scores** (per question, then aggregated):

| Metric | How |
|---|---|
| **Deterministic** | Hard substring check (`contains` / `contains_any`) or strict refusal-sentence match. Catches regressions in known-fact questions. |
| **Faithfulness** | LLM-as-judge: does every claim in the answer trace to the retrieved context? Hallucination detector. |
| **Relevance** | LLM-as-judge: does the answer address the question? |
| **Refusal precision / recall** | Did the bot refuse exactly the questions it should have? |

Each metric has a configurable threshold; the CLI exits non-zero on failure (CI-ready). Default labels live in `eval/datasets/soc2.json` — 10 questions over the sample SOC2 PDF, mixing factual (7) and out-of-scope (3, refusal expected).

**Note on judge model quality:** the judge runs through the same `chat_completion` as the bot, so it inherits whatever provider is configured. Running the eval with a tiny local model (e.g., `llama3.2:1b` via Ollama) will produce noisy or pessimistic judge scores — small models are unreliable evaluators. Running with `gpt-4o-mini` produces meaningful numbers. The harness is the value; the absolute scores depend on the model you wire up.

## Local-model fallback (Ollama)

The OpenAI client respects `OPENAI_BASE_URL`, so pointing it at Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1`) routes the entire pipeline through a local model — useful if your OpenAI key is rate-limited or for offline development. See `.env.example` for the toggle. No code changes needed.

## Project layout

```
app/
  main.py            # FastAPI app entrypoint
  config.py          # Settings (pydantic-settings)
  api/               # HTTP route handlers
  core/              # Ingestion, embedding, retrieval, QA
  models/            # Pydantic request/response schemas
  utils/             # Logging, cost tracking, helpers
eval/                # Labeled question sets + LLM-as-judge scoring rig
tests/               # Unit + integration tests
chroma_db/           # Persisted vector store (gitignored)
```

## Out of scope (intentional)

These would be reasonable next steps but were excluded to keep the build focused:

- Streaming responses
- Multi-document corpus queries
- Conversation history / follow-up questions
- Hybrid retrieval (BM25 + dense)
- Re-ranker for higher precision
- Authentication / rate limiting
- Cloud deployment

## License

MIT.
