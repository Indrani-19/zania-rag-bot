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

The challenge doc references a sample SOC2 PDF and 5 sample questions. The questions are committed at `samples/spec_questions.json`; fetch the PDF and run:

```bash
curl -L -o /tmp/soc2.pdf https://productfruits.com/docs/soc2-type2.pdf

curl -X POST http://localhost:8000/qa \
  -F "document=@/tmp/soc2.pdf" \
  -F "questions=@samples/spec_questions.json"
```

Add `?verbose=true` to also see the source page snippets and retrieval scores.

The spec also mentions a sample JSON document hosted as a Google Sheet; export it as JSON if you want to test the JSON path, or use any nested object — the bot recursively flattens to `key.path: value` lines before chunking.

### Smoke test with your own files

```bash
curl -X POST http://localhost:8000/qa \
  -F "document=@your.pdf" \
  -F 'questions=@questions.json'
```

`questions.json` accepts either `["q1", "q2"]` or `{"questions": ["q1", "q2"]}`.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /qa` | One-shot: upload a document + questions, get answers (matches the spec example) |
| `POST /documents` | Index a document, return a `document_id` |
| `POST /documents/{id}/questions` | Re-query an indexed document (no re-embedding cost) |
| `DELETE /documents/{id}` | Remove a document and its embeddings |
| `GET /health` | Liveness probe |

## Output

```json
[{"question": "What is X?", "answer": "X is..."}]
```

Add `?verbose=true` to also get `sources` (page snippets) and `retrieval_score` (cosine similarity of best chunk).

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
| Stateful endpoints + `/qa` convenience | Stateful is the right shape for RAG; `/qa` matches the spec's single-bundle example |
| Output is a list of `{question, answer}` objects | Spec example was malformed JSON; a list preserves duplicate questions |
| `retrieval_score`, not `confidence` | Cosine similarity measures *retrieval relevance*, not *answer correctness* — naming should not lie |
| JSON docs flattened to `key.path: value` lines, then chunked | Preserves structure for retrieval; arrays past 50 elements are summarized to bound chunk count |
| Chroma, persisted to disk | Zero-setup, no cloud account, survives restarts |
| `uvicorn --workers 1` | Chroma's persistent client isn't safe for multi-process writes against the same dir |
| Scanned PDFs detected and rejected with 422 | `pypdf` can't OCR; failing loudly beats mysterious empty answers |
| LLM mocked in tests by default | Avoids burning the budget on every commit |
| Provider via `OPENAI_BASE_URL` | One env var routes the whole pipeline through any OpenAI-compatible provider (Ollama, vLLM, etc.) |
| No auth | Coding-challenge scope — documented, not half-implemented |

## Cost

Every LLM and embedding call is logged to `cost_log.jsonl`. `COST_HARD_CAP_USD` (default `$4`) aborts any request that would push spend over the cap. Typical end-to-end run (index a 50-page PDF + answer 5 questions): ~$0.005.

## Testing

```bash
make test          # 33 unit tests, all mocked, no API calls
make lint          # ruff
```

CI runs both on every push (`.github/workflows/ci.yml`).

## Evaluation harness

A QA bot you can't measure is a QA bot you can't trust. `eval/` ships a labeled question set and a scoring rig that runs the same code path the API serves.

```bash
make eval          # uses eval/datasets/soc2.json against /tmp/soc2.pdf
```

| Metric | How |
| --- | --- |
| **Deterministic** | Substring (`contains` / `contains_any`) or strict refusal-sentence match |
| **Faithfulness** | LLM-as-judge: every claim in the answer traceable to retrieved context? |
| **Relevance** | LLM-as-judge: does the answer address the question? |
| **Refusal precision / recall** | Did the bot refuse exactly the questions it should have? |

Configurable thresholds; CLI exits non-zero on regression (CI-ready). The judge inherits whatever provider you've configured — small local models (e.g. `llama3.2:1b`) make poor judges. Run with `gpt-4o-mini` for meaningful numbers.

## Local-model fallback (Ollama)

Set `OPENAI_BASE_URL=http://localhost:11434/v1` (and matching `LLM_MODEL` / `EMBEDDING_MODEL`) — the OpenAI SDK routes everything through Ollama. Useful when offline or the OpenAI key is rate-limited. See `.env.example` for the toggle. No code changes needed.

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
samples/         # Spec-provided sample question set
tests/           # Mocked unit tests
```

## Out of scope (intentional)

Streaming responses · multi-document corpus queries · conversation history · hybrid retrieval (BM25 + dense) · re-ranker · auth / rate limiting · cloud deployment.

## License

MIT.
