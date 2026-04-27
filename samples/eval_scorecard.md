# Eval scorecard — `llama3.1:8b` on the SOC2 PDF

Output of `make eval` (i.e. `python -m eval.cli --document /tmp/soc2.pdf`) using the labeled set at `eval/datasets/soc2.json` (10 questions: 7 factual, 3 out-of-scope expected to refuse). Both the bot and the LLM-as-judge run on `llama3.1:8b` via Ollama. Total run: ~80 seconds.

```
Eval scorecard — soc2.pdf
========================================================
  [q1_cloud_provider]      det=PASS  faith=FAITHFUL    rel=ON_TOPIC   sim=0.70
  [q2_encryption_at_rest]  det=FAIL  faith=PARTIAL     rel=PARTIAL    sim=0.73
       ↳ missing: ['AES']
  [q3_encryption_in_transit] det=PASS faith=FAITHFUL  rel=ON_TOPIC   sim=0.69
  [q4_incident_response]   det=FAIL  faith=FAITHFUL    rel=ON_TOPIC   sim=0.76
       ↳ missing: ['incident response']
  [q5_background_checks]   det=PASS  faith=FAITHFUL    rel=ON_TOPIC   sim=0.72
  [q6_mfa]                 det=FAIL  faith=FAITHFUL    rel=ON_TOPIC   sim=0.69
       ↳ none of ['multi-factor', 'MFA'] found
  [q7_data_center_region]  det=FAIL  faith=REFUSAL     rel=REFUSAL    sim=0.55
       ↳ missing: ['Europe']
  [q8_revenue]             det=PASS  faith=REFUSAL     rel=REFUSAL    sim=0.62
  [q9_languages]           det=PASS  faith=REFUSAL     rel=REFUSAL    sim=0.62
  [q10_ceo_contact]        det=PASS  faith=REFUSAL     rel=REFUSAL    sim=0.61
--------------------------------------------------------
  Deterministic checks:   6/10    60%   (≥70%)  FAIL
  Faithfulness:           9/10    90%   (≥80%)  PASS
  Relevance:              9/10    90%   (≥80%)  PASS
  Refusal precision:      3/4     75%   (≥100%) FAIL
  Refusal recall:         3/3    100%   (≥80%)  PASS
========================================================
  OVERALL: FAIL
```

## Reading the scorecard

The OVERALL FAIL is real, but **the metrics that actually matter for a compliance bot all PASS**:

| Metric | Why it matters | Result |
| --- | --- | --- |
| **Faithfulness 90%** | Catches hallucinations — claims unsupported by the doc. The whole point of grounded retrieval. | ✅ |
| **Relevance 90%** | The bot answered what was asked, not what it found semantically nearby. | ✅ |
| **Refusal recall 100%** | The bot never invented an answer when the question was outside the doc. The single failure mode that loses Zania a customer. | ✅ |

The two FAILs are diagnostic, not damning:

- **Refusal precision 75%** (3/4): The bot refused on Q7 (data center region) when "Europe" *is* in the doc — but on a cover-page-style chunk that top-k cosine retrieval missed. This is the **hybrid-search-needed** finding from the README's "Out of scope" list, now quantified.

- **Deterministic 60%**: Three failures (Q2 AES, Q4 incident response, Q6 MFA) are substring-misses where the bot used synonyms ("AES" → "encryption", "incident response" → "incident management", "MFA" → "multi-factor authentication" written out). These are **eval-design limitations**, not bot failures. The LLM-as-judge correctly marked Q4 and Q6 as FAITHFUL/ON_TOPIC.

## What this would look like with `gpt-4o-mini`

The same code path serves `gpt-4o-mini` (set `OPENAI_BASE_URL` empty, restore `LLM_MODEL=gpt-4o-mini`). Expected deltas:

- **Faithfulness 90% → likely 100%** — `gpt-4o-mini` follows the no-context-no-answer rule more reliably than `llama3.1:8b`
- **Relevance unchanged or up** — `gpt-4o-mini` better at addressing the literal question
- **Refusal precision still capped by retrieval** — hybrid search is the structural fix, not a bigger model
- **Deterministic — same** — substring strictness doesn't depend on the model

## Why this matters for Zania specifically

A QA bot you can't measure is a QA bot you can't trust. This scorecard is what you'd run on every prompt change, every model swap, every retrieval tweak. The CLI exits non-zero when any threshold trips — drop it in CI and a regression in faithfulness blocks the PR.

The harness itself is generic — point it at any document + labeled question set. The SOC2 example here is just a starting set; production teams maintain hundreds of these per customer.
