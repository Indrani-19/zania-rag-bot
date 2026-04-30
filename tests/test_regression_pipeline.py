"""End-to-end pipeline regression suite.

Locks in observable behavior of `answer_question()` over the full stack —
ingestion → Chroma → intent classification → retrieval → similarity floor →
prompt routing — without any real OpenAI calls.

Embeddings are replaced with a deterministic 256-dim hashed bag-of-words so
retrieval is reproducible. The LLM is mocked. Hybrid / rerank / query-rewrite
are pinned off so this suite tests the base path; turn them on in their own
dedicated suites. If anyone breaks intent routing, the floor short-circuit,
the listing/summary code paths, or the prompts those paths use, this fires.
"""

import hashlib
import math
import re
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from app.config import settings
from app.core import vectorstore
from app.core.ingestion import ingest
from app.core.qa import (
    GREETING_RESPONSE,
    HELP_RESPONSE,
    INSUFFICIENT_CONTEXT_ANSWER,
    LOW_SIGNAL_RESPONSE,
    answer_question,
)


_FIXTURE_DOC = (
    b'{"title":"Acme Cloud Security Policy",'
    b'"infrastructure":{"cloud_provider":"AWS","primary_region":"us-east-1"},'
    b'"encryption":{"at_rest":"AES-256-GCM","in_transit":"TLS 1.3"},'
    b'"incident_response":{"documented":true,"sla_hours":4},'
    b'"access_control":{"mfa_required":true,"sso":"Okta"}}'
)


_EMBED_DIM = 256


def _bag_of_words_embed(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for text in texts:
        vec = [0.0] * _EMBED_DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if len(tok) <= 2:
                continue
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % _EMBED_DIM
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        out.append([v / norm for v in vec] if norm > 0 else vec)
    return out


async def _mock_embed_texts(texts, request_id=None):
    return _bag_of_words_embed(texts)


@pytest_asyncio.fixture
async def indexed_doc(monkeypatch):
    # Word-bag cosine on a tiny doc lands in ~0.0 (no overlap) to ~0.4 (overlap).
    # 0.1 cleanly separates "off-topic" (0.0) from "factual match" (>0.1).
    monkeypatch.setattr(settings, "similarity_floor", 0.1)

    # Pin retrieval upgrades off — this suite covers the base pipeline.
    monkeypatch.setattr(settings, "rerank_enabled", False)
    monkeypatch.setattr(settings, "hybrid_enabled", False)
    monkeypatch.setattr(settings, "query_rewrite_enabled", False)

    # `embed_texts` is imported into both vectorstore and retriever; patch both.
    monkeypatch.setattr("app.core.vectorstore.embed_texts", _mock_embed_texts)
    monkeypatch.setattr("app.core.retriever.embed_texts", _mock_embed_texts)

    chunks = ingest(_FIXTURE_DOC, "policy.json")
    document_id = "regression-doc"
    await vectorstore.index_document(document_id, chunks)
    try:
        yield document_id
    finally:
        await vectorstore.delete_document(document_id)


# ---------------------------------------------------------------------------
# Canned-intent paths: no LLM, no retrieval, no Chroma touch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("question,expected", [
    ("hi", GREETING_RESPONSE),
    ("hello there", GREETING_RESPONSE),
    ("thanks", GREETING_RESPONSE),
    ("ok", GREETING_RESPONSE),
    ("help", HELP_RESPONSE),
    ("what can you do?", HELP_RESPONSE),
    ("how does this work?", HELP_RESPONSE),
    ("?", LOW_SIGNAL_RESPONSE),
    ("!!!", LOW_SIGNAL_RESPONSE),
    ("a", LOW_SIGNAL_RESPONSE),
])
async def test_canned_intents_short_circuit_without_llm_or_retrieval(
    indexed_doc, question, expected
):
    with (
        patch("app.core.qa.chat_completion", AsyncMock()) as mock_llm,
        patch("app.core.qa.retrieve", AsyncMock()) as mock_retrieve,
    ):
        result = await answer_question(indexed_doc, question)

    assert result.answer == expected
    assert result.sources == []
    assert result.retrieval_score is None
    mock_llm.assert_not_called()
    mock_retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# Factual path: retrieval clears floor → strict QA prompt → sources attached
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_factual_question_routes_to_strict_qa_prompt_with_sources(indexed_doc):
    with patch("app.core.qa.chat_completion", AsyncMock(return_value="AWS.")) as mock_llm:
        result = await answer_question(
            indexed_doc, "What cloud provider does the entity use?"
        )

    assert result.answer == "AWS."
    assert result.sources, "factual answer must carry source attribution"
    assert result.retrieval_score is not None
    assert result.retrieval_score >= settings.similarity_floor
    mock_llm.assert_called_once()
    system_prompt = mock_llm.call_args.kwargs["system"]
    assert "question-answering assistant" in system_prompt
    user_prompt = mock_llm.call_args.kwargs["user"]
    assert "CONTEXT:" in user_prompt and "QUESTION:" in user_prompt


# ---------------------------------------------------------------------------
# Floor short-circuit: off-topic question with zero overlap → fixed refusal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_off_topic_question_short_circuits_via_similarity_floor(indexed_doc):
    with patch("app.core.qa.chat_completion", AsyncMock()) as mock_llm:
        result = await answer_question(
            indexed_doc, "What is the CEO's personal phone number?"
        )

    assert result.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert result.sources == []
    assert result.retrieval_score is not None
    assert result.retrieval_score < settings.similarity_floor
    mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Listing intent: takes the broad list_chunks path with the listing prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_listing_intent_routes_to_listing_prompt(indexed_doc):
    with patch(
        "app.core.qa.chat_completion",
        AsyncMock(return_value="• AWS\n• AES-256-GCM\n• TLS 1.3"),
    ) as mock_llm:
        result = await answer_question(indexed_doc, "List all the security controls.")

    assert mock_llm.called
    system_prompt = mock_llm.call_args.kwargs["system"]
    assert "extraction assistant" in system_prompt
    # Listing path skips the per-question retrieval and so does not produce a score.
    assert result.retrieval_score is None
    assert result.answer.startswith("•")


# ---------------------------------------------------------------------------
# Summary intent: takes the summarizer path with the summary prompt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summary_intent_routes_to_summary_prompt(indexed_doc):
    with patch(
        "app.core.qa.chat_completion",
        AsyncMock(return_value="• Cloud: AWS\n• Encryption: AES-256"),
    ) as mock_llm:
        result = await answer_question(indexed_doc, "Summarize this document.")

    assert mock_llm.called
    system_prompt = mock_llm.call_args.kwargs["system"]
    assert "summarizer" in system_prompt
    assert result.retrieval_score is None
    assert result.answer.startswith("•")
