from unittest.mock import AsyncMock, patch

import pytest

from app.core.qa import (
    INSUFFICIENT_CONTEXT_ANSWER,
    _excerpt,
    _format_context,
    answer_question,
)
from app.core.retriever import Hit, Retrieval


def _retrieval(hits: list[Hit], floor: float = 0.5) -> Retrieval:
    max_sim = max((h.similarity for h in hits), default=0.0)
    return Retrieval(hits=hits, max_similarity=max_sim, below_floor=max_sim < floor)


@pytest.mark.asyncio
async def test_short_circuit_returns_fixed_phrase_without_calling_llm():
    weak_hit = Hit(text="irrelevant", source="doc.pdf", page=1, similarity=0.1)
    with (
        patch("app.core.qa.retrieve", AsyncMock(return_value=_retrieval([weak_hit]))),
        patch("app.core.qa.chat_completion", AsyncMock()) as mock_llm,
    ):
        result = await answer_question("doc-id", "What is X?")

    assert result.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert result.sources == []
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_above_floor_calls_llm_and_attaches_sources():
    strong_hit = Hit(text="Alice is the CTO.", source="doc.pdf", page=3, similarity=0.9)
    with (
        patch("app.core.qa.retrieve", AsyncMock(return_value=_retrieval([strong_hit]))),
        patch("app.core.qa.chat_completion", AsyncMock(return_value="Alice.")) as mock_llm,
    ):
        result = await answer_question("doc-id", "Who is CTO?")

    assert result.answer == "Alice."
    assert mock_llm.called
    assert len(result.sources) == 1
    assert result.sources[0].page == 3


def test_format_context_orders_hits_by_descending_similarity():
    hits = [
        Hit(text="low", source="d.pdf", page=1, similarity=0.3),
        Hit(text="high", source="d.pdf", page=2, similarity=0.9),
        Hit(text="mid", source="d.pdf", page=3, similarity=0.6),
    ]
    block = _format_context(_retrieval(hits, floor=0.0))
    assert block.index("high") < block.index("mid") < block.index("low")


def test_excerpt_truncates_long_text_with_ellipsis():
    long = "x" * 400
    out = _excerpt(long, max_chars=100)
    assert len(out) <= 103
    assert out.endswith("...")


def test_excerpt_keeps_short_text_unchanged():
    assert _excerpt("short", max_chars=100) == "short"


# --- Intent classification ---


from app.core.qa import (  # noqa: E402
    GREETING_RESPONSE,
    HELP_RESPONSE,
    LOW_SIGNAL_RESPONSE,
    _is_greeting,
    _is_help_request,
    _is_listing_intent,
    _is_low_signal,
    _is_summary_intent,
)


@pytest.mark.parametrize(
    "text",
    ["hi", "Hi!", "hello", "hey there", "thanks", "thank you", "ok", "got it", "great", "bye"],
)
def test_greeting_detected(text):
    assert _is_greeting(text)


@pytest.mark.parametrize(
    "text",
    ["hi, what is the retention period?", "thanks for the info, but what about SLAs?", "help me find X"],
)
def test_greeting_does_not_match_substantive_questions(text):
    assert not _is_greeting(text)


@pytest.mark.parametrize(
    "text",
    ["help", "what can you do?", "what can I ask?", "how does this work?", "what is this?", "how to use this"],
)
def test_help_detected(text):
    assert _is_help_request(text)


@pytest.mark.parametrize("text", ["", " ", "?", "!!!", "a", "1", "..."])
def test_low_signal_detected(text):
    assert _is_low_signal(text)


@pytest.mark.parametrize("text", ["hi", "ok", "thanks", "What is X?", "help"])
def test_low_signal_does_not_match_real_input(text):
    assert not _is_low_signal(text)


@pytest.mark.parametrize(
    "text",
    [
        "summarize this document",
        "give me a summary",
        "TL;DR",
        "tldr please",
        "what is this document about?",
        "give me an overview",
        "main points please",
    ],
)
def test_summary_intent_detected(text):
    assert _is_summary_intent(text)


@pytest.mark.parametrize(
    "text",
    [
        "list all the security controls",
        "list the cloud providers",
        "give me all the SLAs",
        "what are all the third-party vendors?",
        "enumerate every region",
    ],
)
def test_listing_intent_detected(text):
    assert _is_listing_intent(text)


@pytest.mark.parametrize(
    "text",
    ["What is the retention period?", "Do you support SAML?", "Which cloud providers?"],
)
def test_factual_questions_do_not_trigger_special_intents(text):
    assert not _is_summary_intent(text)
    assert not _is_listing_intent(text)
    assert not _is_greeting(text)
    assert not _is_help_request(text)
    assert not _is_low_signal(text)


# --- Canned response paths skip retrieval and LLM entirely ---


@pytest.mark.asyncio
async def test_greeting_returns_canned_response_without_llm_or_retrieval():
    with (
        patch("app.core.qa.retrieve", AsyncMock()) as mock_retrieve,
        patch("app.core.qa.chat_completion", AsyncMock()) as mock_llm,
    ):
        result = await answer_question("doc-id", "hi")
    assert result.answer == GREETING_RESPONSE
    assert result.sources == []
    assert result.retrieval_score is None
    mock_retrieve.assert_not_called()
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_help_returns_canned_response_without_llm_or_retrieval():
    with (
        patch("app.core.qa.retrieve", AsyncMock()) as mock_retrieve,
        patch("app.core.qa.chat_completion", AsyncMock()) as mock_llm,
    ):
        result = await answer_question("doc-id", "help")
    assert result.answer == HELP_RESPONSE
    mock_retrieve.assert_not_called()
    mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_low_signal_returns_canned_response_without_llm_or_retrieval():
    with (
        patch("app.core.qa.retrieve", AsyncMock()) as mock_retrieve,
        patch("app.core.qa.chat_completion", AsyncMock()) as mock_llm,
    ):
        result = await answer_question("doc-id", "?")
    assert result.answer == LOW_SIGNAL_RESPONSE
    mock_retrieve.assert_not_called()
    mock_llm.assert_not_called()
