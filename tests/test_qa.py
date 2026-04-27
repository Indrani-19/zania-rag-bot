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
