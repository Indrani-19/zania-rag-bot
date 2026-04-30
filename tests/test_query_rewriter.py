from unittest.mock import AsyncMock, patch

import pytest

from app import config
from app.core.query_rewriter import rewrite_for_retrieval
from app.core.retriever import retrieve


@pytest.mark.asyncio
async def test_rewrite_appends_hypothetical_to_question():
    with patch(
        "app.core.query_rewriter.chat_completion",
        AsyncMock(return_value="The system uses AES-256 encryption for data at rest."),
    ):
        result = await rewrite_for_retrieval("What encryption is used?")

    assert "What encryption is used?" in result
    assert "AES-256" in result


@pytest.mark.asyncio
async def test_retrieve_uses_rewritten_query_for_embedding(monkeypatch):
    monkeypatch.setattr(config.settings, "query_rewrite_enabled", True)
    monkeypatch.setattr(config.settings, "hybrid_enabled", False)
    monkeypatch.setattr(config.settings, "rerank_enabled", False)
    monkeypatch.setattr(config.settings, "retrieval_top_k", 4)

    captured: list[str] = []

    async def _capture_embed(texts, **_kw):
        captured.extend(texts)
        return [[0.0] * 1536]

    with (
        patch("app.core.retriever.embed_texts", side_effect=_capture_embed),
        patch(
            "app.core.query_rewriter.chat_completion",
            AsyncMock(return_value="Hypothetical answer text."),
        ),
        patch(
            "app.core.retriever.vectorstore.query",
            AsyncMock(return_value=[]),
        ),
    ):
        await retrieve("doc1", "What encryption is used?")

    assert captured, "embed_texts must be called"
    assert "What encryption is used?" in captured[0]
    assert "Hypothetical answer" in captured[0]


@pytest.mark.asyncio
async def test_retrieve_skips_rewrite_when_flag_off(monkeypatch):
    monkeypatch.setattr(config.settings, "query_rewrite_enabled", False)
    monkeypatch.setattr(config.settings, "hybrid_enabled", False)
    monkeypatch.setattr(config.settings, "rerank_enabled", False)

    with (
        patch(
            "app.core.retriever.embed_texts",
            AsyncMock(return_value=[[0.0] * 1536]),
        ),
        patch("app.core.query_rewriter.chat_completion") as mock_rewrite,
        patch(
            "app.core.retriever.vectorstore.query",
            AsyncMock(return_value=[]),
        ),
    ):
        await retrieve("doc1", "test")

    mock_rewrite.assert_not_called()
