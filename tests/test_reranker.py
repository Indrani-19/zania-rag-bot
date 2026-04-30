from unittest.mock import AsyncMock, patch

import pytest

from app import config
from app.core.retriever import retrieve


def _hit(text: str, similarity: float) -> dict:
    return {
        "text": text,
        "metadata": {"source": "test.pdf", "page": 1},
        "similarity": similarity,
    }


@pytest.mark.asyncio
async def test_retrieve_skips_rerank_when_flag_off(monkeypatch):
    monkeypatch.setattr(config.settings, "rerank_enabled", False)
    monkeypatch.setattr(config.settings, "retrieval_top_k", 4)

    with (
        patch(
            "app.core.retriever.embed_texts",
            AsyncMock(return_value=[[0.0] * 1536]),
        ),
        patch(
            "app.core.retriever.vectorstore.query",
            AsyncMock(return_value=[_hit(f"chunk {i}", 0.9 - i * 0.1) for i in range(4)]),
        ) as mock_query,
        patch("app.core.reranker.rerank") as mock_rerank,
    ):
        result = await retrieve("doc1", "test question")

    mock_query.assert_called_once()
    assert mock_query.call_args.args[2] == 4
    mock_rerank.assert_not_called()
    assert len(result.hits) == 4
    assert result.hits[0].similarity == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_retrieve_fetches_wider_and_reranks_when_flag_on(monkeypatch):
    monkeypatch.setattr(config.settings, "rerank_enabled", True)
    monkeypatch.setattr(config.settings, "retrieval_top_k", 3)
    monkeypatch.setattr(config.settings, "retrieval_fetch_k", 10)

    fetched = [_hit(f"chunk {i}", 0.5 - i * 0.01) for i in range(10)]
    rerank_scores = [0.1, 0.2, 0.3, 0.4, 0.95, 0.85, 0.75, 0.05, 0.15, 0.25]

    with (
        patch(
            "app.core.retriever.embed_texts",
            AsyncMock(return_value=[[0.0] * 1536]),
        ),
        patch(
            "app.core.retriever.vectorstore.query",
            AsyncMock(return_value=fetched),
        ) as mock_query,
        patch(
            "app.core.reranker.rerank",
            AsyncMock(return_value=rerank_scores),
        ) as mock_rerank,
    ):
        result = await retrieve("doc1", "test question")

    assert mock_query.call_args.args[2] == 10
    mock_rerank.assert_called_once()
    assert len(result.hits) == 3
    assert result.hits[0].text == "chunk 4"
    assert result.hits[0].similarity == pytest.approx(0.95)
    assert result.hits[1].text == "chunk 5"
    assert result.hits[2].text == "chunk 6"
    assert result.max_similarity == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_rerank_score_drives_below_floor_decision(monkeypatch):
    monkeypatch.setattr(config.settings, "rerank_enabled", True)
    monkeypatch.setattr(config.settings, "retrieval_top_k", 2)
    monkeypatch.setattr(config.settings, "retrieval_fetch_k", 5)
    monkeypatch.setattr(config.settings, "similarity_floor", 0.5)

    fetched = [_hit(f"c{i}", 0.9) for i in range(5)]

    with (
        patch(
            "app.core.retriever.embed_texts",
            AsyncMock(return_value=[[0.0] * 1536]),
        ),
        patch(
            "app.core.retriever.vectorstore.query",
            AsyncMock(return_value=fetched),
        ),
        patch(
            "app.core.reranker.rerank",
            AsyncMock(return_value=[0.1, 0.2, 0.3, 0.15, 0.05]),
        ),
    ):
        result = await retrieve("doc1", "irrelevant query")

    assert result.below_floor is True
    assert result.max_similarity == pytest.approx(0.3)
