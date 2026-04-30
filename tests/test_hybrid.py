from unittest.mock import AsyncMock, patch

import pytest

from app import config
from app.core import bm25
from app.core.retriever import _rrf_fuse, retrieve


def _hit(text: str, page: int = 1, similarity: float = 0.5) -> dict:
    return {
        "text": text,
        "metadata": {"source": "doc.pdf", "page": page},
        "similarity": similarity,
    }


def test_rrf_fuse_combines_two_lists_with_dup_chunks_summing():
    dense = [_hit("alpha"), _hit("beta"), _hit("gamma")]
    lexical = [_hit("beta"), _hit("alpha"), _hit("delta")]
    fused = _rrf_fuse(dense, lexical, rrf_k=60)

    texts = [h["text"] for h in fused]
    assert "alpha" in texts and "beta" in texts
    # alpha and beta appear in BOTH lists — they should outrank gamma/delta
    # which appear in only one.
    top_two = set(texts[:2])
    assert top_two == {"alpha", "beta"}


def test_rrf_fuse_empty_inputs():
    assert _rrf_fuse([], [], 60) == []
    only_dense = _rrf_fuse([_hit("x")], [], 60)
    assert len(only_dense) == 1 and only_dense[0]["similarity"] > 0


@pytest.mark.asyncio
async def test_retrieve_skips_hybrid_when_flag_off(monkeypatch):
    monkeypatch.setattr(config.settings, "hybrid_enabled", False)
    monkeypatch.setattr(config.settings, "rerank_enabled", False)
    monkeypatch.setattr(config.settings, "query_rewrite_enabled", False)
    monkeypatch.setattr(config.settings, "retrieval_top_k", 4)

    with (
        patch(
            "app.core.retriever.embed_texts",
            AsyncMock(return_value=[[0.0] * 1536]),
        ),
        patch(
            "app.core.retriever.vectorstore.query",
            AsyncMock(return_value=[_hit(f"c{i}", similarity=0.9 - i * 0.1) for i in range(4)]),
        ),
        patch("app.core.bm25.query") as mock_bm25,
    ):
        result = await retrieve("doc1", "test query")

    mock_bm25.assert_not_called()
    assert len(result.hits) == 4


@pytest.mark.asyncio
async def test_retrieve_runs_hybrid_when_flag_on(monkeypatch):
    monkeypatch.setattr(config.settings, "hybrid_enabled", True)
    monkeypatch.setattr(config.settings, "rerank_enabled", False)
    monkeypatch.setattr(config.settings, "query_rewrite_enabled", False)
    monkeypatch.setattr(config.settings, "retrieval_top_k", 3)
    monkeypatch.setattr(config.settings, "retrieval_fetch_k", 10)

    dense = [_hit(f"d{i}", similarity=0.5) for i in range(5)]
    lexical = [_hit(f"d{i}", similarity=0.0) for i in (1, 0, 4)]

    with (
        patch(
            "app.core.retriever.embed_texts",
            AsyncMock(return_value=[[0.0] * 1536]),
        ),
        patch(
            "app.core.retriever.vectorstore.query",
            AsyncMock(return_value=dense),
        ),
        patch(
            "app.core.bm25.query",
            AsyncMock(return_value=lexical),
        ) as mock_bm25,
    ):
        result = await retrieve("doc1", "test query")

    mock_bm25.assert_called_once()
    assert len(result.hits) == 3


def test_bm25_tokenize_strips_punctuation_and_lowercases():
    tokens = bm25._tokenize("AES-256 Encryption (in transit)!")
    assert tokens == ["aes", "256", "encryption", "in", "transit"]


@pytest.mark.asyncio
async def test_bm25_query_returns_empty_when_no_chunks(monkeypatch):
    bm25._index_cache.clear()
    with patch(
        "app.core.bm25.vectorstore.list_chunks",
        AsyncMock(return_value=[]),
    ):
        result = await bm25.query("missing-doc", "any question", top_k=4)
    assert result == []


@pytest.mark.asyncio
async def test_bm25_query_ranks_lexically_relevant_chunks(monkeypatch):
    bm25._index_cache.clear()
    chunks = [
        {"text": "weather is sunny today", "metadata": {"page": 1}},
        {"text": "AES-256 encryption is mandatory", "metadata": {"page": 2}},
        {"text": "the dog ran fast", "metadata": {"page": 3}},
    ]
    with patch(
        "app.core.bm25.vectorstore.list_chunks",
        AsyncMock(return_value=chunks),
    ):
        result = await bm25.query("test-doc", "AES encryption", top_k=2)

    assert len(result) >= 1
    assert "AES" in result[0]["text"]
