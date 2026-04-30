import asyncio
import re
from typing import Any

from rank_bm25 import BM25Okapi  # type: ignore

from app.core import vectorstore


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


_index_cache: dict[str, tuple[BM25Okapi, list[dict[str, Any]]]] = {}
_build_lock: asyncio.Lock | None = None


async def _get_or_build_index(document_id: str) -> tuple[BM25Okapi, list[dict[str, Any]]] | None:
    global _build_lock
    if _build_lock is None:
        _build_lock = asyncio.Lock()
    async with _build_lock:
        cached = _index_cache.get(document_id)
        if cached is not None:
            return cached

        chunks = await vectorstore.list_chunks(document_id, limit=10_000)
        if not chunks:
            return None

        tokenized = [_tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        _index_cache[document_id] = (bm25, chunks)
        return bm25, chunks


async def query(
    document_id: str, question: str, top_k: int
) -> list[dict[str, Any]]:
    built = await _get_or_build_index(document_id)
    if built is None:
        return []
    bm25, chunks = built

    tokens = _tokenize(question)
    if not tokens:
        return []

    def _score() -> list[float]:
        return list(bm25.get_scores(tokens))

    scores = await asyncio.to_thread(_score)
    ranked = sorted(
        ((c, s) for c, s in zip(chunks, scores) if s > 0),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    return [
        {
            "text": c["text"],
            "metadata": c["metadata"],
            "bm25_score": float(s),
        }
        for c, s in ranked
    ]


def invalidate(document_id: str) -> None:
    _index_cache.pop(document_id, None)
