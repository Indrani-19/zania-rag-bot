from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.core import bm25, vectorstore
from app.core.embeddings import embed_texts


@dataclass
class Hit:
    text: str
    source: str | None
    page: int | None
    similarity: float


@dataclass
class Retrieval:
    hits: list[Hit]
    max_similarity: float
    below_floor: bool


def _chunk_key(h: dict[str, Any]) -> tuple:
    md = h.get("metadata") or {}
    return (md.get("source"), md.get("page"), md.get("row"), h["text"][:64])


def _rrf_fuse(
    dense: list[dict[str, Any]],
    lexical: list[dict[str, Any]],
    rrf_k: int,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion — combine two ranked lists by `1 / (rrf_k + rank)`.
    `rrf_k` (typically 60) damps the contribution of low-ranked items so the
    top of each list dominates without one list completely shadowing the other.
    """
    scored: dict[tuple, dict[str, Any]] = {}
    for rank, h in enumerate(dense):
        key = _chunk_key(h)
        scored[key] = {**h, "_rrf": 1.0 / (rrf_k + rank + 1)}
    for rank, h in enumerate(lexical):
        key = _chunk_key(h)
        bonus = 1.0 / (rrf_k + rank + 1)
        if key in scored:
            scored[key]["_rrf"] += bonus
        else:
            scored[key] = {**h, "_rrf": bonus}

    fused = sorted(scored.values(), key=lambda h: h["_rrf"], reverse=True)
    for h in fused:
        h["similarity"] = h.pop("_rrf")
    return fused


async def retrieve(
    document_id: str, question: str, request_id: str | None = None
) -> Retrieval:
    if settings.query_rewrite_enabled:
        from app.core.query_rewriter import rewrite_for_retrieval

        retrieval_query = await rewrite_for_retrieval(question, request_id=request_id)
    else:
        retrieval_query = question

    [embedding] = await embed_texts([retrieval_query], request_id=request_id)

    # Fetch wider when reranking or hybrid — both want a richer candidate set
    # before fusing/scoring. When neither is on, fetch_k == top_k.
    fetch_k = (
        settings.retrieval_fetch_k
        if (settings.rerank_enabled or settings.hybrid_enabled)
        else settings.retrieval_top_k
    )
    raw_hits = await vectorstore.query(document_id, embedding, fetch_k)

    if settings.hybrid_enabled:
        # Use the original question for BM25, not the HyDE-rewritten one — BM25
        # wants the user's literal terms; HyDE was for semantic embedding.
        lexical_hits = await bm25.query(document_id, question, fetch_k)
        if lexical_hits or raw_hits:
            raw_hits = _rrf_fuse(raw_hits, lexical_hits, settings.rrf_k)
            raw_hits = raw_hits[:fetch_k]

    if settings.rerank_enabled and raw_hits:
        from app.core.reranker import rerank

        scores = await rerank(question, [h["text"] for h in raw_hits])
        for h, s in zip(raw_hits, scores):
            h["similarity"] = s
        raw_hits.sort(key=lambda h: h["similarity"], reverse=True)

    raw_hits = raw_hits[: settings.retrieval_top_k]

    hits = [
        Hit(
            text=h["text"],
            source=h["metadata"].get("source"),
            page=h["metadata"].get("page"),
            similarity=h["similarity"],
        )
        for h in raw_hits
    ]
    max_sim = max((h.similarity for h in hits), default=0.0)
    return Retrieval(
        hits=hits,
        max_similarity=max_sim,
        below_floor=max_sim < settings.similarity_floor,
    )
