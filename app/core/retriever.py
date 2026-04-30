from dataclasses import dataclass

from app.config import settings
from app.core import vectorstore
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


async def retrieve(
    document_id: str, question: str, request_id: str | None = None
) -> Retrieval:
    [embedding] = await embed_texts([question], request_id=request_id)

    # Fetch wider when reranking — give the cross-encoder more candidates to
    # choose from. When rerank is off, fetch_k == top_k (no extra work).
    fetch_k = (
        settings.retrieval_fetch_k if settings.rerank_enabled
        else settings.retrieval_top_k
    )
    raw_hits = await vectorstore.query(document_id, embedding, fetch_k)

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
