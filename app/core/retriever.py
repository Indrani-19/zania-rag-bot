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
    raw_hits = await vectorstore.query(
        document_id, embedding, settings.retrieval_top_k
    )
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
