import asyncio
import logging
from typing import Any

import chromadb
from langchain_core.documents import Document

from app.config import settings
from app.core.embeddings import embed_texts
from app.core.hashing import hash_chunk


logger = logging.getLogger(__name__)

COLLECTION_NAME = "documents"

_client: Any | None = None
_collection: Any | None = None


def get_collection() -> Any:
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": settings.embedding_model,
        },
    )

    # Refuse to operate on a collection embedded with a different model —
    # silent dimension mismatches between embedding models corrupt retrieval.
    stored_model = (_collection.metadata or {}).get("embedding_model")
    if stored_model and stored_model != settings.embedding_model:
        raise RuntimeError(
            f"Chroma collection was embedded with {stored_model!r}, "
            f"but settings.embedding_model={settings.embedding_model!r}. "
            f"Wipe {settings.chroma_persist_dir!r} to re-index."
        )
    return _collection


def _existing_embeddings(content_hashes: list[str]) -> dict[str, list[float]]:
    if not content_hashes:
        return {}
    result = get_collection().get(
        where={"content_hash": {"$in": content_hashes}},
        include=["metadatas", "embeddings"],
    )
    out: dict[str, list[float]] = {}
    metadatas = result.get("metadatas")
    embeddings = result.get("embeddings")
    if metadatas is None or embeddings is None:
        return out
    for meta, emb in zip(metadatas, embeddings):
        h = (meta or {}).get("content_hash")
        if h and h not in out:
            out[h] = list(emb)
    return out


def _delete_document_sync(document_id: str) -> int:
    coll = get_collection()
    existing = coll.get(where={"document_id": document_id}, include=[])
    n = len(existing.get("ids") or [])
    if n:
        coll.delete(where={"document_id": document_id})
    return n


async def delete_document(document_id: str) -> int:
    return await asyncio.to_thread(_delete_document_sync, document_id)


async def index_document(
    document_id: str,
    documents: list[Document],
    request_id: str | None = None,
) -> int:
    if not documents:
        return 0

    # Make ingest idempotent: re-uploading the same document_id replaces, not duplicates.
    await delete_document(document_id)

    chunks = [
        {
            "text": d.page_content,
            "metadata": d.metadata,
            "content_hash": hash_chunk(d.page_content),
        }
        for d in documents
    ]

    cached = await asyncio.to_thread(
        _existing_embeddings, list({c["content_hash"] for c in chunks})
    )
    to_embed = [c for c in chunks if c["content_hash"] not in cached]
    cache_hits = len(chunks) - len(to_embed)

    if to_embed:
        new_embeddings = await embed_texts(
            [c["text"] for c in to_embed], request_id=request_id
        )
        for c, emb in zip(to_embed, new_embeddings):
            cached[c["content_hash"]] = emb

    embeddings = [cached[c["content_hash"]] for c in chunks]

    await asyncio.to_thread(_upsert_sync, document_id, chunks, embeddings)
    logger.info(
        "Indexed document_id=%s chunks=%d cache_hits=%d new_embeddings=%d",
        document_id, len(chunks), cache_hits, len(to_embed),
    )
    return len(chunks)


def _upsert_sync(
    document_id: str,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> None:
    coll = get_collection()
    ids = [f"{document_id}:{i}" for i in range(len(chunks))]
    coll.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {**(c["metadata"] or {}), "document_id": document_id, "content_hash": c["content_hash"]}
            for c in chunks
        ],
    )


def _query_sync(
    document_id: str, query_embedding: list[float], k: int
) -> list[dict[str, Any]]:
    result = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"document_id": document_id},
        include=["documents", "metadatas", "distances"],
    )
    documents_list = (result.get("documents") or [[]])[0]
    metadatas_list = (result.get("metadatas") or [[]])[0]
    distances_list = (result.get("distances") or [[]])[0]

    return [
        {
            "text": doc,
            "metadata": meta or {},
            "distance": dist,
            "similarity": _distance_to_similarity(dist),
        }
        for doc, meta, dist in zip(documents_list, metadatas_list, distances_list)
    ]


async def query(
    document_id: str, query_embedding: list[float], k: int
) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_query_sync, document_id, query_embedding, k)


def _distance_to_similarity(d: float) -> float:
    return max(0.0, min(1.0, 1.0 - d))
