import asyncio

from openai import AsyncOpenAI

from app.config import settings
from app.utils.cost import tracker


_openai_client: AsyncOpenAI | None = None
_local_model = None  # lazily-loaded sentence_transformers.SentenceTransformer
_local_model_lock = None  # asyncio.Lock — created on first use


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return _openai_client


async def _embed_via_openai(
    texts: list[str], request_id: str | None
) -> list[list[float]]:
    tracker.check_budget()
    response = await asyncio.wait_for(
        _get_openai_client().embeddings.create(
            model=settings.embedding_model,
            input=texts,
        ),
        timeout=settings.llm_timeout_s,
    )
    tracker.record(
        model=settings.embedding_model,
        input_tokens=response.usage.total_tokens,
        operation="embedding",
        request_id=request_id,
    )
    return [item.embedding for item in response.data]


async def _embed_via_local(
    texts: list[str], request_id: str | None
) -> list[list[float]]:
    # Imported lazily so the dep is only loaded when actually used.
    import asyncio
    from sentence_transformers import SentenceTransformer  # type: ignore

    global _local_model, _local_model_lock
    if _local_model_lock is None:
        _local_model_lock = asyncio.Lock()
    async with _local_model_lock:
        if _local_model is None:
            # Loaded once per process. ~90 MB download on first run; cached after.
            _local_model = await asyncio.to_thread(
                SentenceTransformer, settings.embedding_model
            )

    def _encode() -> list[list[float]]:
        # normalize_embeddings=True so cosine similarity matches what Chroma expects.
        vectors = _local_model.encode(  # type: ignore[union-attr]
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.tolist() for v in vectors]

    return await asyncio.to_thread(_encode)


async def embed_texts(
    texts: list[str], request_id: str | None = None
) -> list[list[float]]:
    if not texts:
        return []
    if settings.embedding_provider == "local":
        return await _embed_via_local(texts, request_id)
    return await _embed_via_openai(texts, request_id)
