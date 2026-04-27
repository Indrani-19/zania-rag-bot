from openai import AsyncOpenAI

from app.config import settings
from app.utils.cost import tracker


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return _client


async def embed_texts(
    texts: list[str], request_id: str | None = None
) -> list[list[float]]:
    if not texts:
        return []
    tracker.check_budget()
    response = await _get_client().embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    tracker.record(
        model=settings.embedding_model,
        input_tokens=response.usage.total_tokens,
        operation="embedding",
        request_id=request_id,
    )
    return [item.embedding for item in response.data]
