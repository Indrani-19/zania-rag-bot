import asyncio
import time

from openai import AsyncOpenAI

from app.config import settings
from app.observability.metrics import llm_request_duration_seconds, llm_tokens_total
from app.utils.cost import tracker


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        # max_retries=2 caps storm-burn under transient OpenAI errors (auditor finding).
        _client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            max_retries=2,
        )
    return _client


async def chat_completion(
    system: str,
    user: str,
    request_id: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> str:
    tracker.check_budget()
    start = time.perf_counter()
    response = await asyncio.wait_for(
        _get_client().chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        timeout=settings.llm_timeout_s,
    )
    llm_request_duration_seconds.labels(operation="completion").observe(
        time.perf_counter() - start
    )
    llm_tokens_total.labels(operation="completion", kind="prompt").inc(
        response.usage.prompt_tokens
    )
    llm_tokens_total.labels(operation="completion", kind="completion").inc(
        response.usage.completion_tokens
    )
    tracker.record(
        model=settings.llm_model,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        operation="completion",
        request_id=request_id,
    )
    return (response.choices[0].message.content or "").strip()
