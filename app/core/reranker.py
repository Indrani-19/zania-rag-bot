import asyncio
import math
import time

from app.config import settings
from app.observability.metrics import llm_request_duration_seconds


_model = None  # lazily-loaded sentence_transformers.CrossEncoder
_model_lock = None  # asyncio.Lock — created on first use


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


async def rerank(question: str, candidates: list[str]) -> list[float]:
    if not candidates:
        return []

    from sentence_transformers import CrossEncoder  # type: ignore

    global _model, _model_lock
    if _model_lock is None:
        _model_lock = asyncio.Lock()
    async with _model_lock:
        if _model is None:
            _model = await asyncio.to_thread(CrossEncoder, settings.rerank_model)

    pairs = [(question, c) for c in candidates]

    def _score() -> list[float]:
        raw = _model.predict(pairs, show_progress_bar=False)  # type: ignore[union-attr]
        return [_sigmoid(float(r)) for r in raw]

    start = time.perf_counter()
    scores = await asyncio.to_thread(_score)
    llm_request_duration_seconds.labels(operation="rerank").observe(
        time.perf_counter() - start
    )
    return scores
