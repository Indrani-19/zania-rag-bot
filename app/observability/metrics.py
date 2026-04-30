from fastapi import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
)

llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed",
    ["operation", "kind"],
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency",
    ["operation"],
)


async def metrics_endpoint(_request) -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
