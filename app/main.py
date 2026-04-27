import logging
from pathlib import Path

import openai
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from app.api import documents, qa
from app.config import settings
from app.core.ingestion import EmptyPdfError, IngestionError, ScannedPdfError
from app.utils.cost import BudgetExceeded


_STATIC_DIR = Path(__file__).parent / "static"


logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


app = FastAPI(
    title="Zania RAG Bot",
    description="Document question-answering API powered by Retrieval-Augmented Generation.",
    version="0.1.0",
)


app.include_router(documents.router, tags=["documents"])
app.include_router(qa.router, tags=["qa"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")


def _problem(status: int, type_: str, title: str, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"type": type_, "title": title, "status": status, "detail": detail},
    )


@app.exception_handler(BudgetExceeded)
async def _budget_handler(request: Request, exc: BudgetExceeded) -> JSONResponse:
    return _problem(402, "budget_exceeded", "OpenAI budget cap reached", str(exc))


@app.exception_handler(ScannedPdfError)
async def _scanned_pdf_handler(request: Request, exc: ScannedPdfError) -> JSONResponse:
    return _problem(422, "scanned_pdf", "PDF has no extractable text", str(exc))


@app.exception_handler(EmptyPdfError)
async def _empty_pdf_handler(request: Request, exc: EmptyPdfError) -> JSONResponse:
    return _problem(422, "empty_pdf", "PDF has no pages", str(exc))


@app.exception_handler(IngestionError)
async def _ingestion_handler(request: Request, exc: IngestionError) -> JSONResponse:
    return _problem(422, "ingestion_error", "Could not parse uploaded document", str(exc))


@app.exception_handler(ValidationError)
async def _validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
    # Body validation done via model_validate (multipart paths) bypasses FastAPI's
    # auto 422 — surface those as 422 too instead of leaking a 500.
    return _problem(422, "validation_error", "Request payload failed validation", str(exc))


def _extract_openai_error(exc: openai.APIStatusError) -> tuple[str | None, str]:
    # OpenAI returns {"error": {"code": ..., "type": ..., "message": ...}} on status errors.
    code: str | None = None
    message = str(exc)
    try:
        body = exc.response.json() if exc.response is not None else None
        if isinstance(body, dict):
            err = body.get("error") or {}
            code = err.get("code") or err.get("type")
            message = err.get("message") or message
    except Exception:
        pass
    return code, message


@app.exception_handler(openai.APIConnectionError)
async def _llm_unreachable_handler(
    request: Request, exc: openai.APIConnectionError
) -> JSONResponse:
    target = settings.openai_base_url or "api.openai.com"
    return _problem(
        503,
        "llm_unreachable",
        "Could not reach the LLM provider",
        f"Connection to {target} failed: {exc}",
    )


@app.exception_handler(openai.APIStatusError)
async def _llm_status_handler(
    request: Request, exc: openai.APIStatusError
) -> JSONResponse:
    code, message = _extract_openai_error(exc)
    status = exc.status_code or 502
    logger = logging.getLogger(__name__)
    logger.warning("llm.upstream_error status=%s code=%s message=%s", status, code, message[:200])

    if code == "insufficient_quota":
        return _problem(
            402,
            "llm_quota_exhausted",
            "LLM provider quota exhausted",
            f"Upstream returned insufficient_quota: {message}",
        )
    if status == 429:
        return _problem(429, "llm_rate_limited", "LLM provider rate-limited the request", message)
    if status == 401:
        # Don't leak 401 to the caller — they didn't fail to auth, our key did.
        return _problem(
            503,
            "llm_auth_failed",
            "LLM provider rejected the API key",
            "Server-side credential issue. Check OPENAI_API_KEY.",
        )
    return _problem(
        502,
        "llm_upstream_error",
        f"LLM provider returned HTTP {status}",
        message,
    )
