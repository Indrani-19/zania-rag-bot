import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_serves_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "<title>Zania RAG Bot</title>" in body
    # Element ids the page's JS binds to — guards against accidental rename.
    for marker in ('id="doc-input"', 'id="composer"', 'id="send-btn"', 'id="messages"'):
        assert marker in body


def _fake_embedding(_texts: list[str]) -> list[list[float]]:
    # Returns a deterministic 1536-dim vector per call (matches text-embedding-3-small dims).
    return [[0.001] * 1536 for _ in _texts]


def test_qa_endpoint_returns_answers_with_mocked_llm(client):
    with (
        patch(
            "app.core.vectorstore.embed_texts",
            AsyncMock(side_effect=lambda texts, **_kw: _fake_embedding(texts)),
        ),
        patch(
            "app.core.retriever.embed_texts",
            AsyncMock(side_effect=lambda texts, **_kw: _fake_embedding(texts)),
        ),
        patch(
            "app.core.qa.chat_completion",
            AsyncMock(return_value="The retention period is 90 days."),
        ),
    ):
        document_payload = json.dumps({"policy": "Retention period is 90 days."}).encode()
        questions_payload = json.dumps(["What is the retention period?"]).encode()

        response = client.post(
            "/qa",
            files={
                "document": ("policy.json", document_payload, "application/json"),
                "questions": ("questions.json", questions_payload, "application/json"),
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    assert body[0]["question"] == "What is the retention period?"
    assert body[0]["answer"] == "The retention period is 90 days."
    assert "sources" not in body[0]
    assert "retrieval_score" not in body[0]


def test_qa_endpoint_verbose_includes_sources(client):
    with (
        patch(
            "app.core.vectorstore.embed_texts",
            AsyncMock(side_effect=lambda texts, **_kw: _fake_embedding(texts)),
        ),
        patch(
            "app.core.retriever.embed_texts",
            AsyncMock(side_effect=lambda texts, **_kw: _fake_embedding(texts)),
        ),
        patch(
            "app.core.qa.chat_completion",
            AsyncMock(return_value="42 days."),
        ),
    ):
        doc = json.dumps({"policy": "Retention is 42 days."}).encode()
        qs = json.dumps({"questions": ["how long?"]}).encode()

        response = client.post(
            "/qa?verbose=true",
            files={
                "document": ("policy.json", doc, "application/json"),
                "questions": ("questions.json", qs, "application/json"),
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert "sources" in body[0]
    assert "retrieval_score" in body[0]


def _fake_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/embeddings")


def _fake_response(status: int, body: dict) -> httpx.Response:
    return httpx.Response(status_code=status, json=body, request=_fake_request())


def test_qa_returns_402_when_openai_quota_exhausted(client):
    quota_body = {
        "error": {
            "message": "You exceeded your current quota.",
            "type": "insufficient_quota",
            "code": "insufficient_quota",
        }
    }
    err = openai.RateLimitError(
        "quota exhausted",
        response=_fake_response(429, quota_body),
        body=quota_body,
    )
    with patch(
        "app.core.vectorstore.embed_texts",
        AsyncMock(side_effect=err),
    ):
        response = client.post(
            "/qa",
            files={
                "document": ("p.json", b'{"x":"y"}', "application/json"),
                "questions": ("q.json", b'["q?"]', "application/json"),
            },
        )

    assert response.status_code == 402, response.text
    body = response.json()
    assert body["type"] == "llm_quota_exhausted"
    assert body["status"] == 402
    assert "insufficient_quota" in body["detail"]


def test_qa_rejects_blank_question(client):
    response = client.post(
        "/qa",
        files={
            "document": ("p.json", b'{"x":"y"}', "application/json"),
            "questions": ("q.json", b'["valid?", "   "]', "application/json"),
        },
    )
    assert response.status_code == 422
    assert "empty" in response.text.lower() or "whitespace" in response.text.lower()


def test_qa_returns_504_when_llm_call_times_out(client, monkeypatch):
    monkeypatch.setattr(config.settings, "llm_timeout_s", 0.1)

    class _HangingClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**_kwargs):
                    await asyncio.sleep(60)

    from app.core import llm as _llm
    monkeypatch.setattr(_llm, "_get_client", lambda: _HangingClient)

    with (
        patch(
            "app.core.vectorstore.embed_texts",
            AsyncMock(side_effect=lambda texts, **_kw: _fake_embedding(texts)),
        ),
        patch(
            "app.core.retriever.embed_texts",
            AsyncMock(side_effect=lambda texts, **_kw: _fake_embedding(texts)),
        ),
    ):
        response = client.post(
            "/qa",
            files={
                "document": ("p.json", b'{"policy":"x"}', "application/json"),
                "questions": ("q.json", b'["What is the policy?"]', "application/json"),
            },
        )

    assert response.status_code == 504, response.text
    body = response.json()
    assert body["type"] == "llm_timeout"
    assert body["status"] == 504


def test_qa_same_bytes_yields_same_document_id(client):
    captured: list[str] = []

    real_index = None
    from app.core import vectorstore as _vs

    real_index = _vs.index_document

    async def _capture(document_id, documents, request_id=None):
        captured.append(document_id)
        return await real_index(document_id, documents, request_id=request_id)

    with (
        patch(
            "app.core.vectorstore.embed_texts",
            AsyncMock(side_effect=lambda texts, **_kw: _fake_embedding(texts)),
        ),
        patch(
            "app.core.retriever.embed_texts",
            AsyncMock(side_effect=lambda texts, **_kw: _fake_embedding(texts)),
        ),
        patch(
            "app.core.qa.chat_completion",
            AsyncMock(return_value="ok"),
        ),
        patch("app.api.qa.vectorstore.index_document", side_effect=_capture),
    ):
        doc_bytes = json.dumps({"policy": "Retention is 90 days."}).encode()
        qs_bytes = json.dumps(["What is the retention period?"]).encode()
        for _ in range(2):
            response = client.post(
                "/qa",
                files={
                    "document": ("policy.json", doc_bytes, "application/json"),
                    "questions": ("questions.json", qs_bytes, "application/json"),
                },
            )
            assert response.status_code == 200, response.text

    assert len(captured) == 2
    assert captured[0] == captured[1]


def test_documents_upload_rejects_oversized_stream(client, monkeypatch):
    monkeypatch.setattr(config.settings, "max_upload_size_mb", 1)
    payload = b"x" * (2 * 1024 * 1024)
    response = client.post(
        "/documents",
        files={"document": ("big.txt", payload, "application/octet-stream")},
    )
    assert response.status_code == 413, response.text


def test_qa_returns_503_when_openai_unreachable(client):
    err = openai.APIConnectionError(message="connect failed", request=_fake_request())
    with patch(
        "app.core.vectorstore.embed_texts",
        AsyncMock(side_effect=err),
    ):
        response = client.post(
            "/qa",
            files={
                "document": ("p.json", b'{"x":"y"}', "application/json"),
                "questions": ("q.json", b'["q?"]', "application/json"),
            },
        )

    assert response.status_code == 503, response.text
    assert response.json()["type"] == "llm_unreachable"
