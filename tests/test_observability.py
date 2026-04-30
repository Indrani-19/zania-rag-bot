import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.observability.logging import request_id_var


@pytest.fixture
def client():
    return TestClient(app)


def test_metrics_endpoint_returns_200_with_counter_name(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_server_echoes_client_supplied_request_id(client):
    response = client.get("/health", headers={"X-Request-ID": "my-trace-id"})
    assert response.headers.get("X-Request-ID") == "my-trace-id"


def test_server_generates_request_id_when_none_supplied(client):
    response = client.get("/health")
    rid = response.headers.get("X-Request-ID")
    assert rid is not None and len(rid) == 36  # UUID4 canonical form


def test_json_log_includes_required_fields(caplog):
    handler_saw = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            handler_saw.append(record)

    capture = _Capture()
    token = request_id_var.set("test-req-id")
    logger = logging.getLogger("test.observability")
    logger.addHandler(capture)
    try:
        logger.warning("hello from test")
    finally:
        logger.removeHandler(capture)
        request_id_var.reset(token)

    assert handler_saw, "no log records captured"
    record = handler_saw[0]
    assert record.getMessage() == "hello from test"
    assert record.levelname == "WARNING"
    assert getattr(record, "request_id", None) == "test-req-id"
