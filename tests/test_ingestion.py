from unittest.mock import MagicMock, patch

import pytest

from app.core.ingestion import (
    EmptyPdfError,
    IngestionError,
    JSON_ARRAY_EXPAND_LIMIT,
    ScannedPdfError,
    flatten_json,
    ingest,
    load_json,
    load_pdf,
)


def test_flatten_json_simple_dict():
    result = flatten_json({"name": "Alice", "age": 30})
    assert "name: Alice" in result
    assert "age: 30" in result


def test_flatten_json_nested_dict_uses_dotted_paths():
    result = flatten_json({"user": {"profile": {"email": "a@b.com"}}})
    assert any("user.profile.email: a@b.com" in line for line in result)


def test_flatten_json_array_under_limit_expands_all():
    result = flatten_json({"tags": ["x", "y", "z"]})
    assert any("tags[0]: x" in line for line in result)
    assert any("tags[2]: z" in line for line in result)
    assert not any("truncated" in line for line in result)


def test_flatten_json_array_over_limit_truncates_with_marker():
    big = list(range(JSON_ARRAY_EXPAND_LIMIT + 10))
    result = flatten_json({"items": big})
    assert any("truncated" in line for line in result)
    assert any(f"items[{JSON_ARRAY_EXPAND_LIMIT - 1}]" in line for line in result)
    assert not any(f"items[{JSON_ARRAY_EXPAND_LIMIT}]:" in line for line in result)


def test_load_json_invalid_raises_ingestion_error():
    with pytest.raises(IngestionError):
        load_json(b"{ not valid json")


def test_load_pdf_empty_raises():
    mock_reader = MagicMock()
    mock_reader.pages = []
    with patch("app.core.ingestion.PdfReader", return_value=mock_reader):
        with pytest.raises(EmptyPdfError):
            load_pdf(b"%PDF-fake")


def test_load_pdf_no_text_raises_scanned_error():
    page = MagicMock()
    page.extract_text.return_value = ""
    mock_reader = MagicMock()
    mock_reader.pages = [page, page, page]
    with patch("app.core.ingestion.PdfReader", return_value=mock_reader):
        with pytest.raises(ScannedPdfError):
            load_pdf(b"%PDF-fake")


def test_ingest_unsupported_extension_raises():
    with pytest.raises(IngestionError):
        ingest(b"data", "file.docx")
