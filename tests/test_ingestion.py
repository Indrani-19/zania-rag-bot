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


# ----- xlsx -----


from io import BytesIO  # noqa: E402

from openpyxl import Workbook  # noqa: E402

from app.core.ingestion import load_xlsx  # noqa: E402


def _build_xlsx(rows: list[list], sheet_name: str = "Sheet1") -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    for row in rows:
        ws.append(row)
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_load_xlsx_emits_one_block_per_data_row_with_headers():
    content = _build_xlsx([
        ["question", "answer"],
        ["What is the retention?", "90 days."],
        ["What is the SLA?", "99.9% uptime."],
    ])
    rows = load_xlsx(content)
    assert len(rows) == 2
    (s1, r1, t1), (s2, r2, t2) = rows
    assert s1 == 1 and s2 == 1
    assert r1 == 2 and r2 == 3
    assert "question: What is the retention?" in t1
    assert "answer: 90 days." in t1
    assert "answer: 99.9% uptime." in t2
    assert t1.startswith("[Sheet1, row 2]")
    assert t2.startswith("[Sheet1, row 3]")


def test_load_xlsx_skips_empty_rows():
    content = _build_xlsx([
        ["question", "answer"],
        ["Q1", "A1"],
        [None, None],
        ["", ""],
        ["Q2", "A2"],
    ])
    rows = load_xlsx(content)
    assert len(rows) == 2
    assert "Q1" in rows[0][2] and "Q2" in rows[1][2]


def test_load_xlsx_assigns_generic_header_when_missing():
    content = _build_xlsx([
        [None, "answer"],
        ["X", "Y"],
    ])
    rows = load_xlsx(content)
    assert "col1: X" in rows[0][2]
    assert "answer: Y" in rows[0][2]


def test_load_xlsx_with_no_data_rows_raises():
    content = _build_xlsx([["question", "answer"]])  # header only
    with pytest.raises(IngestionError):
        load_xlsx(content)


def test_load_xlsx_invalid_bytes_raises():
    with pytest.raises(IngestionError):
        load_xlsx(b"not an xlsx file at all")


def test_ingest_xlsx_attaches_sheet_index_as_page():
    content = _build_xlsx([
        ["question", "answer"],
        ["Where are data centers?", "US Central, GCP."],
    ])
    docs = ingest(content, "kb.xlsx")
    assert len(docs) >= 1
    assert all(d.metadata.get("source") == "kb.xlsx" for d in docs)
    assert any(d.metadata.get("page") == 1 for d in docs)
    combined = "\n".join(d.page_content for d in docs)
    assert "US Central" in combined


def test_ingest_xlsx_emits_one_chunk_per_row():
    rows_data = [["question", "answer"]] + [
        [f"Q{i}?", f"A{i}."] for i in range(1, 51)
    ]
    content = _build_xlsx(rows_data)
    docs = ingest(content, "kb.xlsx")
    assert len(docs) == 50
    assert all(d.metadata.get("row") for d in docs)
    assert {d.metadata["row"] for d in docs} == set(range(2, 52))
