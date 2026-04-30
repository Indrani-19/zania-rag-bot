import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core.qa import INSUFFICIENT_CONTEXT_ANSWER, Answer, Source
from eval.metrics import (
    CheckResult,
    check_contains,
    check_contains_any,
    check_refusal,
    evaluate_expected,
)
from eval.baseline import (
    Baseline,
    MetricSnapshot,
    compare,
    load_baseline,
    save_baseline,
    snapshot_from_report,
)
from eval.runner import EvalReport, QuestionResult, Thresholds, report_passes, run_eval


def test_check_contains_passes_when_all_substrings_present():
    r = check_contains("Encryption uses AES-256 and TLS 1.3.", ["AES", "TLS"])
    assert r.passed


def test_check_contains_fails_with_detail_listing_missing():
    r = check_contains("Only AES is mentioned.", ["AES", "TLS"])
    assert not r.passed
    assert "TLS" in r.detail


def test_check_contains_any_passes_on_first_match():
    r = check_contains_any("Uses MFA for all admins.", ["multi-factor", "MFA"])
    assert r.passed


def test_check_refusal_matches_canonical_sentence():
    assert check_refusal(INSUFFICIENT_CONTEXT_ANSWER).passed
    assert not check_refusal("Yes, AWS is used.").passed


def test_evaluate_expected_dispatches_by_key():
    assert evaluate_expected({"contains": ["AWS"]}, "Uses AWS.").passed
    assert evaluate_expected({"refusal": True}, INSUFFICIENT_CONTEXT_ANSWER).passed
    assert evaluate_expected({"contains_any": ["MFA", "2FA"]}, "MFA enabled.").passed


def _result(category: str, deterministic_pass: bool, faith: str, rel: str) -> QuestionResult:
    return QuestionResult(
        id="x", question="?", category=category, answer="x",
        deterministic=CheckResult("contains", deterministic_pass, ""),
        faithfulness=faith, relevance=rel, retrieval_score=0.7,
    )


def test_report_aggregations():
    report = EvalReport(document="x.pdf", results=[
        _result("factual", True, "FAITHFUL", "ON_TOPIC"),
        _result("factual", True, "FAITHFUL", "ON_TOPIC"),
        _result("factual", False, "PARTIAL", "ON_TOPIC"),
        _result("out_of_scope", True, "REFUSAL", "REFUSAL"),
        _result("out_of_scope", False, "FAITHFUL", "OFF_TOPIC"),  # bot answered when it shouldn't
    ])
    assert report.deterministic_passed == 3
    assert report.faithful_passed == 4  # FAITHFUL + REFUSAL count as pass
    assert report.relevant_passed == 4  # ON_TOPIC + REFUSAL count as pass
    # Refusal precision: only 1 refusal, on a correctly-labeled out_of_scope → 1/1
    assert report.refusal_precision == (1, 1)
    # Refusal recall: 2 out_of_scope, only 1 got refused → 1/2
    assert report.refusal_recall == (1, 2)


def test_report_passes_blocks_on_low_faithfulness():
    bad = EvalReport(document="x", results=[
        _result("factual", True, "UNFAITHFUL", "ON_TOPIC"),
        _result("factual", True, "UNFAITHFUL", "ON_TOPIC"),
    ])
    assert not report_passes(bad, Thresholds())


def test_report_passes_when_all_metrics_above_threshold():
    good = EvalReport(document="x", results=[
        _result("factual", True, "FAITHFUL", "ON_TOPIC"),
        _result("out_of_scope", True, "REFUSAL", "REFUSAL"),
    ])
    assert report_passes(good, Thresholds())


@pytest.mark.asyncio
async def test_runner_end_to_end_with_mocked_qa_and_judges(tmp_path):
    dataset = {
        "document": "doc.json",
        "questions": [
            {"id": "q1", "question": "Cloud?", "expected": {"contains": ["AWS"]}, "category": "factual"},
            {"id": "q2", "question": "Revenue?", "expected": {"refusal": True}, "category": "out_of_scope"},
        ],
    }
    dataset_path = tmp_path / "set.json"
    dataset_path.write_text(json.dumps(dataset))

    doc_path = tmp_path / "doc.json"
    doc_path.write_text('{"cloud":"AWS"}')

    answers_by_q = {
        "Cloud?": Answer(question="Cloud?", answer="AWS is used.",
                         sources=[Source(page=None, snippet="cloud: AWS")], retrieval_score=0.9),
        "Revenue?": Answer(question="Revenue?", answer=INSUFFICIENT_CONTEXT_ANSWER,
                           sources=[], retrieval_score=0.1),
    }

    async def fake_answer(_doc_id, q, **_kw):
        return answers_by_q[q]

    async def fake_index(*_args, **_kw):
        return 1

    async def fake_delete(*_args, **_kw):
        return 0

    with (
        patch("eval.runner.answer_question", side_effect=fake_answer),
        patch("eval.runner.vectorstore.index_document", AsyncMock(side_effect=fake_index)),
        patch("eval.runner.vectorstore.delete_document", AsyncMock(side_effect=fake_delete)),
        patch("eval.judge.chat_completion", AsyncMock(side_effect=["FAITHFUL", "ON_TOPIC", "REFUSAL", "REFUSAL"])),
    ):
        report = await run_eval(doc_path, dataset_path)

    assert report.total == 2
    assert report.deterministic_passed == 2
    assert report.faithful_passed == 2
    assert report.relevant_passed == 2
    assert report.refusal_recall == (1, 1)


# --- Baseline regression check ---


def _good_report() -> EvalReport:
    return EvalReport(document="soc2.pdf", results=[
        _result("factual", True, "FAITHFUL", "ON_TOPIC"),
        _result("factual", True, "FAITHFUL", "ON_TOPIC"),
        _result("factual", True, "FAITHFUL", "ON_TOPIC"),
        _result("out_of_scope", True, "REFUSAL", "REFUSAL"),
    ])


def test_snapshot_from_report_computes_per_metric_ratios():
    snap = snapshot_from_report(_good_report())
    assert snap.deterministic == 1.0
    assert snap.faithfulness == 1.0
    assert snap.relevance == 1.0
    assert snap.refusal_precision == 1.0
    assert snap.refusal_recall == 1.0


def test_snapshot_skips_refusal_metrics_when_no_out_of_scope_questions():
    report = EvalReport(document="x", results=[
        _result("factual", True, "FAITHFUL", "ON_TOPIC"),
    ])
    snap = snapshot_from_report(report)
    assert snap.refusal_precision is None
    assert snap.refusal_recall is None


def test_compare_passes_when_metrics_match_baseline():
    snap = snapshot_from_report(_good_report())
    baseline = Baseline(document="x", tolerance_pp=5.0, metrics=snap)
    result = compare(snap, baseline)
    assert result.passed
    assert all(d.delta_pp == 0 for d in result.deltas)


def test_compare_passes_when_drop_within_tolerance():
    baseline = Baseline(
        document="x", tolerance_pp=5.0,
        metrics=MetricSnapshot(0.90, 0.90, 0.90, 1.0, 1.0),
    )
    current = MetricSnapshot(0.86, 0.87, 0.90, 1.0, 1.0)  # all drops ≤ 5pp
    result = compare(current, baseline)
    assert result.passed
    assert result.regressions == []


def test_compare_fails_when_any_drop_exceeds_tolerance():
    baseline = Baseline(
        document="x", tolerance_pp=5.0,
        metrics=MetricSnapshot(0.90, 0.90, 0.90, 1.0, 1.0),
    )
    current = MetricSnapshot(0.90, 0.80, 0.90, 1.0, 1.0)  # faithfulness dropped 10pp
    result = compare(current, baseline)
    assert not result.passed
    regressed_names = {d.name for d in result.regressions}
    assert regressed_names == {"faithfulness"}


def test_compare_does_not_penalize_improvements():
    baseline = Baseline(
        document="x", tolerance_pp=5.0,
        metrics=MetricSnapshot(0.80, 0.80, 0.80, 1.0, 0.5),
    )
    current = MetricSnapshot(1.00, 1.00, 1.00, 1.0, 1.0)
    result = compare(current, baseline)
    assert result.passed
    assert all(d.delta_pp >= 0 for d in result.deltas)


def test_save_and_load_baseline_round_trips(tmp_path):
    path = tmp_path / "baseline.json"
    saved = save_baseline(_good_report(), path, tolerance_pp=5.0, notes="initial")
    loaded = load_baseline(path)
    assert loaded is not None
    assert loaded.document == saved.document
    assert loaded.tolerance_pp == 5.0
    assert loaded.metrics == saved.metrics
    assert loaded.captured_at == saved.captured_at
    assert loaded.notes == "initial"


def test_load_baseline_returns_none_when_file_missing(tmp_path):
    assert load_baseline(tmp_path / "nope.json") is None
