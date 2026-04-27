import json

import pytest

from app.utils.cost import BudgetExceeded, CostTracker


def test_record_writes_event_to_log(tmp_path):
    log = tmp_path / "log.jsonl"
    tracker = CostTracker(log_path=log)
    tracker.record(model="gpt-4o-mini", input_tokens=1000, output_tokens=500)

    lines = log.read_text().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["model"] == "gpt-4o-mini"
    assert event["input_tokens"] == 1000
    assert event["output_tokens"] == 500
    assert event["cost_usd"] > 0


def test_record_increments_cumulative_cost(tmp_path):
    tracker = CostTracker(log_path=tmp_path / "log.jsonl")
    tracker.record(model="gpt-4o-mini", input_tokens=1000)
    first = tracker.cumulative_cost_usd
    tracker.record(model="gpt-4o-mini", input_tokens=1000)
    assert tracker.cumulative_cost_usd == pytest.approx(first * 2)


def test_load_cumulative_from_existing_log_survives_restart(tmp_path):
    log = tmp_path / "log.jsonl"
    t1 = CostTracker(log_path=log)
    t1.record(model="gpt-4o-mini", input_tokens=10_000, output_tokens=2_000)
    expected = t1.cumulative_cost_usd

    t2 = CostTracker(log_path=log)
    assert t2.cumulative_cost_usd == pytest.approx(expected)


def test_check_budget_raises_when_over_cap(tmp_path):
    tracker = CostTracker(log_path=tmp_path / "log.jsonl", hard_cap_usd=0.0001)
    tracker.record(model="gpt-4o-mini", input_tokens=10_000, output_tokens=10_000)
    with pytest.raises(BudgetExceeded):
        tracker.check_budget()


def test_estimate_unknown_model_raises(tmp_path):
    tracker = CostTracker(log_path=tmp_path / "log.jsonl")
    with pytest.raises(ValueError):
        tracker.estimate("gpt-99-supreme", 100)
