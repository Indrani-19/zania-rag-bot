"""Baseline-relative regression check for the eval harness.

Hard thresholds in `Thresholds` only catch absolute floors — they don't catch
the case where a prompt or model change quietly drops faithfulness from 95%
to 85%. Both pass the 80% floor; the 10pp drop is the regression.

This module captures a per-metric snapshot of the last accepted run as a
checked-in baseline JSON, then compares fresh runs to it with a tolerance
expressed in percentage points.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from eval.runner import EvalReport


@dataclass(frozen=True)
class MetricSnapshot:
    deterministic: float
    faithfulness: float
    relevance: float
    refusal_precision: float | None
    refusal_recall: float | None


@dataclass
class Baseline:
    document: str
    tolerance_pp: float
    metrics: MetricSnapshot
    captured_at: str = ""
    notes: str = ""


@dataclass
class MetricDelta:
    name: str
    baseline: float
    current: float
    delta_pp: float  # current − baseline, in percentage points (positive = improvement)


@dataclass
class RegressionResult:
    passed: bool
    tolerance_pp: float
    deltas: list[MetricDelta] = field(default_factory=list)

    @property
    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.delta_pp < -self.tolerance_pp]


def _ratio(num: int, denom: int) -> float | None:
    return (num / denom) if denom else None


def snapshot_from_report(report: EvalReport) -> MetricSnapshot:
    rp_correct, rp_total = report.refusal_precision
    rr_correct, rr_total = report.refusal_recall
    return MetricSnapshot(
        deterministic=report.deterministic_passed / report.total if report.total else 0.0,
        faithfulness=report.faithful_passed / report.total if report.total else 0.0,
        relevance=report.relevant_passed / report.total if report.total else 0.0,
        refusal_precision=_ratio(rp_correct, rp_total),
        refusal_recall=_ratio(rr_correct, rr_total),
    )


def compare(
    current: MetricSnapshot, baseline: Baseline
) -> RegressionResult:
    deltas: list[MetricDelta] = []
    base = baseline.metrics
    for name in ("deterministic", "faithfulness", "relevance", "refusal_precision", "refusal_recall"):
        b = getattr(base, name)
        c = getattr(current, name)
        # If a metric isn't applicable to either run (e.g. no out-of-scope questions),
        # skip — there's nothing to compare.
        if b is None or c is None:
            continue
        deltas.append(MetricDelta(
            name=name,
            baseline=b,
            current=c,
            delta_pp=(c - b) * 100.0,
        ))
    regressed = any(d.delta_pp < -baseline.tolerance_pp for d in deltas)
    return RegressionResult(passed=not regressed, tolerance_pp=baseline.tolerance_pp, deltas=deltas)


def save_baseline(
    report: EvalReport, path: Path, tolerance_pp: float, notes: str = ""
) -> Baseline:
    baseline = Baseline(
        document=report.document,
        tolerance_pp=tolerance_pp,
        metrics=snapshot_from_report(report),
        captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )
    payload = {
        "document": baseline.document,
        "tolerance_pp": baseline.tolerance_pp,
        "metrics": asdict(baseline.metrics),
        "captured_at": baseline.captured_at,
        "notes": baseline.notes,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return baseline


def load_baseline(path: Path) -> Baseline | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    metrics_raw = raw.get("metrics") or {}
    return Baseline(
        document=raw["document"],
        tolerance_pp=float(raw["tolerance_pp"]),
        metrics=MetricSnapshot(
            deterministic=float(metrics_raw["deterministic"]),
            faithfulness=float(metrics_raw["faithfulness"]),
            relevance=float(metrics_raw["relevance"]),
            refusal_precision=(
                float(metrics_raw["refusal_precision"])
                if metrics_raw.get("refusal_precision") is not None else None
            ),
            refusal_recall=(
                float(metrics_raw["refusal_recall"])
                if metrics_raw.get("refusal_recall") is not None else None
            ),
        ),
        captured_at=raw.get("captured_at", ""),
        notes=raw.get("notes", ""),
    )


def format_regression(result: RegressionResult, baseline: Baseline) -> str:
    lines = [
        "Regression check vs baseline",
        "=" * 56,
        f"  baseline:    {baseline.document}",
        f"  captured:    {baseline.captured_at or '(unknown)'}",
        f"  tolerance:   ±{baseline.tolerance_pp:.1f} pp",
        "-" * 56,
    ]
    for d in result.deltas:
        sign = "+" if d.delta_pp >= 0 else ""
        verdict = "FAIL" if d.delta_pp < -result.tolerance_pp else "ok  "
        lines.append(
            f"  {d.name:<22} base={d.baseline:>5.0%}  now={d.current:>5.0%}  "
            f"Δ={sign}{d.delta_pp:>5.1f}pp  {verdict}"
        )
    lines.append("=" * 56)
    lines.append(f"  REGRESSION CHECK: {'PASS' if result.passed else 'FAIL'}")
    return "\n".join(lines)
