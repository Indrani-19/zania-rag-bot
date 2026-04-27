import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app.core import vectorstore
from app.core.ingestion import ingest
from app.core.qa import Answer, answer_question
from eval.judge import judge_faithfulness, judge_relevance
from eval.metrics import CheckResult, evaluate_expected


logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    id: str
    question: str
    category: str
    answer: str
    deterministic: CheckResult
    faithfulness: str
    relevance: str
    retrieval_score: float | None


@dataclass
class EvalReport:
    document: str
    results: list[QuestionResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def deterministic_passed(self) -> int:
        return sum(1 for r in self.results if r.deterministic.passed)

    @property
    def faithful_passed(self) -> int:
        # Treat REFUSAL as a pass — refusing on out-of-scope is the desired behavior.
        return sum(1 for r in self.results if r.faithfulness in ("FAITHFUL", "REFUSAL"))

    @property
    def relevant_passed(self) -> int:
        return sum(1 for r in self.results if r.relevance in ("ON_TOPIC", "REFUSAL"))

    @property
    def refusal_precision(self) -> tuple[int, int]:
        # Of all answers that WERE refusals, how many were on questions labeled to refuse?
        refused = [r for r in self.results if r.faithfulness == "REFUSAL"]
        if not refused:
            return (0, 0)
        correct = sum(1 for r in refused if r.category == "out_of_scope")
        return (correct, len(refused))

    @property
    def refusal_recall(self) -> tuple[int, int]:
        # Of all questions labeled to refuse, how many actually got refused?
        should_refuse = [r for r in self.results if r.category == "out_of_scope"]
        if not should_refuse:
            return (0, 0)
        correct = sum(1 for r in should_refuse if r.faithfulness == "REFUSAL")
        return (correct, len(should_refuse))


def _format_context(answer: Answer) -> str:
    return "\n---\n".join(s.snippet for s in answer.sources)


async def _evaluate_one(
    question_spec: dict, document_id: str, request_id: str
) -> QuestionResult:
    answer = await answer_question(document_id, question_spec["question"], request_id=request_id)
    deterministic = evaluate_expected(question_spec["expected"], answer.answer)

    context = _format_context(answer) or "(no retrieval hits — bot short-circuited)"
    faithfulness, relevance = await asyncio.gather(
        judge_faithfulness(question_spec["question"], answer.answer, context),
        judge_relevance(question_spec["question"], answer.answer),
    )

    return QuestionResult(
        id=question_spec["id"],
        question=question_spec["question"],
        category=question_spec["category"],
        answer=answer.answer,
        deterministic=deterministic,
        faithfulness=faithfulness,
        relevance=relevance,
        retrieval_score=answer.retrieval_score,
    )


async def run_eval(
    document_path: str | Path,
    dataset_path: str | Path,
    concurrency: int = 4,
) -> EvalReport:
    document_path = Path(document_path)
    dataset_path = Path(dataset_path)

    dataset = json.loads(dataset_path.read_text())
    document_bytes = document_path.read_bytes()
    chunks = ingest(document_bytes, document_path.name)

    document_id = f"eval-{uuid.uuid4()}"
    request_id = document_id

    try:
        await vectorstore.index_document(document_id, chunks, request_id=request_id)

        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded(spec: dict) -> QuestionResult:
            async with semaphore:
                return await _evaluate_one(spec, document_id, request_id)

        results = await asyncio.gather(*[_bounded(q) for q in dataset["questions"]])
    finally:
        await vectorstore.delete_document(document_id)

    return EvalReport(document=dataset.get("document", document_path.name), results=list(results))


@dataclass
class Thresholds:
    deterministic_min: float = 0.70
    faithfulness_min: float = 0.80
    relevance_min: float = 0.80
    refusal_precision_min: float = 1.00  # zero false-refusals tolerated
    refusal_recall_min: float = 0.80


def format_scorecard(report: EvalReport, thresholds: Thresholds) -> str:
    lines: list[str] = []
    lines.append(f"Eval scorecard — {report.document}")
    lines.append("=" * 56)

    for r in report.results:
        det = "PASS" if r.deterministic.passed else "FAIL"
        score = f"{r.retrieval_score:.2f}" if r.retrieval_score is not None else "  -"
        lines.append(
            f"  [{r.id}] det={det} faith={r.faithfulness:<11} rel={r.relevance:<10} sim={score}"
        )
        if not r.deterministic.passed:
            lines.append(f"       ↳ {r.deterministic.detail}")

    lines.append("-" * 56)

    det_n, det_total = report.deterministic_passed, report.total
    faith_n = report.faithful_passed
    rel_n = report.relevant_passed
    rp_correct, rp_total = report.refusal_precision
    rr_correct, rr_total = report.refusal_recall

    def _line(label: str, n: int, total: int, threshold: float) -> str:
        rate = n / total if total else 1.0
        verdict = "PASS" if rate >= threshold else "FAIL"
        return f"  {label:<22} {n:>2}/{total:<2}  {rate:>5.0%}   (≥{threshold:.0%})  {verdict}"

    lines.append(_line("Deterministic checks:", det_n, det_total, thresholds.deterministic_min))
    lines.append(_line("Faithfulness:", faith_n, det_total, thresholds.faithfulness_min))
    lines.append(_line("Relevance:", rel_n, det_total, thresholds.relevance_min))
    if rp_total:
        lines.append(_line("Refusal precision:", rp_correct, rp_total, thresholds.refusal_precision_min))
    if rr_total:
        lines.append(_line("Refusal recall:", rr_correct, rr_total, thresholds.refusal_recall_min))

    overall_pass = (
        det_n / det_total >= thresholds.deterministic_min
        and faith_n / det_total >= thresholds.faithfulness_min
        and rel_n / det_total >= thresholds.relevance_min
        and (not rp_total or rp_correct / rp_total >= thresholds.refusal_precision_min)
        and (not rr_total or rr_correct / rr_total >= thresholds.refusal_recall_min)
    )
    lines.append("=" * 56)
    lines.append(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    return "\n".join(lines)


def report_passes(report: EvalReport, thresholds: Thresholds) -> bool:
    if report.total == 0:
        return False
    rp_correct, rp_total = report.refusal_precision
    rr_correct, rr_total = report.refusal_recall
    return (
        report.deterministic_passed / report.total >= thresholds.deterministic_min
        and report.faithful_passed / report.total >= thresholds.faithfulness_min
        and report.relevant_passed / report.total >= thresholds.relevance_min
        and (not rp_total or rp_correct / rp_total >= thresholds.refusal_precision_min)
        and (not rr_total or rr_correct / rr_total >= thresholds.refusal_recall_min)
    )
