from dataclasses import dataclass

from app.core.qa import INSUFFICIENT_CONTEXT_ANSWER


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _ci(text: str) -> str:
    return text.lower()


def check_contains(answer: str, required: list[str]) -> CheckResult:
    missing = [s for s in required if _ci(s) not in _ci(answer)]
    if missing:
        return CheckResult("contains", False, f"missing: {missing}")
    return CheckResult("contains", True, f"all present: {required}")


def check_contains_any(answer: str, options: list[str]) -> CheckResult:
    found = [s for s in options if _ci(s) in _ci(answer)]
    if not found:
        return CheckResult("contains_any", False, f"none of {options} found")
    return CheckResult("contains_any", True, f"found: {found}")


def check_refusal(answer: str) -> CheckResult:
    is_refusal = answer.strip() == INSUFFICIENT_CONTEXT_ANSWER
    return CheckResult(
        "refusal",
        is_refusal,
        "matched canonical refusal sentence" if is_refusal else f"answered instead: {answer[:80]!r}",
    )


def evaluate_expected(expected: dict, answer: str) -> CheckResult:
    if expected.get("refusal"):
        return check_refusal(answer)
    if "contains" in expected:
        return check_contains(answer, expected["contains"])
    if "contains_any" in expected:
        return check_contains_any(answer, expected["contains_any"])
    return CheckResult("unknown", False, f"no recognized expectation in {expected}")
