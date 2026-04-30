"""Eval CLI — run a labeled question set against the RAG pipeline.

Modes:
  default                 Run eval, print scorecard, exit 0/1 by absolute thresholds.
  --check-regression      Run eval, also compare to eval/baseline.json with a tolerance,
                          exit 1 if any metric drops more than --regression-tolerance-pp.
  --update-baseline       Run eval, then overwrite the baseline file with the result.
                          Use this after an intentional, accepted change.

Usage:
    python -m eval.cli --document /tmp/soc2.pdf
    python -m eval.cli --document /tmp/soc2.pdf --check-regression
    python -m eval.cli --document /tmp/soc2.pdf --update-baseline
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from eval.baseline import (
    compare,
    format_regression,
    load_baseline,
    save_baseline,
    snapshot_from_report,
)
from eval.runner import Thresholds, format_scorecard, report_passes, run_eval


DEFAULT_DATASET = Path(__file__).parent / "datasets" / "soc2.json"
DEFAULT_BASELINE = Path(__file__).parent / "baseline.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RAG evaluation on a labeled question set.")
    p.add_argument("--document", required=True, help="Path to the document (PDF or JSON).")
    p.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help=f"Path to the labeled question set JSON (default: {DEFAULT_DATASET}).",
    )
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--faithfulness-min", type=float, default=0.80)
    p.add_argument("--relevance-min", type=float, default=0.80)
    p.add_argument("--deterministic-min", type=float, default=0.70)

    p.add_argument(
        "--check-regression",
        action="store_true",
        help="Compare metrics against the baseline file; fail if any metric drops > tolerance.",
    )
    p.add_argument(
        "--update-baseline",
        action="store_true",
        help="Overwrite the baseline file with this run's metrics. Use after intentional changes.",
    )
    p.add_argument("--baseline-path", default=str(DEFAULT_BASELINE))
    p.add_argument(
        "--regression-tolerance-pp",
        type=float,
        default=5.0,
        help="Tolerance in percentage points before a drop counts as a regression (default: 5.0).",
    )
    return p.parse_args()


async def _amain() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()

    if args.check_regression and args.update_baseline:
        print("Cannot pass both --check-regression and --update-baseline.", file=sys.stderr)
        return 2

    thresholds = Thresholds(
        deterministic_min=args.deterministic_min,
        faithfulness_min=args.faithfulness_min,
        relevance_min=args.relevance_min,
    )
    report = await run_eval(args.document, args.dataset, concurrency=args.concurrency)
    print(format_scorecard(report, thresholds))
    absolute_pass = report_passes(report, thresholds)

    baseline_path = Path(args.baseline_path)

    if args.update_baseline:
        baseline = save_baseline(
            report, baseline_path, tolerance_pp=args.regression_tolerance_pp
        )
        print(f"\nBaseline written → {baseline_path}")
        print(f"  document={baseline.document} captured={baseline.captured_at}")
        return 0 if absolute_pass else 1

    if args.check_regression:
        baseline = load_baseline(baseline_path)
        if baseline is None:
            print(
                f"\nNo baseline at {baseline_path}. Run with --update-baseline first.",
                file=sys.stderr,
            )
            return 2
        result = compare(snapshot_from_report(report), baseline)
        print()
        print(format_regression(result, baseline))
        return 0 if (absolute_pass and result.passed) else 1

    return 0 if absolute_pass else 1


def main() -> None:
    sys.exit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
