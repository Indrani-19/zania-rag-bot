"""Eval CLI — run a labeled question set against the RAG pipeline.

Usage:
    python -m eval.cli --document /tmp/soc2.pdf --dataset eval/datasets/soc2.json
    python -m eval.cli --document /tmp/soc2.pdf  # uses the default dataset
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from eval.runner import Thresholds, format_scorecard, report_passes, run_eval


DEFAULT_DATASET = Path(__file__).parent / "datasets" / "soc2.json"


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
    return p.parse_args()


async def _amain() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    thresholds = Thresholds(
        deterministic_min=args.deterministic_min,
        faithfulness_min=args.faithfulness_min,
        relevance_min=args.relevance_min,
    )
    report = await run_eval(args.document, args.dataset, concurrency=args.concurrency)
    print(format_scorecard(report, thresholds))
    return 0 if report_passes(report, thresholds) else 1


def main() -> None:
    sys.exit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
