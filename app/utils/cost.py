import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings


# Verify against https://openai.com/api/pricing/ — these prices change.
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "text-embedding-3-small": {"input": 0.02 / 1_000_000, "output": 0.0},
    # Local Ollama models — free to run, tracked for parity with the cloud path.
    "llama3.2:1b": {"input": 0.0, "output": 0.0},
    "llama3.2:3b": {"input": 0.0, "output": 0.0},
    "nomic-embed-text": {"input": 0.0, "output": 0.0},
}


class BudgetExceeded(Exception):
    pass


@dataclass
class UsageEvent:
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    operation: str
    request_id: str | None = None


class CostTracker:
    def __init__(self, log_path: str | Path, hard_cap_usd: float | None = None):
        self.log_path = Path(log_path)
        self.hard_cap_usd = hard_cap_usd
        self._lock = threading.Lock()
        self._cumulative_cost = self._load_cumulative()

    def _load_cumulative(self) -> float:
        if not self.log_path.exists():
            return 0.0
        total = 0.0
        with self.log_path.open() as f:
            for line in f:
                try:
                    total += json.loads(line)["cost_usd"]
                except (json.JSONDecodeError, KeyError):
                    continue
        return total

    @property
    def cumulative_cost_usd(self) -> float:
        return self._cumulative_cost

    def check_budget(self) -> None:
        if self.hard_cap_usd is not None and self._cumulative_cost >= self.hard_cap_usd:
            raise BudgetExceeded(
                f"Cumulative OpenAI spend ${self._cumulative_cost:.4f} >= hard cap ${self.hard_cap_usd}"
            )

    def estimate(self, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        if model not in PRICING:
            raise ValueError(f"Unknown model for pricing: {model}")
        prices = PRICING[model]
        return input_tokens * prices["input"] + output_tokens * prices["output"]

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
        operation: str = "completion",
        request_id: str | None = None,
    ) -> UsageEvent:
        cost = self.estimate(model, input_tokens, output_tokens)

        event = UsageEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            operation=operation,
            request_id=request_id,
        )

        with self._lock:
            with self.log_path.open("a") as f:
                f.write(json.dumps(asdict(event)) + "\n")
            self._cumulative_cost += cost

        return event


tracker = CostTracker(
    log_path="cost_log.jsonl",
    hard_cap_usd=settings.cost_hard_cap_usd,
)
