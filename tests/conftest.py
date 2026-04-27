import os

# Must run before any `app.*` import — Settings() validates OPENAI_API_KEY at module load.
os.environ.setdefault("OPENAI_API_KEY", "sk-fake-test-key")

import pytest


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    from app import config
    from app.core import embeddings, llm, vectorstore
    from app.utils import cost

    monkeypatch.setattr(config.settings, "openai_api_key", "sk-fake-test-key")
    monkeypatch.setattr(config.settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(config.settings, "cost_hard_cap_usd", 100.0)

    monkeypatch.setattr(
        cost,
        "tracker",
        cost.CostTracker(log_path=tmp_path / "cost.jsonl", hard_cap_usd=100.0),
    )

    monkeypatch.setattr(vectorstore, "_client", None)
    monkeypatch.setattr(vectorstore, "_collection", None)
    monkeypatch.setattr(embeddings, "_client", None)
    monkeypatch.setattr(llm, "_client", None)

    yield
