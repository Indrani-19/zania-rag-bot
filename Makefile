.PHONY: help install test lint run docker-up docker-down eval clean

help:
	@echo "Targets:"
	@echo "  install      Create venv and install dev deps"
	@echo "  test         Run unit tests (mocked LLM, no API calls)"
	@echo "  lint         Run ruff"
	@echo "  run          Start the FastAPI server locally on :8000"
	@echo "  docker-up    Build and start the service via docker compose"
	@echo "  docker-down  Stop the docker-compose stack"
	@echo "  eval         Run the eval harness against /tmp/soc2.pdf"
	@echo "  clean        Remove caches, chroma persist dir, and cost log"

install:
	python3.12 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements-dev.txt

test:
	./venv/bin/pytest -v

lint:
	./venv/bin/ruff check app eval tests

run:
	./venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

docker-up:
	docker compose up --build

docker-down:
	docker compose down

eval:
	@test -f /tmp/soc2.pdf || (echo "Sample PDF missing. Fetch with:"; \
	  echo "  curl -L -o /tmp/soc2.pdf https://productfruits.com/docs/soc2-type2.pdf"; exit 1)
	./venv/bin/python -m eval.cli --document /tmp/soc2.pdf

clean:
	rm -rf chroma_db cost_log.jsonl .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
