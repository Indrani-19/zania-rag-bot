FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY samples ./samples

# HF Spaces' container filesystem is mostly read-only; writable areas are HOME and /data.
# Point the sentence-transformers model cache at HOME so first-call downloads succeed.
ENV HF_HOME=/tmp/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache

# Port 7860 is HF Spaces' Docker SDK default. For local Docker runs you can still
# `docker run -p 8000:7860 ...` and reach the app at http://localhost:8000.
EXPOSE 7860

# workers=1 is intentional: Chroma's persistent client is not safe for concurrent
# multi-process writes against the same on-disk directory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
