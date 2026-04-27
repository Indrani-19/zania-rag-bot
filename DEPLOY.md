# Deploying zania-rag-bot to Hugging Face Spaces (Docker SDK) with Groq

## Section 1 — Get Groq access

1. Go to **https://console.groq.com/** and sign up (Google/GitHub SSO works).
2. In the left sidebar click **API Keys** → **Create API Key**, name it `zania-rag-bot`, copy the key (starts with `gsk_...`). You will not be able to view it again.
3. Set these values in your env:
   - `OPENAI_API_KEY` = `gsk_...` (Groq is OpenAI-API-compatible, so the existing OpenAI client works)
   - `OPENAI_BASE_URL` = `https://api.groq.com/openai/v1`
   - `LLM_MODEL` = `llama-3.1-8b-instant` (fast, recommended for the demo). For higher quality use `llama-3.3-70b-versatile`.
4. **Note:** Groq does not serve embeddings. Keep embeddings local via `sentence-transformers` (configured in Section 2).

## Section 2 — Create the Hugging Face Space

1. Sign up / log in at **https://huggingface.co/join**.
2. Click your avatar → **New Space**. Set **Owner** = your username, **Space name** = `zania-rag-bot`, **License** = MIT, **SDK** = **Docker** (blank template), **Visibility** = Public. Click **Create Space**.
3. The README.md in this repo already includes the YAML frontmatter HF Spaces requires (`sdk: docker`, `app_port: 7860`).
4. Link your GitHub repo to the Space as a second remote and push:
   ```bash
   git remote add space https://huggingface.co/spaces/<USERNAME>/zania-rag-bot
   git push space main
   ```
   (You'll be prompted for an HF access token — create one at **Settings → Access Tokens → New token (write)**.)
5. In the Space, go to **Settings → Variables and secrets → New secret** and add:
   - `OPENAI_API_KEY` = your Groq key
   - `OPENAI_BASE_URL` = `https://api.groq.com/openai/v1`
   - `LLM_MODEL` = `llama-3.1-8b-instant`
   - `EMBEDDING_PROVIDER` = `local`
   - `EMBEDDING_MODEL` = `sentence-transformers/all-MiniLM-L6-v2`
   - `SIMILARITY_FLOOR` = `0.3`  *(MiniLM scores ~0.1–0.2 lower than OpenAI embeddings; the default 0.5 will cause every query to short-circuit refuse)*
   - `RETRIEVAL_TOP_K` = `6`  *(default is 4; 6 catches more context and helps multi-part questions without much cost increase)*
   - `DEMO_PRELOAD` = `true`
6. The Space auto-rebuilds on every push to the `space` remote.

## Section 3 — Dockerfile change

Already applied in this repo:
- `EXPOSE 7860`
- CMD uses `--port 7860`
- `samples/` is copied into the image so `DEMO_PRELOAD=true` finds `spec_kb.json`
- `HF_HOME` / `SENTENCE_TRANSFORMERS_HOME` set to `/tmp/hf_cache` so the first-call model download has a writable directory

## Section 4 — Common pitfalls

- **Build logs:** open the Space page → **Logs** tab (top right). The **Build** sub-tab shows the Docker build; **Container** shows runtime stdout. Read this on every first deploy.
- **Free tier:** 2 vCPU / 16 GB RAM / 50 GB disk — plenty for MiniLM embeddings.
- **Cold start:** the first request after a build downloads `all-MiniLM-L6-v2` (~90 MB) into the container's HF cache. ~10–20s extra latency once per container lifetime.
- **Sleep behavior:** Docker Spaces on the free CPU tier **do** sleep after ~48 hours of inactivity and cold-start on the next request (~30s). To keep it always-on, upgrade to a paid hardware tier or pin the Space.
- **Chroma persistence:** without persistent storage the `chroma_db` directory is wiped on every rebuild/restart. Enable it via **Settings → Persistent storage** (paid add-on) and set `CHROMA_PERSIST_DIR=/data/chroma_db`. For a coding-challenge demo, `DEMO_PRELOAD=true` re-ingests `samples/spec_kb.json` on every cold start, which is good enough.
