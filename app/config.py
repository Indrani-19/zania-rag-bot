from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    # When set (e.g. http://localhost:11434/v1 for Ollama), the OpenAI client targets
    # that base URL instead of api.openai.com. Keeps the SDK call sites unchanged.
    openai_base_url: str | None = None

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def _empty_base_url_is_none(cls, v: str | None) -> str | None:
        # `OPENAI_BASE_URL=` in .env reads as ""; the SDK treats that as a real (broken) URL.
        return v or None

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    # "openai" -> call the OpenAI-compatible endpoint at openai_base_url for embeddings
    #             (works for OpenAI, Ollama, vLLM, anything that exposes /v1/embeddings).
    # "local"  -> run sentence-transformers in-process (no API call, no key required).
    #             Required when chat is on a provider like Groq that doesn't host embeddings.
    embedding_provider: str = "openai"

    # When DEMO_PRELOAD=true, samples/spec_kb.json is indexed at app startup under a
    # well-known document_id so first-time visitors to the chat UI can ask the spec's
    # five sample questions immediately, with no upload step.
    demo_preload: bool = False
    demo_document_id: str = "demo-spec-kb"

    chroma_persist_dir: str = "./chroma_db"
    retrieval_top_k: int = 4
    similarity_floor: float = 0.5
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Cross-encoder reranking. When enabled, fetch retrieval_fetch_k candidates
    # from Chroma, score them with the rerank model, and return retrieval_top_k.
    # The Hit.similarity field becomes the (sigmoid'd) rerank score, so the
    # similarity_floor short-circuit still works — recalibrate it for the rerank
    # score distribution if you turn this on.
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"
    retrieval_fetch_k: int = 20

    # Hybrid retrieval — BM25 lexical search fused with dense via Reciprocal
    # Rank Fusion. Catches exact-token queries (model names, IDs, keywords)
    # that pure dense retrieval misses. RRF score is the Hit.similarity when
    # hybrid is on, so similarity_floor needs recalibration like with rerank.
    hybrid_enabled: bool = False
    rrf_k: int = 60

    # Query rewriting. When enabled, run a lightweight LLM call to rewrite
    # the question into a hypothetical answer (HyDE), then embed THAT for
    # retrieval. Helps when query vocabulary differs from doc vocabulary.
    # Adds one LLM call (~200-500 ms) before retrieval.
    query_rewrite_enabled: bool = False

    max_upload_size_mb: int = 50
    cost_hard_cap_usd: float = 4.0
    log_level: str = "INFO"
    llm_timeout_s: float = 30.0


settings = Settings()
