from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "rag-chatbot"
    app_version: str = "0.1.0"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    database_url: str = "postgresql+psycopg2://rag:rag@localhost:5432/rag"
    redis_url: str = "redis://localhost:6379/0"
    storage_dir: str = "storage"

    openai_api_key: str = ""
    openai_base_url: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    # "local" (sentence-transformers on-device, default), "huggingface"
    # (HF Inference API), "openai" (OpenAI-compatible /embeddings endpoint),
    # or "hash" (deterministic offline embedder, for tests/CI)
    embedding_provider: str = "local"
    # HF Inference API token (https://huggingface.co/settings/tokens).
    # Empty = anonymous requests (heavily rate-limited).
    hf_token: str = ""
    hf_base_url: str = "https://router.huggingface.co/hf-inference"

    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 80
    retrieval_top_k: int = 8
    context_token_budget: int = 3000
    score_threshold: float = 0.0

    # --- Phase 6: security -------------------------------------------------
    # Empty key = auth disabled (local dev). Production must set both.
    api_key: str = ""
    admin_key: str = ""  # falls back to api_key when empty
    rate_limit_per_minute: int = 600  # 0 disables rate limiting
    rate_limit_burst: int = 120

    # --- Phase 6: observability -------------------------------------------
    log_level: str = "INFO"
    # Standard OTLP variable; empty = spans are not exported (still traced in logs).
    otel_exporter_otlp_endpoint: str = ""

    # --- Phase 6: caching / robustness ------------------------------------
    query_cache_ttl: int = 60  # 0 disables the exact-query cache
    max_upload_bytes: int = 20_000_000
    llm_max_retries: int = 2
    llm_timeout_seconds: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
