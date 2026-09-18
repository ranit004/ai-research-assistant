"""Application configuration loaded from environment variables (and an optional .env file).

Secrets are typed as `SecretStr` so Pydantic never renders their value in
repr/logging output.  Call `.get_secret_value()` only at the call-site that
actually needs the raw string (e.g. an HTTP client constructor).
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings.  Every field has a safe development default.

    Secrets (API keys, database credentials) must never be hardcoded here;
    inject them via environment variables or a local .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_name: str = "AI Research Assistant"
    environment: str = "development"  # development | staging | production
    debug: bool = False
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Qdrant ─────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "research_docs"
    qdrant_api_key: SecretStr | None = None

    # ── Embeddings ─────────────────────────────────────────────────────────────
    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # ── Ingestion limits ───────────────────────────────────────────────────────
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    allowed_mime_types: list[str] = [
        "application/pdf",
        "text/plain",
        "text/markdown",
    ]
    chunk_size: int = 512        # tokens / characters per chunk
    chunk_overlap: int = 64      # overlap between consecutive chunks

    # ── Research engine ────────────────────────────────────────────────────────
    llm_model: str = "gpt-4o-mini"
    research_llm_temperature: float = 0.0
    max_sub_questions: int = 4          # hard cap on decomposition
    max_research_iterations: int = 3    # prevents infinite refinement loops
    evidence_min_score: float = 0.35    # minimum cosine similarity to consider a hit
    evidence_top_k: int = 5             # retrieved hits per sub-question

    # ── Conversations ──────────────────────────────────────────────────────────
    max_question_length: int = 2000     # hard cap on user question chars
    max_history_turns: int = 10         # turns (user+assistant pairs) loaded into graph


settings = Settings()
