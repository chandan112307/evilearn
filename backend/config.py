"""Application configuration."""

import os


class Settings:
    """Application settings loaded from environment variables."""

    # Database
    SQLITE_DB_PATH: str = os.environ.get("SQLITE_DB_PATH", "./evilearn.db")

    # Vector Store
    CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")

    # LLM
    LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "")
    LLM_MODEL: str = os.environ.get("LLM_MODEL", "llama3-8b-8192")
    LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "groq")  # groq or openai
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002")

    # Retrieval
    TOP_K_RESULTS: int = int(os.environ.get("TOP_K_RESULTS", "5"))

    # Upload
    MAX_FILE_SIZE_MB: int = int(os.environ.get("MAX_FILE_SIZE_MB", "50"))
    ALLOWED_EXTENSIONS: list[str] = [".pdf", ".txt"]

    # CORS
    CORS_ORIGINS: list[str] = os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")


settings = Settings()
