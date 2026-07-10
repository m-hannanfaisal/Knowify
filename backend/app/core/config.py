from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings class loaded from environment variables and dotenv file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Configuration
    PROJECT_NAME: str = "RAG Chatbot API"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"

    # JWT Security Configuration
    JWT_SECRET: str = "super_secret_signing_key_change_me_in_production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ISSUER_URL: str | None = None

    # LLM API
    LLM_API_KEY: str = "placeholder_key"
    TAVILY_API_KEY: str = "placeholder_key"



    # Services
    QDRANT_URL: str = "http://localhost:6333"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Development & Deployment Modes
    # Note: docker/redis modes will be wired up in the final deployment phase.
    QDRANT_MODE: str = "local"  # "local" (embedded) or "docker" (networked service)
    QDRANT_LOCAL_PATH: str = "data/qdrant_local"
    CACHE_MODE: str = "memory"  # "memory" (in-memory dict) or "redis" (networked Redis service)



# Instantiate settings to be imported across the application
settings = Settings()
