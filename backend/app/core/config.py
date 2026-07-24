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
    ADMIN_EMAILS: str = "admin@example.com"



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


def get_llm_client_and_model(api_key: str | None = None, default_model: str = "gpt-4o-mini") -> tuple:
    """Returns a tuple of (AsyncOpenAI client, model_name).

    Automatically configures for Groq if key starts with 'gsk_'.
    """
    from openai import AsyncOpenAI
    key = api_key or settings.LLM_API_KEY
    if key and key.startswith("gsk_"):
        client = AsyncOpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        if "gpt" in default_model or default_model in ["gpt-4o-mini", "gpt-3.5-turbo"]:
            model = "llama-3.1-8b-instant"
        else:
            model = default_model
        return client, model
    else:
        client = AsyncOpenAI(api_key=key)
        return client, default_model

