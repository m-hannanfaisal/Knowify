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

    # LLM API
    LLM_API_KEY: str = "placeholder_key"

    # Services
    QDRANT_URL: str = "http://localhost:6333"
    REDIS_URL: str = "redis://localhost:6379/0"


# Instantiate settings to be imported across the application
settings = Settings()
