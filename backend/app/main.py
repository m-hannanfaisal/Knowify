from fastapi import FastAPI
import structlog

from app.core.config import settings
from app.logging.logger import configure_logging

# Configure logging at startup
configure_logging()
logger = structlog.get_logger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint to verify backend service status.

    Returns:
        dict[str, str]: Service status
    """
    logger.info("health_check_called", status="healthy")
    return {"status": "healthy"}
