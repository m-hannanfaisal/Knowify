import uuid
from fastapi import FastAPI, Request
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from app.core.config import settings
from app.core.metrics import metrics_tracker
from app.core.limiter import limiter
from app.logging.logger import configure_logging, trace_id_var
from app.api.router import router as api_router
from app.api.admin_router import admin_router

# Configure logging at startup
configure_logging()
logger = structlog.get_logger(__name__)


class TraceIdMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware to extract or generate Request/Trace IDs and propagate via structlog."""

    async def dispatch(self, request: Request, call_next):
        trace_id = (
            request.headers.get("X-Trace-ID")
            or request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )
        token = trace_id_var.set(trace_id)
        try:
            response = await call_next(request)
            response.headers["X-Trace-ID"] = trace_id
            return response
        finally:
            trace_id_var.reset(token)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
)

# Register rate limiter components on application state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Register Middleware
app.add_middleware(TraceIdMiddleware)

# Include API Router under both prefixed and root paths for maximum backwards compatibility
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(api_router)
app.include_router(admin_router, prefix="/admin")



@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint to verify backend service status.

    Returns:
        dict[str, str]: Service status
    """
    logger.info("health_check_called", status="healthy")
    return {"status": "healthy"}


@app.get("/metrics")
async def get_metrics() -> dict:
    """Telemetry endpoint exposing retrieval hit rates, cache hits, retries, and latencies.

    Returns:
        dict: Usage metrics and accuracy statistics.
    """
    metrics = metrics_tracker.get_metrics()
    logger.info("metrics_retrieved", total_requests=metrics["total_requests"])
    return metrics
