import logging
import sys
from contextvars import ContextVar
from typing import Any
import structlog

# Global thread-safe ContextVar to store trace ID
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def add_trace_id(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Structlog processor that injects the current request's trace_id from ContextVar."""
    tid = trace_id_var.get()
    if tid:
        event_dict["trace_id"] = tid
    return event_dict


def configure_logging() -> None:
    """Configures structlog to output structured JSON logs to stdout with trace ID propagation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            add_trace_id,  # Contextvar processor
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Disable standard logging handlers if they interfere
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.error").handlers = []
