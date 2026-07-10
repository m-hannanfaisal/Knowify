import logging
import sys
import re
from contextvars import ContextVar
from typing import Any
import structlog

# Global thread-safe ContextVar to store trace ID
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

EMAIL_REGEX = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_REGEX = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b|\b(?:\+?\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b')


def redact_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
    text = PHONE_REGEX.sub("[REDACTED_PHONE]", text)
    return text

def redact_pii(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Recursively redacts PII like email and phone numbers from all event fields."""
    for key, value in list(event_dict.items()):
        if isinstance(value, str):
            event_dict[key] = redact_text(value)
        elif isinstance(value, dict):
            event_dict[key] = redact_pii(logger, method_name, value)
        elif isinstance(value, list):
            event_dict[key] = [
                redact_text(item) if isinstance(item, str)
                else redact_pii(logger, method_name, item) if isinstance(item, dict)
                else item for item in value
            ]
    return event_dict


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
            redact_pii,    # Production PII redaction processor
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

