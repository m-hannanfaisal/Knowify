import pytest
from app.ingestion import store

@pytest.fixture(autouse=True)
def clear_qdrant_singleton() -> None:
    """Clears the global Qdrant client connection singleton before and after each test run.

    This forces a fresh client initialization within each test function's event loop,
    completely avoiding 'Event loop is closed' errors in pytest-asyncio.
    """
    store._qdrant_client_singleton = None
    yield
    store._qdrant_client_singleton = None
