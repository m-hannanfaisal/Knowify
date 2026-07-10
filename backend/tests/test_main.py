import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_check() -> None:
    """Test the /health endpoint to ensure the service is running and healthy."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.asyncio
async def test_metrics_endpoint() -> None:
    """Test the /metrics endpoint to ensure telemetry is exposed correctly."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "retrieval_hit_rate" in data
    assert "cache_hit_rate" in data


@pytest.mark.asyncio
async def test_trace_id_middleware() -> None:
    """Test that requests automatically receive X-Trace-ID response headers."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health", headers={"X-Trace-ID": "test-id-123"})
    assert response.status_code == 200
    assert response.headers.get("X-Trace-ID") == "test-id-123"


