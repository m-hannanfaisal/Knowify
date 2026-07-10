import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app

client = TestClient(app)


def test_list_documents() -> None:
    """Test GET /admin/documents endpoint."""
    response = client.get("/admin/documents")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2
    assert data[0]["filename"] == "test_doc.docx"
    assert data[1]["file_type"] == "pdf"


def test_delete_document() -> None:
    """Test DELETE /admin/documents/{id} endpoint with mock Qdrant client delete."""
    with patch("app.api.admin_router.QdrantStore") as mock_store_class:
        mock_store = mock_store_class.return_value
        mock_store.client.delete = AsyncMock(return_value=True)

        response = client.delete("/admin/documents/test_doc.docx")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "deleted" in data["message"]


def test_delete_document_not_found() -> None:
    """Test DELETE /admin/documents/{id} returns 404 for missing documents."""
    response = client.delete("/admin/documents/nonexistent.docx")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_reindex_document() -> None:
    """Test POST /admin/documents/{id}/reindex endpoint."""
    response = client.post("/admin/documents/report.pdf/reindex")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "queued" in data["message"]


def test_list_conversations_unfiltered() -> None:
    """Test GET /admin/conversations without query parameters."""
    response = client.get("/admin/conversations")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["conversation_id"] == "session_api_123"
    assert data[0]["user_id"] == "test_user_api"


def test_list_conversations_filtered() -> None:
    """Test GET /admin/conversations with query parameters."""
    # 1. Matching filter
    resp_match = client.get("/admin/conversations?user_id=test_user_api")
    assert resp_match.status_code == 200
    assert len(resp_match.json()) == 1

    # 2. Non-matching filter
    resp_miss = client.get("/admin/conversations?user_id=nonexistent_user")
    assert resp_miss.status_code == 200
    assert len(resp_miss.json()) == 0


def test_get_conversation_trace() -> None:
    """Test GET /admin/conversations/{id}/trace endpoint."""
    response = client.get("/admin/conversations/session_api_123/trace")
    assert response.status_code == 200
    data = response.json()
    assert "steps" in data
    assert data["total_latency_ms"] == 625


def test_get_conversation_trace_not_found() -> None:
    """Test GET /admin/conversations/{id}/trace returns 404 for missing traces."""
    response = client.get("/admin/conversations/nonexistent_session/trace")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_user_memory() -> None:
    """Test GET /admin/memory/{user_id} endpoint."""
    with patch("app.api.admin_router.QdrantStore") as mock_store_class:
        mock_store = mock_store_class.return_value
        mock_record = AsyncMock()
        mock_record.id = 123
        mock_record.payload = {"text": "User lives in New York."}
        mock_store.client.scroll = AsyncMock(return_value=([mock_record], None))

        response = client.get("/admin/memory/user_id_1")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user_id_1"
        assert len(data["memories"]) == 1
        assert data["memories"][0]["text"] == "User lives in New York."


@pytest.mark.asyncio
async def test_delete_user_memory() -> None:
    """Test DELETE /admin/memory/{user_id}/{memory_id} endpoint."""
    with patch("app.api.admin_router.QdrantStore") as mock_store_class:
        mock_store = mock_store_class.return_value
        mock_store.client.delete = AsyncMock(return_value=True)

        response = client.delete("/admin/memory/user_id_1/123")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "deleted" in data["message"]


def test_get_costs() -> None:
    """Test GET /admin/costs endpoint."""
    response = client.get("/admin/costs")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert data[0]["user_id"] == "user_id_1"
    assert "cost" in data[0]


def test_get_user_usage() -> None:
    """Test GET /admin/usage/{user_id} endpoint."""
    response = client.get("/admin/usage/test_user_api")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test_user_api"
    assert data["tokens_limit"] == 5000
    assert "tokens_used" in data
    assert "resets_at" in data
    import datetime
    datetime.datetime.fromisoformat(data["resets_at"])


def test_get_user_usage_not_found() -> None:
    """Test GET /admin/usage/{user_id} returns 404 for missing usage profiles."""
    response = client.get("/admin/usage/nonexistent_user")
    assert response.status_code == 404


def test_update_user_limit() -> None:
    """Test PATCH /admin/usage/{user_id} endpoint."""
    response = client.patch("/admin/usage/test_user_api", json={"tokens_limit": 8000})
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test_user_api"
    assert data["tokens_limit"] == 8000
    assert "resets_at" in data
    import datetime
    datetime.datetime.fromisoformat(data["resets_at"])


