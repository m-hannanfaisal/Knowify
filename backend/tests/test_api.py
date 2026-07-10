import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_chat_endpoint_stream() -> None:
    """Test the POST /chat SSE streaming endpoint with mocked downstream services."""
    # Define custom async generator to simulate generate_response yields
    async def mock_response_generator(*args, **kwargs):
        yield {"type": "token", "text": "FastAPI is "}
        yield {"type": "token", "text": "async."}
        yield {
            "type": "citations",
            "citations": [{"filename": "doc1.txt", "location": "chunk 0"}]
        }

    # Patch retrieval, orchestration graph, and generation stream
    with patch("app.api.router.retrieve_memories", new_callable=AsyncMock) as mock_memory, \
         patch("app.api.router.handle_query", new_callable=AsyncMock) as mock_graph, \
         patch("app.api.router.generate_response", side_effect=mock_response_generator) as mock_gen:

        mock_memory.return_value = ["User likes python."]
        mock_graph.return_value = {
            "rewritten_query": "What is FastAPI?",
            "route": "rag",
            "retrieved_chunks": [],
            "insufficient_information": False,
            "fallback_response": None
        }

        chat_payload = {
            "query": "Explain FastAPI async",
            "user_id": "test_user_api",
            "conversation_id": "session_api_123"
        }

        # Query the prefixed endpoint
        response = client.post("/api/v1/chat", json=chat_payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Read streaming SSE events response
        content = response.text
        assert "FastAPI is" in content
        assert "async." in content
        assert "citations" in content
        assert "doc1.txt" in content

        # Verify history preservation via GET /conversations
        history_response = client.get("/api/v1/conversations/session_api_123")
        assert history_response.status_code == 200
        history_data = history_response.json()
        assert history_data["conversation_id"] == "session_api_123"
        assert len(history_data["history"]) == 2
        assert history_data["history"][0]["content"] == "Explain FastAPI async"
        assert history_data["history"][1]["content"] == "FastAPI is async."


@pytest.mark.asyncio
async def test_upload_file_endpoint() -> None:
    """Test the POST /upload endpoint routing text and images to correct ingestion pipelines."""
    # 1. Test text file upload (routed to embed_and_store)
    with patch("app.api.router.embed_and_store", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = 5  # mock returns 5 chunks parsed

        file_payload = {"file": ("test_doc.docx", b"Mock word content text.", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        response = client.post("/api/v1/upload", files=file_payload, data={"collection_name": "test_collection"})

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test_doc.docx"
        assert data["chunks_processed"] == 5
        mock_embed.assert_called_once()

    # 2. Test image file upload (routed to ingest_image)
    with patch("app.api.router.ingest_image", new_callable=AsyncMock) as mock_ingest_image:
        mock_ingest_image.return_value = 1  # mock returns 1 chunk parsed

        image_payload = {"file": ("chart.png", b"Fake PNG bytes.", "image/png")}
        response = client.post("/api/v1/upload", files=image_payload, data={"collection_name": "test_collection"})

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "chart.png"
        assert data["chunks_processed"] == 1
        mock_ingest_image.assert_called_once()


def test_get_nonexistent_conversation() -> None:
    """Verifies that GET /conversations/ returns an empty list for sessions with no history."""
    response = client.get("/api/v1/conversations/nonexistent_session")
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == "nonexistent_session"
    assert data["history"] == []


def test_get_usage() -> None:
    """Test the GET /api/v1/usage endpoint."""
    response = client.get("/api/v1/usage")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "default_user"
    assert "tokens_limit" in data
    assert "tokens_used" in data
    assert "resets_at" in data
    import datetime
    # Should be valid ISO format
    datetime.datetime.fromisoformat(data["resets_at"])


def test_get_user_usage_api() -> None:
    """Test the GET /api/v1/usage/{user_id} endpoint."""
    response = client.get("/api/v1/usage/test_user_api")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test_user_api"
    assert data["tokens_limit"] in [5000, 8000]
    assert "tokens_used" in data
    assert "resets_at" in data
    import datetime
    datetime.datetime.fromisoformat(data["resets_at"])




