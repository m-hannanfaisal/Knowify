import pytest
from unittest.mock import AsyncMock, patch

from app.core.config import settings
from app.ingestion.embeddings import MockEmbeddingProvider
from app.ingestion.splitter import DocumentChunk
from app.ingestion.store import QdrantStore
from app.orchestrator.mcp.client import call_mcp_tool
from app.orchestrator.mcp.server import query_knowledge_base


@pytest.mark.asyncio
async def test_call_mcp_tool_fallback_mock() -> None:
    """Verifies that call_mcp_tool falls back to local mocks if configuration is missing or commands fail."""
    # 1. Missing server configuration -> mock fallback
    res_missing = await call_mcp_tool("nonexistent_server", "tool_name", {"param": "val"})
    assert "Mock response" in res_missing

    # 2. Configured server, but fails to launch stdio -> mock fallback
    # The config has 'filesystem' registered, but npx/node might fail or connection timeout
    res_filesystem = await call_mcp_tool("filesystem", "read_file", {"path": "test.txt"})
    assert "Mock filesystem" in res_filesystem

    res_search = await call_mcp_tool("web-search", "google_search", {"query": "weather"})
    assert "Mock Search results" in res_search


@pytest.mark.asyncio
async def test_call_mcp_tool_session_mock() -> None:
    """Test client stdio_client session setup and tool invocation flow using mocks."""
    # Setup mock Tavily/Filesystem MCP connection response
    mock_mcp_response = AsyncMock()
    mock_mcp_response.content = [AsyncMock(type="text", text="Mock stdio connection result")]

    with patch("app.orchestrator.mcp.client.stdio_client") as mock_stdio, \
         patch("app.orchestrator.mcp.client.ClientSession") as mock_session_class:

        # Setup sessions mocks
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = mock_mcp_response
        mock_session_class.return_value.__aenter__.return_value = mock_session

        # Setup stdio mock streams
        mock_stdio.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())

        res = await call_mcp_tool("filesystem", "read_file", {"path": "test.txt"})
        assert res == "Mock stdio connection result"


@pytest.mark.asyncio
async def test_mcp_server_rag_tool_smoke() -> None:
    """Smoke test: starts RAG tool query_knowledge_base against a populated local test corpus."""
    collection_name = "test_mcp_server_collection"
    embedding_provider = MockEmbeddingProvider(dimension=64)

    store = QdrantStore(url=settings.QDRANT_URL)
    await store.ensure_collection(collection_name, 64)

    # Ingest document
    chunks = [
        DocumentChunk("FastAPI documentation info: Use APIRouter to group endpoints.", "doc1", "txt", 1, 0),
    ]
    texts = [c.text for c in chunks]
    embeddings = await embedding_provider.embed_documents(texts)
    await store.upsert_chunks(collection_name, chunks, embeddings)

    # Call query_knowledge_base tool directly (coroutine)
    res = await query_knowledge_base("Explain FastAPI routing", collection_name=collection_name)

    assert "FastAPI" in res
    assert "doc1" in res
    assert "Mock RAG Answer" in res  # Verified offline mock answer format

    # Clean up
    await store.client.delete_collection(collection_name)
