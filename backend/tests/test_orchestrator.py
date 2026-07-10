import pytest
from unittest.mock import AsyncMock, patch

from app.core.config import settings
from app.ingestion.embeddings import MockEmbeddingProvider
from app.ingestion.splitter import DocumentChunk
from app.ingestion.store import QdrantStore
from app.retrieval.reranker import MockReranker
from app.orchestrator.rewriter import query_rewriter
from app.orchestrator.router import route_query
from app.orchestrator.service import handle_query


@pytest.mark.asyncio
async def test_query_rewriter_mock() -> None:
    """Test rewriter falls back or calls OpenAI API completion correctly."""
    history = [
        {"role": "user", "content": "What is FastAPI?"},
        {"role": "assistant", "content": "FastAPI is a python web framework."},
    ]
    memory = ["User is building a RAG chatbot."]

    # 1. Test fallback when using placeholder key
    rewritten = await query_rewriter(
        query="Explain its features.",
        conversation_history=history,
        long_term_memory=memory,
        api_key="placeholder_key"
    )
    assert rewritten == "Mock Rewritten: Explain its features."

    # 2. Test mocked OpenAI client call
    with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value.choices = [
            AsyncMock(message=AsyncMock(content="Explain the features of FastAPI"))
        ]

        rewritten_mocked = await query_rewriter(
            query="Explain its features.",
            conversation_history=history,
            long_term_memory=memory,
            api_key="valid_api_key"
        )
        assert rewritten_mocked == "Explain the features of FastAPI"


@pytest.mark.asyncio
async def test_route_query_mock() -> None:
    """Test router classifies query types using offline rules or LLM completions."""
    # 1. Test offline rules when using placeholder key
    assert await route_query("What is the weather today?", api_key="placeholder_key") == "web_search"
    assert await route_query("How to search documentation?", api_key="placeholder_key") == "rag"
    assert await route_query("Hello there!", api_key="placeholder_key") == "direct"

    # 2. Test mocked OpenAI routing call
    with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value.choices = [
            AsyncMock(message=AsyncMock(content="rag"))
        ]

        decision = await route_query("Query requiring RAG database", api_key="valid_api_key")
        assert decision == "rag"


@pytest.mark.asyncio
async def test_handle_query_integration() -> None:
    """Integration test: verifies the orchestrated rewriter, router, and retriever pipeline."""
    collection_name = "test_orchestrator_collection"
    embedding_provider = MockEmbeddingProvider(dimension=64)
    reranker = MockReranker()

    store = QdrantStore(url=settings.QDRANT_URL)
    await store.ensure_collection(collection_name, 64)

    # Ingest document
    chunks = [
        DocumentChunk("FastAPI documentation: Use APIRouter to group API endpoints.", "doc1", "txt", 1, 0),
    ]
    texts = [c.text for c in chunks]
    embeddings = await embedding_provider.embed_documents(texts)
    await store.upsert_chunks(collection_name, chunks, embeddings)

    history = []
    memory = []

    # 1. RAG query integration
    res_rag = await handle_query(
        query="Explain FastAPI documentation routing",
        conversation_history=history,
        long_term_memory=memory,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        reranker=reranker,
        api_key="placeholder_key",
        qdrant_url=settings.QDRANT_URL
    )

    assert res_rag["route"] == "rag"
    assert len(res_rag["retrieved_chunks"]) > 0
    assert "FastAPI" in res_rag["retrieved_chunks"][0].text

    # 2. Direct query integration
    res_direct = await handle_query(
        query="Hello, how are you?",
        conversation_history=history,
        long_term_memory=memory,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        reranker=reranker,
        api_key="placeholder_key",
        qdrant_url=settings.QDRANT_URL
    )

    assert res_direct["route"] == "direct"
    assert len(res_direct["retrieved_chunks"]) == 0

    # 3. Web Search query integration
    res_web = await handle_query(
        query="What is the current weather in Paris?",
        conversation_history=history,
        long_term_memory=memory,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        reranker=reranker,
        api_key="placeholder_key",
        qdrant_url=settings.QDRANT_URL
    )

    assert res_web["route"] == "web_search"
    assert len(res_web["retrieved_chunks"]) == 0

    # Clean up
    await store.client.delete_collection(collection_name)


@pytest.mark.asyncio
async def test_orchestrator_retry_graph_traversal() -> None:
    """Verifies that the retry loop correctly traverses the graph edges when evaluation fails once."""
    collection_name = "test_graph_traversal"
    embedding_provider = MockEmbeddingProvider(dimension=64)
    reranker = MockReranker()

    store = QdrantStore(url=settings.QDRANT_URL)
    await store.ensure_collection(collection_name, 64)

    # Ingest document
    chunks = [
        DocumentChunk("FastAPI documentation info.", "doc1", "txt", 1, 0),
    ]
    texts = [c.text for c in chunks]
    embeddings = await embedding_provider.embed_documents(texts)
    await store.upsert_chunks(collection_name, chunks, embeddings)

    # Mock evaluate_relevance to return insufficient on the first attempt and sufficient on the second
    mock_evals = [
        {"sufficient": False, "reasoning": "First attempt insufficient", "feedback_for_rewrite": "needs X"},
        {"sufficient": True, "reasoning": "Second attempt sufficient", "feedback_for_rewrite": ""}
    ]

    with patch("app.orchestrator.service.evaluate_relevance", new_callable=AsyncMock) as mock_eval:
        mock_eval.side_effect = mock_evals

        res = await handle_query(
            query="Explain FastAPI documentation",
            conversation_history=[],
            long_term_memory=[],
            collection_name=collection_name,
            embedding_provider=embedding_provider,
            reranker=reranker,
            api_key="placeholder_key",
            qdrant_url=settings.QDRANT_URL
        )

        # Confirm that evaluate_relevance was called twice (initial + one retry)
        assert mock_eval.call_count == 2
        assert res["route"] == "rag"
        assert res["insufficient_information"] is False
        assert len(res["retrieved_chunks"]) > 0
        assert "Feedback: needs X" in res["rewritten_query"] or "needs X" in res["rewritten_query"]

    await store.client.delete_collection(collection_name)

