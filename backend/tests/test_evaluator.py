import pytest
from unittest.mock import AsyncMock, patch

from app.core.config import settings
from app.ingestion.embeddings import MockEmbeddingProvider
from app.ingestion.splitter import DocumentChunk
from app.ingestion.store import QdrantStore
from app.retrieval.reranker import MockReranker
from app.retrieval.models import RetrievedChunk
from app.evaluation.relevance_evaluator import evaluate_relevance
from app.orchestrator.service import handle_query


@pytest.mark.asyncio
async def test_evaluate_relevance_mock() -> None:
    """Test evaluate_relevance offline rules and mock LLM json parsing."""
    chunks = [RetrievedChunk("Some text", "source.txt", "txt", 1, 0, 0.9)]

    # 1. Test offline rules
    res_sufficient = await evaluate_relevance("Normal query", "Normal search", chunks, api_key="placeholder_key")
    assert res_sufficient["sufficient"] is True

    res_insufficient = await evaluate_relevance("force_insufficient", "Normal search", chunks, api_key="placeholder_key")
    assert res_insufficient["sufficient"] is False
    assert res_insufficient["feedback_for_rewrite"] == "needs X"

    # 2. Test mock completions json parsing
    with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value.choices = [
            AsyncMock(message=AsyncMock(content='{"sufficient": false, "reasoning": "Missing detail", "feedback_for_rewrite": "needs Y"}'))
        ]
        res_llm = await evaluate_relevance("OpenAI query", "OpenAI search", chunks, api_key="valid_api_key")
        assert res_llm["sufficient"] is False
        assert res_llm["reasoning"] == "Missing detail"
        assert res_llm["feedback_for_rewrite"] == "needs Y"


@pytest.mark.asyncio
async def test_orchestrator_retry_loop_success_first_try() -> None:
    """Verifies that retrieval succeeds on first try if sufficient."""
    collection_name = "test_retry_first_try"
    embedding_provider = MockEmbeddingProvider(dimension=64)
    reranker = MockReranker()

    store = QdrantStore(url=settings.QDRANT_URL)
    await store.ensure_collection(collection_name, 64)

    # Ingest document
    chunks = [
        DocumentChunk("FastAPI documentation: Use APIRouter.", "doc1", "txt", 1, 0),
    ]
    texts = [c.text for c in chunks]
    embeddings = await embedding_provider.embed_documents(texts)
    await store.upsert_chunks(collection_name, chunks, embeddings)

    # Query routes to RAG and is sufficient immediately
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

    assert res["route"] == "rag"
    assert res["insufficient_information"] is False
    assert len(res["retrieved_chunks"]) > 0
    assert res["fallback_response"] is None

    await store.client.delete_collection(collection_name)


@pytest.mark.asyncio
async def test_orchestrator_retry_loop_success_on_retry() -> None:
    """Verifies query rewriting and re-retrieval succeeds after one retry."""
    collection_name = "test_retry_success_retry"
    embedding_provider = MockEmbeddingProvider(dimension=64)
    reranker = MockReranker()

    store = QdrantStore(url=settings.QDRANT_URL)
    await store.ensure_collection(collection_name, 64)

    # Ingest document
    chunks = [
        DocumentChunk("FastAPI documentation: Use APIRouter.", "doc1", "txt", 1, 0),
    ]
    texts = [c.text for c in chunks]
    embeddings = await embedding_provider.embed_documents(texts)
    await store.upsert_chunks(collection_name, chunks, embeddings)

    # Query containing 'sufficient_on_retry' triggers mock evaluator to return sufficient=False on 1st try,
    # then sufficient=True on 2nd try (since the rewritten query will have the feedback context injected)
    res = await handle_query(
        query="docs sufficient_on_retry",
        conversation_history=[],
        long_term_memory=[],
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        reranker=reranker,
        api_key="placeholder_key",
        qdrant_url=settings.QDRANT_URL
    )

    assert res["route"] == "rag"
    assert res["insufficient_information"] is False
    # Check that query rewriter was called with feedback and rewritten query has the feedback
    assert "Feedback: needs X" in res["rewritten_query"] or "needs X" in res["rewritten_query"]
    assert res["fallback_response"] is None

    await store.client.delete_collection(collection_name)


@pytest.mark.asyncio
async def test_orchestrator_retry_loop_exhausted_fallback() -> None:
    """Verifies that after exhausting retries, the orchestrator falls back to a safe message."""
    collection_name = "test_retry_fallback"
    embedding_provider = MockEmbeddingProvider(dimension=64)
    reranker = MockReranker()

    store = QdrantStore(url=settings.QDRANT_URL)
    await store.ensure_collection(collection_name, 64)

    # Ingest document
    chunks = [
        DocumentChunk("FastAPI documentation: Use APIRouter.", "doc1", "txt", 1, 0),
    ]
    texts = [c.text for c in chunks]
    embeddings = await embedding_provider.embed_documents(texts)
    await store.upsert_chunks(collection_name, chunks, embeddings)

    # Query containing 'always_insufficient' triggers mock evaluator to return sufficient=False continuously
    res = await handle_query(
        query="docs always_insufficient",
        conversation_history=[],
        long_term_memory=[],
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        reranker=reranker,
        api_key="placeholder_key",
        qdrant_url=settings.QDRANT_URL
    )

    assert res["route"] == "rag"
    assert res["insufficient_information"] is True
    assert len(res["retrieved_chunks"]) == 0
    assert "I'm sorry, I cannot find enough relevant information" in res["fallback_response"]

    await store.client.delete_collection(collection_name)
