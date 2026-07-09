import pytest
from unittest.mock import AsyncMock, patch

from app.core.config import settings
from app.ingestion.embeddings import MockEmbeddingProvider
from app.ingestion.splitter import DocumentChunk
from app.ingestion.store import QdrantStore
from app.retrieval.models import RetrievedChunk
from app.retrieval.hybrid import tokenize, reciprocal_rank_fusion
from app.retrieval.reranker import MockReranker
from app.retrieval.service import retrieve


def test_tokenize() -> None:
    """Test the basic alphanumeric tokenizer."""
    assert tokenize("Hello, World! RAG 2026.") == ["hello", "world", "rag", "2026"]


def test_reciprocal_rank_fusion() -> None:
    """Test RRF logic merges and orders dense and sparse lists correctly."""
    dense = [
        RetrievedChunk("Doc A", "fileA", "txt", 1, 0, 0.9),
        RetrievedChunk("Doc B", "fileB", "txt", 1, 1, 0.8),
        RetrievedChunk("Doc C", "fileC", "txt", 1, 2, 0.7),
    ]
    sparse = [
        RetrievedChunk("Doc C", "fileC", "txt", 1, 2, 5.0),
        RetrievedChunk("Doc A", "fileA", "txt", 1, 0, 4.0),
        RetrievedChunk("Doc B", "fileB", "txt", 1, 1, 3.0),
    ]

    fused = reciprocal_rank_fusion(dense, sparse, k=10)
    assert len(fused) == 3
    # Verify descending ordering
    assert fused[0].score >= fused[1].score >= fused[2].score


@pytest.mark.asyncio
async def test_mock_reranker() -> None:
    """Test MockReranker reranks chunks based on text query word overlap and length."""
    reranker = MockReranker()
    chunks = [
        RetrievedChunk("This is about apple pie.", "file1", "txt", 1, 0, 0.5),
        RetrievedChunk("This is about banana split dessert.", "file2", "txt", 1, 0, 0.5),
        RetrievedChunk("We love delicious hot apple pie recipe.", "file3", "txt", 1, 0, 0.5),
    ]

    # Query for "apple pie"
    reranked = await reranker.rerank("apple pie", chunks, top_k=2)
    assert len(reranked) == 2
    # "We love delicious hot apple pie recipe." or "This is about apple pie." should rank highest.
    # Words in query: {'apple', 'pie'}
    # Doc 1 overlap: {'apple', 'pie'} -> score 2.0 + len(24)/10000 = 2.0024
    # Doc 3 overlap: {'apple', 'pie'} -> score 2.0 + len(39)/10000 = 2.0039
    # Doc 2 overlap: {} -> score 0.0 + len/10000
    # Therefore, Doc 3 must rank first, Doc 1 second.
    assert reranked[0].source_filename == "file3"
    assert reranked[1].source_filename == "file1"


@pytest.mark.asyncio
async def test_hybrid_retrieval_integration() -> None:
    """Integration test: Populates Qdrant and runs full retrieve pipeline."""
    collection_name = "test_retrieval_collection"
    embedding_provider = MockEmbeddingProvider(dimension=64)
    reranker = MockReranker()

    store = QdrantStore(url=settings.QDRANT_URL)
    await store.ensure_collection(collection_name, 64)

    # Ingest some dummy documents
    chunks = [
        DocumentChunk("FastAPI is a modern async web framework for python.", "doc1", "txt", 1, 0),
        DocumentChunk("Qdrant is a vector database with hybrid search support.", "doc2", "txt", 1, 1),
        DocumentChunk("Redis is an in-memory database used for caching and sessions.", "doc3", "txt", 1, 2),
    ]
    texts = [c.text for c in chunks]
    embeddings = await embedding_provider.embed_documents(texts)
    await store.upsert_chunks(collection_name, chunks, embeddings)

    # Search for "FastAPI Python"
    results = await retrieve(
        query="FastAPI Python",
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        reranker=reranker,
        top_k=2,
        qdrant_url=settings.QDRANT_URL
    )

    assert len(results) <= 2
    # FastAPI doc should be in the results
    assert any("FastAPI" in r.text for r in results)

    # Clean up
    await store.client.delete_collection(collection_name)
