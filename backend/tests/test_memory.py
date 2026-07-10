import pytest
from unittest.mock import AsyncMock, patch

from app.core.config import settings
from app.ingestion.embeddings import MockEmbeddingProvider
from app.ingestion.store import QdrantStore
from app.memory.long_term import extract_memories, store_memories, retrieve_memories, consolidate_memories


class SameVectorMockEmbeddingProvider(MockEmbeddingProvider):
    """Subclass of MockEmbeddingProvider that returns identical vectors to simulate high similarity (>0.82) in tests."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Returns identical normalized vectors for any input text
        vector_dim = self.dimension
        base_vector = [1.0 / (vector_dim**0.5)] * vector_dim
        return [base_vector for _ in texts]


@pytest.mark.asyncio
async def test_extract_memories() -> None:
    """Test memory extraction under mock LLM completions."""
    # 1. Test fallback when using placeholder key
    memories_fallback = await extract_memories("Mock transcript", "user1", api_key="placeholder_key")
    assert len(memories_fallback) == 2
    assert "Python" in memories_fallback[0]

    # 2. Test mocked OpenAI chat completion call
    with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value.choices = [
            AsyncMock(message=AsyncMock(content='{"memories": ["User likes coffee.", "User works from home."]}'))
        ]
        memories = await extract_memories("User: I love drinking coffee. Assistant: Great.", "user1", api_key="valid_key")
        assert len(memories) == 2
        assert "coffee" in memories[0]
        assert "home" in memories[1]


@pytest.mark.asyncio
async def test_store_and_retrieve_memories() -> None:
    """Test storing and semantically retrieving memories round-trip."""
    user_id = "test_user_store"
    collection_name = f"memories_{user_id}"
    embedding_provider = MockEmbeddingProvider(dimension=64)

    store = QdrantStore(url=settings.QDRANT_URL)
    await store.ensure_collection(collection_name, 64)

    # Ingest memories
    memories = [
        "User plays tennis on weekends.",
        "User prefers dark chocolate over milk chocolate.",
    ]
    await store_memories(user_id, memories, embedding_provider, qdrant_url=settings.QDRANT_URL)

    # Retrieve memories
    retrieved = await retrieve_memories(
        user_id, "tennis", embedding_provider, top_k=1, qdrant_url=settings.QDRANT_URL
    )
    assert len(retrieved) == 1
    # Check that retrieved contains one of our memories
    assert any("tennis" in r.lower() for r in retrieved)

    # Clean up
    await store.client.delete_collection(collection_name)


@pytest.mark.asyncio
async def test_consolidate_memories() -> None:
    """Test memory consolidation merging duplicate fixtures using custom embedding provider."""
    user_id = "test_user_consolidate"
    collection_name = f"memories_{user_id}"
    embedding_provider = SameVectorMockEmbeddingProvider(dimension=64)

    store = QdrantStore(url=settings.QDRANT_URL)
    await store.ensure_collection(collection_name, 64)

    # Store near-duplicate memories (they will get same vector from our mock provider)
    memories = [
        "User is a python software engineer.",
        "User works as a python developer.",
    ]
    await store_memories(user_id, memories, embedding_provider, qdrant_url=settings.QDRANT_URL)

    # Verify we have 2 memories initially
    scroll_res_before = await store.client.scroll(collection_name, limit=10)
    assert len(scroll_res_before[0]) == 2

    # Run consolidation (Mock merge picks longest text: "User is a python software engineer.")
    await consolidate_memories(
        user_id,
        embedding_provider=embedding_provider,
        api_key="placeholder_key",
        qdrant_url=settings.QDRANT_URL
    )

    # Verify they were merged into 1 memory
    scroll_res_after = await store.client.scroll(collection_name, limit=10)
    records = scroll_res_after[0]
    assert len(records) == 1
    assert records[0].payload["text"] == "User is a python software engineer."

    # Clean up
    await store.client.delete_collection(collection_name)
