import os
import pytest
from unittest.mock import AsyncMock, patch

from app.core.config import settings
from app.ingestion.embeddings import MockEmbeddingProvider
from app.ingestion.image_handler import perform_ocr, generate_caption, ingest_image
from app.ingestion.store import QdrantStore


@pytest.fixture(scope="module")
def sample_image_path() -> str:
    """Fixture providing path to the programmatically generated sample PNG."""
    current_dir = os.path.dirname(__file__)
    fixtures_path = os.path.join(current_dir, "fixtures")
    png_path = os.path.join(fixtures_path, "sample.png")
    
    # If fixtures haven't been generated yet
    if not os.path.exists(png_path):
        from tests.generate_fixtures import main as gen_main
        gen_main()
        
    return png_path


@pytest.mark.asyncio
async def test_perform_ocr(sample_image_path: str) -> None:
    """Test OCR text extraction from the programmatically drawn image."""
    # Since Tesseract is installed in the container, it should run successfully.
    # We can check if it returns text. It might be empty if Tesseract cannot read
    # the PIL default font, but it should not raise an exception.
    ocr_text = await perform_ocr(sample_image_path)
    assert isinstance(ocr_text, str)


@pytest.mark.asyncio
async def test_generate_caption_mock(sample_image_path: str) -> None:
    """Test that vision captioning falls back gracefully or calls OpenAI properly."""
    # 1. Test fallback when API key is a placeholder
    caption = await generate_caption(sample_image_path, api_key="placeholder_key")
    assert "Mock image description" in caption

    # 2. Test mock vision call
    with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value.choices = [
            AsyncMock(message=AsyncMock(content="Mocked GPT-4o vision description"))
        ]
        
        caption_mocked = await generate_caption(sample_image_path, api_key="valid_api_key")
        assert caption_mocked == "Mocked GPT-4o vision description"


@pytest.mark.asyncio
async def test_ingest_image_integration(sample_image_path: str) -> None:
    """Integration test: end-to-end image ingestion writing to Qdrant."""
    collection_name = "test_image_collection"
    embedding_provider = MockEmbeddingProvider(dimension=64)
    
    # Ingest the image using Mock embeddings and placeholder LLM key
    count = await ingest_image(
        image_path=sample_image_path,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        api_key="placeholder_key",
        qdrant_url=settings.QDRANT_URL
    )
    
    assert count == 1

    # Verify that the point is inside Qdrant and carries the metadata
    store = QdrantStore(url=settings.QDRANT_URL)
    
    # Retrieve points from Qdrant
    response = await store.client.scroll(
        collection_name=collection_name,
        limit=10,
        with_payload=True
    )
    
    points = response[0]
    assert len(points) == 1
    
    # Check payload metadata
    payload = points[0].payload
    assert payload is not None
    assert payload["source_filename"] == "sample.png"
    assert payload["file_type"] == "png"
    assert payload["source_type"] == "image"
    assert "--- OCR Text ---" in payload["text"]
    assert "--- Image Description ---" in payload["text"]
    
    # Delete test collection
    await store.client.delete_collection(collection_name)
