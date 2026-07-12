import asyncio
import base64
import os
import time
import pytesseract
import structlog
from openai import AsyncOpenAI
from PIL import Image

from app.core.config import settings
from app.ingestion.embeddings import BaseEmbeddingProvider
from app.ingestion.splitter import DocumentChunk
from app.ingestion.store import QdrantStore

logger = structlog.get_logger(__name__)


async def perform_ocr(image_path: str) -> str:
    """Performs OCR on an image file asynchronously using pytesseract.

    Args:
        image_path (str): Absolute path to the image.

    Returns:
        str: Extracted text content.
    """

    def _run() -> str:
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            logger.warning("pytesseract_ocr_failed", error=str(e), image_path=image_path)
            # If tesseract is not installed on path (e.g. windows local host),
            # return a fallback string or raise depending on preference.
            # We return empty string so it doesn't crash the entire pipeline.
            return ""

    return await asyncio.to_thread(_run)


async def generate_caption(image_path: str, api_key: str | None = None) -> str:
    """Generates a natural-language caption/description using a vision-capable LLM.

    Args:
        image_path (str): Absolute path to the image.
        api_key (str | None): Optional OpenAI API key override.

    Returns:
        str: Generated caption describing the image content.
    """
    api_key = api_key or settings.LLM_API_KEY
    if api_key == "placeholder_key" or not api_key:
        logger.info("skip_vision_llm_call", reason="No valid LLM API key provided")
        return "Mock image description (no API key provided)"

    def _encode_image() -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    try:
        base64_image = await asyncio.to_thread(_encode_image)
        file_type = "image/png" if image_path.endswith(".png") else "image/jpeg"

        is_groq = api_key.startswith("gsk_")
        if is_groq:
            base_url = "https://api.groq.com/openai/v1"
            model = "llama-3.2-11b-vision-preview"
        else:
            base_url = "https://api.openai.com/v1"
            model = "gpt-4o-mini"

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe the content of this image in detail. "
                                "If there are charts, tables, or diagrams, explain what they show and summarize the data."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{file_type};base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        caption = response.choices[0].message.content or ""
        return caption.strip()

    except Exception as e:
        logger.error("vision_llm_call_failed", error=str(e), image_path=image_path)
        return f"Failed to generate caption: {e}"


async def ingest_image(
    image_path: str,
    collection_name: str,
    embedding_provider: BaseEmbeddingProvider,
    api_key: str | None = None,
    qdrant_url: str | None = None,
) -> int:
    """Ingests an image, extracts text via OCR and vision captioning, embeds, and stores in Qdrant.

    Args:
        image_path (str): Path to the image file.
        collection_name (str): Destination collection name.
        embedding_provider (BaseEmbeddingProvider): Pluggable embedding service.
        api_key (str | None): Optional OpenAI API key override.
        qdrant_url (str | None): Optional Qdrant connection URL.

    Returns:
        int: Number of chunks stored (always 1).
    """
    start_time = time.perf_counter()
    filename = os.path.basename(image_path)
    file_type = os.path.splitext(filename)[1].lower().replace(".", "")

    logger.info("image_ingestion_received", filename=filename, file_type=file_type)

    # Execute OCR and vision captioning in parallel
    ocr_task = perform_ocr(image_path)
    caption_task = generate_caption(image_path, api_key=api_key)

    ocr_text, caption = await asyncio.gather(ocr_task, caption_task)

    logger.info("ocr_completed", filename=filename, text_length=len(ocr_text))
    logger.info("caption_completed", filename=filename, caption_length=len(caption))

    # Combine OCR text and Caption
    combined_text = f"--- OCR Text ---\n{ocr_text}\n\n--- Image Description ---\n{caption}"

    # Create chunk with metadata
    chunk = DocumentChunk(
        text=combined_text,
        source_filename=filename,
        file_type=file_type,
        page_number=1,
        chunk_index=0,
        metadata={"source_type": "image"},
    )

    # Embed and store
    embeddings = await embedding_provider.embed_documents([combined_text])

    store = QdrantStore(url=qdrant_url)
    await store.ensure_collection(collection_name, embedding_provider.dimension)
    await store.upsert_chunks(collection_name, [chunk], embeddings)

    end_time = time.perf_counter()
    duration = int((end_time - start_time) * 1000)

    logger.info("image_ingested_successfully", filename=filename, duration_ms=duration)

    return 1
