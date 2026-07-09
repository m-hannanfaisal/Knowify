import os
import time
import structlog

from app.ingestion.embeddings import BaseEmbeddingProvider
from app.ingestion.parser import DocumentParser, ParsedTable
from app.ingestion.splitter import DocumentChunk, RecursiveCharacterTextSplitter, TabularSplitter
from app.ingestion.store import QdrantStore

logger = structlog.get_logger(__name__)


async def embed_and_store(
    file_path: str,
    collection_name: str,
    embedding_provider: BaseEmbeddingProvider,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    row_group_size: int = 10,
    qdrant_url: str | None = None,
) -> int:
    """Parses a document, splits it into chunks, embeds, and stores the chunks in Qdrant.

    Args:
        file_path (str): Path to the document file.
        collection_name (str): Qdrant collection name.
        embedding_provider (BaseEmbeddingProvider): Pluggable embedding service.
        chunk_size (int): Size of character chunks.
        chunk_overlap (int): Overlap of character chunks.
        row_group_size (int): Number of rows per chunk for tabular files.
        qdrant_url (str | None): Optional custom Qdrant connection URL.

    Returns:
        int: The number of chunks successfully processed and stored.
    """
    start_time = time.perf_counter()
    filename = os.path.basename(file_path)
    file_type = os.path.splitext(filename)[1].lower().replace(".", "")

    logger.info("file_ingestion_received", filename=filename, file_type=file_type)

    # 1. Parse document
    parser = DocumentParser()
    parsed_output = parser.parse(file_path)

    # 2. Chunk documents
    chunks: list[DocumentChunk] = []
    if isinstance(parsed_output, ParsedTable):
        splitter = TabularSplitter(row_group_size=row_group_size)
        text_chunks = splitter.split_dataframe(parsed_output.df)
        for idx, text in enumerate(text_chunks):
            chunks.append(
                DocumentChunk(
                    text=text,
                    source_filename=filename,
                    file_type=file_type,
                    page_number=None,
                    chunk_index=idx,
                )
            )
    else:  # list[ParsedPage]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunk_index = 0
        for page in parsed_output:
            page_text_chunks = text_splitter.split_text(page.content)
            for text in page_text_chunks:
                chunks.append(
                    DocumentChunk(
                        text=text,
                        source_filename=filename,
                        file_type=file_type,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

    chunks_created_time = time.perf_counter()
    chunks_duration = int((chunks_created_time - start_time) * 1000)
    logger.info("chunks_created", count=len(chunks), duration_ms=chunks_duration)

    if not chunks:
        logger.info("no_chunks_to_store", filename=filename)
        return 0

    # 3. Embed text chunks
    texts = [chunk.text for chunk in chunks]
    embeddings = await embedding_provider.embed_documents(texts)

    # 4. Save to Qdrant
    store = QdrantStore(url=qdrant_url)
    await store.ensure_collection(collection_name, embedding_provider.dimension)
    await store.upsert_chunks(collection_name, chunks, embeddings)

    end_time = time.perf_counter()
    total_duration = int((end_time - start_time) * 1000)
    logger.info(
        "embeddings_stored",
        count=len(chunks),
        duration_ms=total_duration,
        filename=filename,
        collection=collection_name,
    )

    return len(chunks)
