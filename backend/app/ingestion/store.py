import uuid
import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.core.config import settings
from app.ingestion.splitter import DocumentChunk

logger = structlog.get_logger(__name__)


class QdrantStore:
    """Handles async operations on the Qdrant vector database."""

    def __init__(self, url: str | None = None) -> None:
        self.url = url or settings.QDRANT_URL
        self.client = AsyncQdrantClient(url=self.url)

    async def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Checks if a collection exists, and creates it if not.

        Args:
            collection_name (str): Collection name.
            vector_size (int): Dimension of the vectors.
        """
        response = await self.client.get_collections()
        exists = any(c.name == collection_name for c in response.collections)
        if not exists:
            logger.info("creating_qdrant_collection", collection=collection_name, size=vector_size)
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    async def upsert_chunks(
        self, collection_name: str, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> None:
        """Upserts document chunks and their embeddings into Qdrant.

        Args:
            collection_name (str): Target collection.
            chunks (list[DocumentChunk]): Document chunks.
            embeddings (list[list[float]]): Corresponding vectors.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks list and embeddings list must have the same length")

        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, embeddings):
            # Generate deterministic UUID v5 to avoid duplication on re-indexing
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{chunk.source_filename}_{chunk.chunk_index}"))

            payload = {
                "text": chunk.text,
                "source_filename": chunk.source_filename,
                "file_type": chunk.file_type,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata,
            }


            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        if points:
            await self.client.upsert(collection_name=collection_name, points=points)
            logger.info("upserted_points_to_qdrant", count=len(points), collection=collection_name)
