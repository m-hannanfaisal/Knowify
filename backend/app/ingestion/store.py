import asyncio
import uuid
from typing import Any, Optional
import structlog
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.core.config import settings
from app.ingestion.splitter import DocumentChunk

logger = structlog.get_logger(__name__)


class AsyncQdrantLocalWrapper:
    """An asynchronous wrapper around the synchronous QdrantClient for local embedded mode.

    This offloads all blocking synchronous calls to a background thread pool, maintaining
    compliance with async I/O rules and matching AsyncQdrantClient's interface.
    """

    def __init__(self, path: str) -> None:
        self._sync_client = QdrantClient(path=path)
        logger.info("initialized_local_embedded_qdrant_wrapper", path=path)

    async def get_collections(self, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self._sync_client.get_collections, **kwargs)

    async def create_collection(
        self, collection_name: str, vectors_config: Any, **kwargs: Any
    ) -> Any:
        return await asyncio.to_thread(
            self._sync_client.create_collection,
            collection_name=collection_name,
            vectors_config=vectors_config,
            **kwargs,
        )

    async def upsert(self, collection_name: str, points: list[Any], **kwargs: Any) -> Any:
        return await asyncio.to_thread(
            self._sync_client.upsert,
            collection_name=collection_name,
            points=points,
            **kwargs,
        )

    async def query_points(
        self, collection_name: str, query: Any, limit: int, with_payload: bool = True, **kwargs: Any
    ) -> Any:
        return await asyncio.to_thread(
            self._sync_client.query_points,
            collection_name=collection_name,
            query=query,
            limit=limit,
            with_payload=with_payload,
            **kwargs,
        )

    async def scroll(
        self, collection_name: str, limit: int = 10, with_payload: bool = True, **kwargs: Any
    ) -> Any:
        return await asyncio.to_thread(
            self._sync_client.scroll,
            collection_name=collection_name,
            limit=limit,
            with_payload=with_payload,
            **kwargs,
        )

    async def delete_collection(self, collection_name: str, **kwargs: Any) -> Any:
        return await asyncio.to_thread(
            self._sync_client.delete_collection,
            collection_name=collection_name,
            **kwargs,
        )

    async def delete(self, collection_name: str, points_selector: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(
            self._sync_client.delete,
            collection_name=collection_name,
            points_selector=points_selector,
            **kwargs,
        )



# Global singleton instance for Qdrant client connection to avoid SQLite database locks
_qdrant_client_singleton: Optional[Any] = None


def get_qdrant_client() -> Any:
    """Returns a shared global singleton Qdrant Client connection instance."""
    global _qdrant_client_singleton
    if _qdrant_client_singleton is None:
        mode = settings.QDRANT_MODE
        if mode == "docker":
            # Note: docker (networked) mode will be wired up during final deployment phase
            _qdrant_client_singleton = AsyncQdrantClient(url=settings.QDRANT_URL)
        else:
            # Local embedded mode
            _qdrant_client_singleton = AsyncQdrantLocalWrapper(path=settings.QDRANT_LOCAL_PATH)
    return _qdrant_client_singleton


class QdrantStore:
    """Handles operations on the Qdrant vector database, routing to local embedded or docker mode."""

    def __init__(self, url: str | None = None) -> None:
        self.url = url or settings.QDRANT_URL
        self.mode = settings.QDRANT_MODE
        self.client = get_qdrant_client()

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
            logger.info(
                "upserted_points_to_qdrant",
                count=len(points),
                collection=collection_name,
                mode=self.mode,
            )
