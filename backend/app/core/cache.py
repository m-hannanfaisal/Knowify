import asyncio
import json
import structlog
from typing import Any, Optional
import redis

from app.core.config import settings

logger = structlog.get_logger(__name__)


class BaseCache:
    """Interface defining caching client contract for get, set, and delete operations."""

    async def get(self, key: str) -> Optional[Any]:
        """Retrieves a value from the cache.

        Args:
            key (str): Key target.

        Returns:
            Optional[Any]: Deserialized value, or None if missing/expired.
        """
        raise NotImplementedError()

    async def set(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> None:
        """Stores a value in the cache with an optional TTL.

        Args:
            key (str): Key target.
            value (Any): Serialized or raw value.
            expire_seconds (Optional[int]): Time-to-live expiration duration.
        """
        raise NotImplementedError()

    async def delete(self, key: str) -> None:
        """Deletes a key-value entry from the cache.

        Args:
            key (str): Key target.
        """
        raise NotImplementedError()


class MemoryCache(BaseCache):
    """In-memory dict-based cache implementation for fast offline local development."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        logger.info("initialized_in_memory_cache")

    async def get(self, key: str) -> Optional[Any]:
        return self._data.get(key)

    async def set(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> None:
        # Development local memory cache stores values directly without TTL enforcement
        self._data[key] = value

    async def delete(self, key: str) -> None:
        if key in self._data:
            del self._data[key]


class RedisCache(BaseCache):
    """Redis-backed cache client.

    Note: This service will be wired up during the final deployment phase.
    """

    def __init__(self) -> None:
        self.client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        logger.info("initialized_redis_cache_provider")

    async def get(self, key: str) -> Optional[Any]:
        # Offload blocking synchronous Redis call to thread pool as per coding rules
        val = await asyncio.to_thread(self.client.get, key)
        if val is None:
            return None
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return val

    async def set(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> None:
        val_str = json.dumps(value) if not isinstance(value, str) else value
        await asyncio.to_thread(self.client.set, name=key, value=val_str, ex=expire_seconds)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self.client.delete, key)


def get_cache_provider() -> BaseCache:
    """Instantiates the appropriate cache provider based on Settings config."""
    if settings.CACHE_MODE == "redis":
        return RedisCache()
    return MemoryCache()


# Export singleton instance for app-wide import
cache = get_cache_provider()
