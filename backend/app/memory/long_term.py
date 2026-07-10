import asyncio
import json
import time
from typing import Any, Optional
import numpy as np
import structlog
from openai import AsyncOpenAI
from qdrant_client.models import PointIdsList
from langsmith import traceable

from app.core.config import settings
from app.ingestion.embeddings import BaseEmbeddingProvider
from app.ingestion.splitter import DocumentChunk
from app.ingestion.store import QdrantStore

logger = structlog.get_logger(__name__)


@traceable(run_type="llm", name="Knowify Extract Memories")
async def extract_memories(

    conversation_transcript: str, user_id: str, api_key: Optional[str] = None
) -> list[str]:
    """Extracts durable facts worth remembering about the user from a conversation transcript.

    Args:
        conversation_transcript (str): Session dialogue transcript.
        user_id (str): Associated user ID.
        api_key (Optional[str]): Optional OpenAI API key override.

    Returns:
        list[str]: Extracted facts.
    """
    api_key = api_key or settings.LLM_API_KEY
    if api_key == "placeholder_key" or not api_key:
        logger.info("skip_memory_extraction_llm_call", user_id=user_id)
        # Mock memory extraction for testing
        return ["User likes Python programming language.", "User prefers dark mode UI styling."]

    client = AsyncOpenAI(api_key=api_key)

    system_prompt = (
        "You are an expert AI user memory manager.\n"
        "Review the following conversation transcript between a User and an Assistant.\n"
        "Identify and extract durable facts worth remembering about the User (e.g. user preferences, "
        "stated interests, recurring topics,stated facts about the user, work domain, goals).\n"
        "Do not extract temporary details, greeting phrases, or generic conversation turns.\n"
        "Respond with a JSON object containing a single key 'memories' which is a list of strings. "
        "Do not include any explanations or formatting markdown."
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_transcript},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=250,
        )
        res_dict = json.loads(response.choices[0].message.content or "{}")
        memories = [str(m).strip() for m in res_dict.get("memories", [])]
        logger.info("memories_extracted", count=len(memories), user_id=user_id)
        return memories
    except Exception as e:
        logger.error("memory_extraction_failed", error=str(e), user_id=user_id)
        return []


@traceable(run_type="retriever", name="Knowify Store Memories")
async def store_memories(
    user_id: str,
    memories: list[str],

    embedding_provider: BaseEmbeddingProvider,
    qdrant_url: Optional[str] = None,
) -> None:
    """Embeds each memory string and upserts them into a namespaced collection for the user.

    Args:
        user_id (str): Target user ID.
        memories (list[str]): Memories to store.
        embedding_provider (BaseEmbeddingProvider): Embedding generator service.
        qdrant_url (Optional[str]): Optional Qdrant connection URL.
    """
    if not memories:
        return

    collection_name = f"memories_{user_id}"
    store = QdrantStore(url=qdrant_url)

    # Ensure collection exists
    await store.ensure_collection(collection_name, embedding_provider.dimension)

    # Generate document chunks representing memories to reuse store.upsert_chunks
    chunks = [
        DocumentChunk(
            text=memory,
            source_filename=f"memory_{user_id}",
            file_type="memory",
            page_number=None,
            chunk_index=i,
        )
        for i, memory in enumerate(memories)
    ]

    embeddings = await embedding_provider.embed_documents(memories)
    await store.upsert_chunks(collection_name, chunks, embeddings)
    logger.info("memories_stored", count=len(memories), user_id=user_id)


@traceable(run_type="retriever", name="Knowify Retrieve Memories")
async def retrieve_memories(
    user_id: str,
    query: str,

    embedding_provider: BaseEmbeddingProvider,
    top_k: int = 5,
    qdrant_url: Optional[str] = None,
) -> list[str]:
    """Retrieves relevant semantic memories for the user matching the query.

    Args:
        user_id (str): Associated user ID.
        query (str): The search query.
        embedding_provider (BaseEmbeddingProvider): Embedding generator service.
        top_k (int): Retrieve limit.
        qdrant_url (Optional[str]): Optional Qdrant connection URL.

    Returns:
        list[str]: Matching memory facts.
    """
    start_time = time.perf_counter()
    collection_name = f"memories_{user_id}"
    store = QdrantStore(url=qdrant_url)

    # Check if collection exists
    try:
        response = await store.client.get_collections()
        exists = any(c.name == collection_name for c in response.collections)
        if not exists:
            logger.info("memory_collection_not_found_on_retrieve", user_id=user_id)
            return []
    except Exception as e:
        logger.error("check_collection_failed_retrieve", error=str(e), user_id=user_id)
        return []

    # Embed search query
    query_vectors = await embedding_provider.embed_documents([query])
    query_vector = query_vectors[0]

    try:
        hits = await store.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        memories = []
        for hit in hits.points:
            if hit.payload and "text" in hit.payload:
                memories.append(hit.payload["text"])

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "memories_retrieved", count=len(memories), user_id=user_id, latency_ms=latency_ms
        )
        return memories
    except Exception as e:
        logger.error("memory_retrieval_failed", error=str(e), user_id=user_id)
        return []


@traceable(run_type="chain", name="Knowify Consolidate Memories")
async def consolidate_memories(
    user_id: str,
    embedding_provider: BaseEmbeddingProvider,

    api_key: Optional[str] = None,
    qdrant_url: Optional[str] = None,
) -> None:
    """Merges overlapping or similar memories (similarity > 0.82) for a user to consolidate profiles.

    Args:
        user_id (str): Target user ID.
        embedding_provider (BaseEmbeddingProvider): Embedding generator service.
        api_key (Optional[str]): Optional OpenAI API key override.
        qdrant_url (Optional[str]): Optional Qdrant connection URL.
    """
    collection_name = f"memories_{user_id}"
    store = QdrantStore(url=qdrant_url)

    # Check if collection exists
    try:
        response = await store.client.get_collections()
        exists = any(c.name == collection_name for c in response.collections)
        if not exists:
            logger.info("no_memories_collection_to_consolidate", user_id=user_id)
            return
    except Exception as e:
        logger.error("check_collection_failed_consolidate", error=str(e), user_id=user_id)
        return

    # Scroll to load all existing memories
    try:
        scroll_res = await store.client.scroll(
            collection_name=collection_name, limit=1000, with_payload=True, with_vectors=True
        )
        records = scroll_res[0]
    except Exception as e:
        logger.error("scroll_memories_failed", error=str(e), user_id=user_id)
        return

    before_count = len(records)
    if before_count < 2:
        logger.info("skip_consolidation_too_few_memories", count=before_count, user_id=user_id)
        return

    # Graph clustering based on cosine similarity > 0.82
    similarity_threshold = 0.82
    parent = list(range(before_count))

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Compare all pairs
    for i in range(before_count):
        for j in range(i + 1, before_count):
            v1 = records[i].vector
            v2 = records[j].vector
            if v1 is None or v2 is None:
                continue

            dot = np.dot(v1, v2)
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            sim = float(dot / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0

            if sim >= similarity_threshold:
                union(i, j)

    # Group record indices by root
    groups: dict[int, list[int]] = {}
    for i in range(before_count):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    consolidated_memories = []
    points_to_delete = []

    api_key = api_key or settings.LLM_API_KEY
    use_mock = api_key == "placeholder_key" or not api_key

    for root, indices in groups.items():
        if len(indices) < 2:
            continue

        group_memories = [records[idx].payload["text"] for idx in indices]
        group_ids = [records[idx].id for idx in indices]
        points_to_delete.extend(group_ids)

        merged = ""
        if use_mock:
            # Mock merge logic: pick longest text representing detailed statement
            merged = max(group_memories, key=len)
        else:
            client = AsyncOpenAI(api_key=api_key)
            system_prompt = (
                "You are an expert AI user memory consolidator.\n"
                "Review the following list of overlapping or near-duplicate user memory entries.\n"
                "Merge them into a single, clean, cohesive, and concise memory entry that retains all the important facts.\n"
                "Respond with a JSON object containing a single key 'merged_memory' which is a string. "
                "Do not include any explanations or markdown."
            )
            user_content = "Memories to merge:\n" + "\n".join([f"- {m}" for m in group_memories])
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=150,
                )
                res_dict = json.loads(response.choices[0].message.content or "{}")
                merged = res_dict.get("merged_memory", group_memories[0]).strip()
            except Exception as e:
                logger.error("merge_memories_failed", error=str(e), user_id=user_id)
                merged = group_memories[0]

        consolidated_memories.append(merged)

    # Perform atomic database clean-ups
    if points_to_delete:
        await store.client.delete(
            collection_name=collection_name, points_selector=PointIdsList(points=points_to_delete)
        )
        logger.info("deleted_duplicate_memories", count=len(points_to_delete), user_id=user_id)

    if consolidated_memories:
        await store_memories(
            user_id=user_id,
            memories=consolidated_memories,
            embedding_provider=embedding_provider,
            qdrant_url=qdrant_url,
        )

    after_count = before_count - len(points_to_delete) + len(consolidated_memories)
    logger.info(
        "memory_consolidation_complete",
        user_id=user_id,
        before_count=before_count,
        after_count=after_count,
    )
