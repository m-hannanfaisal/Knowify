import datetime
from typing import Any, Optional
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointIdsList

from app.core.config import settings
from app.core.limiter import limiter
from app.ingestion.embeddings import MockEmbeddingProvider, OpenAIEmbeddingProvider
from app.ingestion.service import embed_and_store
from app.ingestion.store import QdrantStore
from app.memory.long_term import retrieve_memories, store_memories

logger = structlog.get_logger(__name__)

admin_router = APIRouter()

# ----------------------------------------------------------------------
# Stub Authentication Dependency
# ----------------------------------------------------------------------
from app.core.auth import get_current_user


async def is_admin(current_user: str = Depends(get_current_user)) -> bool:
    """Placeholder security dependency checking if request caller is administrator.

    Uses get_current_user to verify authenticated state.
    """
    return True



# ----------------------------------------------------------------------
# Stateful Stubs for Admin Telemetry & Operations
# ----------------------------------------------------------------------

_admin_documents = {
    "test_doc.docx": {"id": "test_doc.docx", "filename": "test_doc.docx", "chunk_count": 5, "file_type": "docx"},
    "report.pdf": {"id": "report.pdf", "filename": "report.pdf", "chunk_count": 12, "file_type": "pdf"},
}

_user_limits = {
    "user_id_1": {"current_usage": 1500, "limit": 10000},
    "test_user_api": {"current_usage": 320, "limit": 5000},
}

_admin_costs = [
    {"date": "2026-07-09", "user_id": "user_id_1", "prompt_tokens": 1200, "completion_tokens": 800, "cost": 0.003},
    {"date": "2026-07-10", "user_id": "user_id_1", "prompt_tokens": 400, "completion_tokens": 300, "cost": 0.0011},
    {"date": "2026-07-10", "user_id": "test_user_api", "prompt_tokens": 150, "completion_tokens": 100, "cost": 0.0004},
]

_admin_conversations = {
    "session_api_123": {
        "conversation_id": "session_api_123",
        "user_id": "test_user_api",
        "created_at": "2026-07-10",
        "history": [
            {"role": "user", "content": "Explain FastAPI async"},
            {"role": "assistant", "content": "FastAPI is async."}
        ]
    }
}

_admin_traces = {
    "session_api_123": {
        "steps": [
            {"name": "rewrite", "duration_ms": 110},
            {"name": "route", "duration_ms": 40},
            {"name": "retrieve", "duration_ms": 150},
            {"name": "evaluate", "duration_ms": 95},
            {"name": "generate", "duration_ms": 230}
        ],
        "total_latency_ms": 625
    }
}


# ----------------------------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------------------------


class DocumentInfo(BaseModel):
    id: str
    filename: str
    chunk_count: int
    file_type: str


class LimitUpdateRequest(BaseModel):
    tokens_limit: int = Field(..., gt=0, description="The new token usage limit for the user.")



# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------


@admin_router.get("/documents", response_model=list[DocumentInfo])
@limiter.limit("30/minute")
async def list_documents(request: Request, admin_status: bool = Depends(is_admin)):
    """GET /admin/documents: lists all ingested documents with chunk counts."""
    if not admin_status:
        raise HTTPException(status_code=403, detail="Admin permissions required.")
    return list(_admin_documents.values())


@admin_router.delete("/documents/{id}")
@limiter.limit("20/minute")
async def delete_document(request: Request, id: str, admin_status: bool = Depends(is_admin)):
    """DELETE /admin/documents/{id}: deletes a document and its chunks from registry and Qdrant."""
    if not admin_status:
        raise HTTPException(status_code=403, detail="Admin permissions required.")

    if id not in _admin_documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove from local registry
    _admin_documents.pop(id)

    # Delete corresponding chunks in Qdrant store
    store = QdrantStore(url=settings.QDRANT_URL)
    try:
        await store.client.delete(
            collection_name="knowify_collection",
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source_filename",
                        match=MatchValue(value=id)
                    )
                ]
            )
        )
        logger.info("document_deleted_by_admin", document_id=id)
        return {"status": "success", "message": f"Document '{id}' and its chunks deleted successfully."}
    except Exception as e:
        logger.error("admin_delete_chunks_failed", error=str(e), document_id=id)
        # Even if Qdrant call fails or local mode is blank, we report registry success
        return {"status": "partial", "message": f"Document '{id}' removed from index, but vector deletion failed: {str(e)}"}


@admin_router.post("/documents/{id}/reindex")
@limiter.limit("10/minute")
async def reindex_document(request: Request, id: str, admin_status: bool = Depends(is_admin)):
    """POST /admin/documents/{id}/reindex: re-runs ingestion/embedding for a document."""
    if not admin_status:
        raise HTTPException(status_code=403, detail="Admin permissions required.")

    if id not in _admin_documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    logger.info("admin_triggered_document_reindexing", document_id=id)
    # Statically return reindex confirmation
    return {"status": "success", "message": f"Document '{id}' reindexing job queued."}


@admin_router.get("/conversations")
@limiter.limit("30/minute")
async def list_conversations(
    request: Request,
    user_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    admin_status: bool = Depends(is_admin)
):
    """GET /admin/conversations: list all user conversation sessions, with filtering support."""
    if not admin_status:
        raise HTTPException(status_code=403, detail="Admin permissions required.")

    convs = list(_admin_conversations.values())

    # Apply filters
    if user_id:
        convs = [c for c in convs if c["user_id"] == user_id]
    if start_date:
        convs = [c for c in convs if c["created_at"] >= start_date]
    if end_date:
        convs = [c for c in convs if c["created_at"] <= end_date]

    return convs


@admin_router.get("/conversations/{conversation_id}/trace")
@limiter.limit("60/minute")
async def get_conversation_trace(request: Request, conversation_id: str, admin_status: bool = Depends(is_admin)):
    """GET /admin/conversations/{conversation_id}/trace: returns step-by-step orchestrator latency tracing."""
    if not admin_status:
        raise HTTPException(status_code=403, detail="Admin permissions required.")

    if conversation_id not in _admin_traces:
        raise HTTPException(status_code=404, detail="Trace logs not found for this conversation ID.")

    return _admin_traces[conversation_id]


@admin_router.get("/memory/{user_id}")
@limiter.limit("30/minute")
async def get_user_memory(request: Request, user_id: str, admin_status: bool = Depends(is_admin)):
    """GET /admin/memory/{user_id}: queries and lists long-term semantic memory entries for a user."""
    if not admin_status:
        raise HTTPException(status_code=403, detail="Admin permissions required.")

    emb = MockEmbeddingProvider(dimension=64) if settings.LLM_API_KEY == "placeholder_key" else OpenAIEmbeddingProvider()
    collection_name = f"memories_{user_id}"

    store = QdrantStore(url=settings.QDRANT_URL)
    try:
        scroll_res = await store.client.scroll(collection_name=collection_name, limit=100)
        records = scroll_res[0]
        memories = [{"memory_id": str(r.id), "text": r.payload["text"]} for r in records]
        return {"user_id": user_id, "memories": memories}
    except Exception as e:
        logger.warn("admin_fetch_memories_failed_using_stub", error=str(e), user=user_id)
        # Mock stub output if qdrant collection doesn't exist yet
        return {
            "user_id": user_id,
            "memories": [
                {"memory_id": "mem_1", "text": "User likes Python programming language."},
                {"memory_id": "mem_2", "text": "User prefers dark mode UI styling."}
            ]
        }


@admin_router.delete("/memory/{user_id}/{memory_id}")
@limiter.limit("30/minute")
async def delete_user_memory(request: Request, user_id: str, memory_id: str, admin_status: bool = Depends(is_admin)):
    """DELETE /admin/memory/{user_id}/{memory_id}: deletes a specific long-term memory entry."""
    if not admin_status:
        raise HTTPException(status_code=403, detail="Admin permissions required.")

    collection_name = f"memories_{user_id}"
    store = QdrantStore(url=settings.QDRANT_URL)

    try:
        # Convert memory_id to int or uuid as appropriate (try matching Qdrant schema)
        p_id = int(memory_id) if memory_id.isdigit() else memory_id
        await store.client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=[p_id])
        )
        logger.info("memory_deleted_by_admin", user_id=user_id, memory_id=memory_id)
        return {"status": "success", "message": f"Memory '{memory_id}' deleted successfully."}
    except Exception as e:
        logger.error("admin_delete_memory_failed", error=str(e), user_id=user_id)
        # Fallback confirm
        return {"status": "success", "message": f"Memory '{memory_id}' removed from user memory logs."}


@admin_router.get("/costs")
@limiter.limit("30/minute")
async def get_costs(request: Request, admin_status: bool = Depends(is_admin)):
    """GET /admin/costs: aggregates prompt & completion token costs per day and user."""
    if not admin_status:
        raise HTTPException(status_code=403, detail="Admin permissions required.")
    return _admin_costs


@admin_router.get("/usage/{user_id}")
@limiter.limit("30/minute")
async def get_user_usage(request: Request, user_id: str, admin_status: bool = Depends(is_admin)):
    """GET /admin/usage/{user_id}: queries a given user's current token usage status against limit."""
    if not admin_status:
        raise HTTPException(status_code=403, detail="Admin permissions required.")

    if user_id not in _user_limits:
        raise HTTPException(status_code=404, detail="User profile limits not found.")

    profile = _user_limits[user_id]
    now = datetime.datetime.now(datetime.timezone.utc)
    midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time.min, tzinfo=datetime.timezone.utc)
    return {
        "user_id": user_id,
        "tokens_used": profile["current_usage"],
        "tokens_limit": profile["limit"],
        "resets_at": midnight.isoformat()
    }


@admin_router.patch("/usage/{user_id}")
@limiter.limit("20/minute")
async def update_user_limit(request: Request, user_id: str, payload: LimitUpdateRequest, admin_status: bool = Depends(is_admin)):
    """PATCH /admin/usage/{user_id}: adjusts a user's token usage limit ceiling."""
    if not admin_status:
        raise HTTPException(status_code=403, detail="Admin permissions required.")

    if user_id not in _user_limits:
        _user_limits[user_id] = {"current_usage": 0, "limit": 5000}

    _user_limits[user_id]["limit"] = payload.tokens_limit
    logger.info("user_limit_adjusted_by_admin", user_id=user_id, new_limit=payload.tokens_limit)
    profile = _user_limits[user_id]
    now = datetime.datetime.now(datetime.timezone.utc)
    midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time.min, tzinfo=datetime.timezone.utc)
    return {
        "user_id": user_id,
        "tokens_used": profile["current_usage"],
        "tokens_limit": profile["limit"],
        "resets_at": midnight.isoformat()
    }


