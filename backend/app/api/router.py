import json
import os
import tempfile
import datetime

from typing import Optional
import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.limiter import limiter
from app.ingestion.embeddings import MockEmbeddingProvider, OpenAIEmbeddingProvider
from app.ingestion.image_handler import ingest_image
from app.ingestion.service import embed_and_store
from app.memory.long_term import retrieve_memories
from app.orchestrator.generation import generate_response
from app.orchestrator.service import handle_query
from app.retrieval.reranker import CohereReranker, MockReranker

logger = structlog.get_logger(__name__)

router = APIRouter()

# Global in-memory conversation history dictionary store
_conversation_history: dict[str, list[dict[str, str]]] = {}


class ChatRequest(BaseModel):
    """Request validation schema for chat endpoint queries."""

    query: str = Field(..., min_length=1, description="The user's query.")
    user_id: str = Field(..., min_length=1, description="Associated user ID namespaces.")
    conversation_id: str = Field(..., min_length=1, description="Unique session ID.")


@router.post("/chat")
@limiter.limit("60/minute")
async def chat(request: Request, payload: ChatRequest):
    """Executes memory retrieval, routes query via StateGraph, and yields SSE stream response."""
    # 0. Enforce token usage limit check (returns 429 if exceeded)
    from app.api.admin_router import _user_limits
    profile = _user_limits.setdefault(payload.user_id, {"current_usage": 0, "limit": 5000})
    if profile["current_usage"] >= profile["limit"]:
        raise HTTPException(
            status_code=429,
            detail="Token limit reached."
        )

    if settings.LLM_API_KEY == "placeholder_key":
        emb = MockEmbeddingProvider(dimension=64)
        reranker = MockReranker()
    else:
        emb = OpenAIEmbeddingProvider()
        reranker = CohereReranker()

    # 1. Retrieve memories
    try:
        memories = await retrieve_memories(
            user_id=payload.user_id,
            query=payload.query,
            embedding_provider=emb,
            top_k=3,
            qdrant_url=settings.QDRANT_URL,
        )
    except Exception as e:
        logger.error("api_retrieve_memories_failed", error=str(e), user=payload.user_id)
        memories = []

    # 2. Load conversation history
    history = _conversation_history.get(payload.conversation_id, [])

    # 3. Execute StateGraph pipeline
    try:
        state = await handle_query(
            query=payload.query,
            conversation_history=history,
            long_term_memory=memories,
            collection_name="knowify_collection",
            embedding_provider=emb,
            reranker=reranker,
            api_key=settings.LLM_API_KEY,
            qdrant_url=settings.QDRANT_URL,
        )
    except Exception as e:
        logger.error("orchestrator_execution_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal query orchestration error.")

    # 4. SSE async generator yielding JSON-encoded tokens + metadata
    async def sse_generator():
        accumulated_text = ""
        try:
            async for event in generate_response(
                query=payload.query,
                rewritten_query=state["rewritten_query"],
                retrieved_chunks=state["retrieved_chunks"],
                conversation_history=history,
                long_term_memory=memories,
                api_key=settings.LLM_API_KEY,
            ):
                if event["type"] == "token":
                    accumulated_text += event["text"]
                elif event["type"] == "metrics":
                    from app.api.admin_router import _user_limits
                    p = _user_limits.setdefault(payload.user_id, {"current_usage": 0, "limit": 5000})
                    p["current_usage"] += event["token_counts"]["total_tokens"]
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error("sse_generation_failed", error=str(e))
            yield f"data: {json.dumps({'type': 'token', 'text': 'Streaming error.'})}\n\n"

        # Preserve dialogue turns in-memory
        new_history = history.copy()
        new_history.append({"role": "user", "content": payload.query})
        new_history.append({"role": "assistant", "content": accumulated_text})
        _conversation_history[payload.conversation_id] = new_history

    return StreamingResponse(sse_generator(), media_type="text/event-stream")



@router.post("/upload")
@limiter.limit("20/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    collection_name: str = Form("knowify_collection"),
):
    """Multipart file upload endpoint routing to correct ingestion parser based on extension."""
    if settings.LLM_API_KEY == "placeholder_key":
        emb = MockEmbeddingProvider(dimension=64)
    else:
        emb = OpenAIEmbeddingProvider()

    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        # Write binary bytes to temporary local file path
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        filename = file.filename
        ext = os.path.splitext(filename)[1].lower().replace(".", "")

        if ext in ["jpg", "jpeg", "png"]:
            chunks_count = await ingest_image(
                file_path=temp_file_path,
                collection_name=collection_name,
                embedding_provider=emb,
                api_key=settings.LLM_API_KEY,
                qdrant_url=settings.QDRANT_URL,
            )
        else:
            chunks_count = await embed_and_store(
                file_path=temp_file_path,
                collection_name=collection_name,
                embedding_provider=emb,
                qdrant_url=settings.QDRANT_URL,
            )

        return {"filename": filename, "chunks_processed": chunks_count}

    except Exception as e:
        logger.error("upload_ingest_failed", error=str(e), filename=file.filename)
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.get("/conversations/{conversation_id}")
@limiter.limit("60/minute")
async def get_conversation(request: Request, conversation_id: str):
    """Returns the chat session history log for the given ID."""
    history = _conversation_history.get(conversation_id, [])
    return {"conversation_id": conversation_id, "history": history}


@router.get("/usage")
@limiter.limit("60/minute")
async def get_default_usage(request: Request, user_id: str = "default_user"):
    """Returns current token usage against limit for the default user."""
    from app.api.admin_router import _user_limits
    limits = _user_limits.get(user_id, {"current_usage": 0, "limit": 5000})
    now = datetime.datetime.now(datetime.timezone.utc)
    midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time.min, tzinfo=datetime.timezone.utc)
    return {
        "user_id": user_id,
        "tokens_used": limits.get("current_usage", 0),
        "tokens_limit": limits.get("limit", 5000),
        "resets_at": midnight.isoformat()
    }


@router.get("/usage/{user_id}")
@limiter.limit("60/minute")
async def get_user_usage(request: Request, user_id: str):
    """Returns user's current token usage against their limit."""
    from app.api.admin_router import _user_limits
    limits = _user_limits.get(user_id, {"current_usage": 0, "limit": 5000})
    now = datetime.datetime.now(datetime.timezone.utc)
    midnight = datetime.datetime.combine(now.date() + datetime.timedelta(days=1), datetime.time.min, tzinfo=datetime.timezone.utc)
    return {
        "user_id": user_id,
        "tokens_used": limits.get("current_usage", 0),
        "tokens_limit": limits.get("limit", 5000),
        "resets_at": midnight.isoformat()
    }



