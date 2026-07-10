import asyncio
import re
import time
from typing import AsyncGenerator, Optional
import structlog
from openai import AsyncOpenAI
from langsmith import traceable

from app.core.config import settings
from app.retrieval.models import RetrievedChunk

logger = structlog.get_logger(__name__)


@traceable(run_type="llm", name="Knowify Grounded Generation")
async def generate_response(

    query: str,
    rewritten_query: str,
    retrieved_chunks: list[RetrievedChunk],
    conversation_history: list[dict[str, str]],
    long_term_memory: list[str],
    api_key: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Generates an answer grounded in the retrieved chunks, streaming tokens and structured citations.

    Cites sources inline using `[source: filename, page/chunk]` format and parses citations into metadata.

    Args:
        query (str): Original user query.
        rewritten_query (str): Rewritten query.
        retrieved_chunks (list[RetrievedChunk]): Retrieved reference document chunks.
        conversation_history (list[dict[str, str]]): List of previous chat turns.
        long_term_memory (list[str]): Relevant user memories.
        api_key (Optional[str]): OpenAI API key override.

    Yields:
        dict: Typed packets (token, citations, metrics) for SSE rendering.
    """
    start_time = time.perf_counter()
    api_key = api_key or settings.LLM_API_KEY

    # 1. Prepare Grounded Retrieval Prompt Chunks Context
    chunks_prompt_text = ""
    for chunk in retrieved_chunks:
        location = (
            f"page {chunk.page_number}"
            if chunk.page_number is not None
            else f"chunk {chunk.chunk_index}"
        )
        chunks_prompt_text += (
            f"Source reference name: {chunk.source_filename} ({location})\n"
            f"Content:\n{chunk.text}\n\n"
        )

    system_prompt = (
        "You are a helpful, expert AI assistant.\n"
        "Answer the user query completely and accurately using only the retrieved document chunks provided below.\n"
        "State facts present ONLY in the retrieved chunks. Do not make any unsupported claims or assumptions.\n"
        "For every claim you make, cite the source inline at the end of the sentence or clause using EXACTLY "
        "the format '[source: filename, page/chunk]'. Replace 'filename' and 'page/chunk' with the exact filename "
        "and page/chunk reference names from the sources provided (e.g. '[source: doc1.txt, chunk 0]')."
    )

    # 2. Format Context Elements: Memory & Chat History
    memory_text = "\n".join([f"- {m}" for m in long_term_memory])
    history_messages = []
    for turn in conversation_history:
        history_messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})

    user_content = (
        f"User long term memory profile context:\n{memory_text}\n\n"
        f"Retrieved Document Chunks:\n{chunks_prompt_text}\n"
        f"User Query: {query}\n"
        f"Rewritten Retrieval Query: {rewritten_query}\n\n"
        "Answer:"
    )

    # 3. Rules-based Mock Streaming for Offline/Dev Environments
    if api_key == "placeholder_key" or not api_key:
        logger.info("skip_openai_streaming_call_using_mock", query=query)
        mock_response_text = (
            f"This is a grounded answer to your query '{query}'. "
            f"Based on the provided facts, the document mentions FastAPI "
            f"components [source: doc1.txt, chunk 0]. Additionally, other files detail "
            f"advanced configuration variables [source: doc2.pdf, page 5]."
        )

        accumulated_text = ""
        # Stream word by word
        words = mock_response_text.split(" ")
        for i, word in enumerate(words):
            word_part = word + (" " if i < len(words) - 1 else "")
            accumulated_text += word_part
            yield {"type": "token", "text": word_part}
            await asyncio.sleep(0.04)

        # Parse inline citations
        matches = re.findall(r"\[source:\s*([^,\]]+),\s*([^\]]+)\]", accumulated_text)
        unique_citations = []
        seen = set()
        for filename, location in matches:
            filename = filename.strip()
            location = location.strip()
            if (filename, location) not in seen:
                seen.add((filename, location))
                # Try to find match in retrieved chunks for full metadata if possible
                matched_chunk = next(
                    (
                        c
                        for c in retrieved_chunks
                        if c.source_filename == filename
                        and (
                            (c.page_number is not None and f"page {c.page_number}" == location)
                            or (c.page_number is None and f"chunk {c.chunk_index}" == location)
                        )
                    ),
                    None,
                )
                unique_citations.append(
                    {
                        "filename": filename,
                        "location": location,
                        "page_number": matched_chunk.page_number if matched_chunk else None,
                        "chunk_index": matched_chunk.chunk_index if matched_chunk else None,
                    }
                )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "mock_generation_complete",
            chunks_count=len(retrieved_chunks),
            citation_count=len(unique_citations),
            latency_ms=latency_ms,
        )

        yield {
            "type": "citations",
            "citations": unique_citations,
        }
        yield {
            "type": "metrics",
            "latency_ms": latency_ms,
            "token_counts": {"prompt_tokens": 150, "completion_tokens": len(words), "total_tokens": 150 + len(words)},
        }
        return

    # 4. Live Stream calling OpenAI Chat Completions API
    client = AsyncOpenAI(api_key=api_key)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_content})

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            stream=True,
        )

        accumulated_text = ""
        completion_tokens = 0

        async for chunk in response:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                accumulated_text += delta
                completion_tokens += 1
                yield {"type": "token", "text": delta}

        # Parse inline citations from generated text
        matches = re.findall(r"\[source:\s*([^,\]]+),\s*([^\]]+)\]", accumulated_text)
        unique_citations = []
        seen = set()
        for filename, location in matches:
            filename = filename.strip()
            location = location.strip()
            if (filename, location) not in seen:
                seen.add((filename, location))
                # Search original retrieved_chunks list for matches to enrich payload metadata
                matched_chunk = next(
                    (
                        c
                        for c in retrieved_chunks
                        if c.source_filename == filename
                        and (
                            (c.page_number is not None and f"page {c.page_number}" == location)
                            or (c.page_number is None and f"chunk {c.chunk_index}" == location)
                        )
                    ),
                    None,
                )
                unique_citations.append(
                    {
                        "filename": filename,
                        "location": location,
                        "page_number": matched_chunk.page_number if matched_chunk else None,
                        "chunk_index": matched_chunk.chunk_index if matched_chunk else None,
                    }
                )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "generation_complete",
            chunks_count=len(retrieved_chunks),
            citation_count=len(unique_citations),
            latency_ms=latency_ms,
        )

        yield {
            "type": "citations",
            "citations": unique_citations,
        }
        yield {
            "type": "metrics",
            "latency_ms": latency_ms,
            "token_counts": {
                "prompt_tokens": len(messages) * 100,  # Estimated
                "completion_tokens": completion_tokens,
                "total_tokens": (len(messages) * 100) + completion_tokens,
            },
        }

    except Exception as e:
        logger.error("generation_api_failed", error=str(e))
        yield {"type": "token", "text": "Generation failed due to internal connection failure."}
