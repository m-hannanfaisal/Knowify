import asyncio
from typing import Optional
import structlog
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

from app.core.config import settings
from app.ingestion.embeddings import MockEmbeddingProvider, OpenAIEmbeddingProvider
from app.retrieval.reranker import MockReranker, CohereReranker
from app.retrieval.service import retrieve

logger = structlog.get_logger(__name__)

# Initialize FastMCP Server
mcp_server = FastMCP("Knowify RAG Server")


async def summarize_rag_results(query: str, chunks: list, api_key: Optional[str] = None) -> str:
    """Synthesizes RAG retrieved chunks into a cited grounded markdown answer.

    Args:
        query (str): The search query.
        chunks (list[RetrievedChunk]): Chunks retrieved from database.
        api_key (Optional[str]): OpenAI API key override.

    Returns:
        str: Cited answer response.
    """
    if not chunks:
        return "I'm sorry, I cannot find enough relevant information in the knowledge base."

    api_key = api_key or settings.LLM_API_KEY
    if api_key == "placeholder_key" or not api_key:
        logger.info("skip_rag_summary_llm_call", query=query)
        # Mock summary for offline testing
        summary = f"Mock RAG Answer for '{query}':\n"
        for c in chunks:
            summary += f"- Found fact: {c.text} (Source: {c.source_filename})\n"
        return summary.strip()

    client = AsyncOpenAI(api_key=api_key)

    chunks_text = "\n\n".join(
        [
            f"Source [{i+1}] (File: {c.source_filename}, Page: {c.page_number}):\n{c.text}"
            for i, c in enumerate(chunks)
        ]
    )

    system_prompt = (
        "You are an expert technical assistant.\n"
        "Formulate a detailed, grounded, and accurate answer to the user's query using only the document chunks provided.\n"
        "For every claim you make, cite the corresponding file source in parentheses (e.g. '(File: source.txt)').\n"
        "If the document chunks do not contain enough information to answer the query, state that you do not "
        "have enough information to answer."
    )

    user_content = f"User Query: {query}\n\nDocument Chunks:\n{chunks_text}\n\nAnswer:"

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=350,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error("summarize_rag_results_failed", error=str(e))
        return "Failed to synthesize a grounded answer due to internal LLM completion failure."


@mcp_server.tool()
async def query_knowledge_base(query: str, collection_name: Optional[str] = None) -> str:
    """Queries the Knowify RAG knowledge base and returns a cited grounded answer.

    Args:
        query (str): The search query.
        collection_name (Optional[str]): Custom collection name target. Defaults to 'knowify_collection'.
    """
    collection = collection_name or "knowify_collection"

    # Select proper providers based on local vs production API key status
    if settings.LLM_API_KEY == "placeholder_key":
        emb = MockEmbeddingProvider(dimension=64)
        reranker = MockReranker()
    else:
        emb = OpenAIEmbeddingProvider()
        reranker = CohereReranker()

    logger.info("mcp_server_query_knowledge_base_called", query=query, collection=collection)

    # 1. Retrieval
    chunks = await retrieve(
        query=query,
        collection_name=collection,
        embedding_provider=emb,
        reranker=reranker,
        qdrant_url=settings.QDRANT_URL,
    )

    # 2. Synthesis Generation
    answer = await summarize_rag_results(query=query, chunks=chunks)
    return answer


if __name__ == "__main__":
    # Launch stdio server loop
    mcp_server.run()
