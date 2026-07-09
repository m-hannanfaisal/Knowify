import time
import structlog

from app.evaluation.relevance_evaluator import evaluate_relevance
from app.ingestion.embeddings import BaseEmbeddingProvider
from app.orchestrator.rewriter import query_rewriter
from app.orchestrator.router import route_query
from app.retrieval.reranker import BaseReranker
from app.retrieval.service import retrieve

logger = structlog.get_logger(__name__)


async def handle_query(
    query: str,
    conversation_history: list[dict[str, str]],
    long_term_memory: list[str],
    collection_name: str,
    embedding_provider: BaseEmbeddingProvider,
    reranker: BaseReranker,
    api_key: str | None = None,
    qdrant_url: str | None = None,
) -> dict:
    """Orchestrates query rewriting, routing, RAG retrieval, and corrective retry loop evaluation.

    Args:
        query (str): The user's original input query.
        conversation_history (list[dict[str, str]]): List of previous turns.
        long_term_memory (list[str]): Relevant context facts retrieved from long-term memory.
        collection_name (str): Destination collection name in Qdrant.
        embedding_provider (BaseEmbeddingProvider): Embedding generator service.
        reranker (BaseReranker): Reranker cross-encoder service.
        api_key (str | None): Optional OpenAI API key override.
        qdrant_url (str | None): Optional Qdrant connection URL.

    Returns:
        dict: The final query processing state containing rewritten query, routing, chunks,
              sufficiency flags, and safe fallbacks if required.
    """
    total_start = time.perf_counter()

    # 1. Rewrite Query (first attempt)
    rewrite_start = time.perf_counter()
    rewritten = await query_rewriter(
        query=query,
        conversation_history=conversation_history,
        long_term_memory=long_term_memory,
        api_key=api_key,
    )
    rewrite_latency = int((time.perf_counter() - rewrite_start) * 1000)

    # 2. Route Query
    route_start = time.perf_counter()
    route = await route_query(rewritten_query=rewritten, api_key=api_key)
    route_latency = int((time.perf_counter() - route_start) * 1000)

    current_query = rewritten

    # 3. Route Execution & Corrective RAG Retry Loop
    retrieved_chunks = []
    insufficient_information = False
    fallback_response = None

    if route == "rag":
        max_retries = 2
        retries_remaining = max_retries
        feedback = None

        while True:
            # If re-indexing in the retry loop, rewrite query with feedback
            if feedback:
                rewrite_start = time.perf_counter()
                current_query = await query_rewriter(
                    query=query,
                    conversation_history=conversation_history,
                    long_term_memory=long_term_memory,
                    feedback=feedback,
                    api_key=api_key,
                )
                rewrite_latency = int((time.perf_counter() - rewrite_start) * 1000)
                logger.info("retry_query_rewritten", query=current_query, feedback=feedback)

            # Retrieve from Module 3
            retrieved_chunks = await retrieve(
                query=current_query,
                collection_name=collection_name,
                embedding_provider=embedding_provider,
                reranker=reranker,
                qdrant_url=qdrant_url,
            )

            # Evaluate relevance and sufficiency (Module 5)
            eval_start = time.perf_counter()
            eval_res = await evaluate_relevance(
                original_query=query,
                rewritten_query=current_query,
                retrieved_chunks=retrieved_chunks,
                api_key=api_key,
            )
            eval_latency = int((time.perf_counter() - eval_start) * 1000)

            retry_count = max_retries - retries_remaining
            logger.info(
                "relevance_evaluated",
                retry_count=retry_count,
                sufficient=eval_res["sufficient"],
                reasoning=eval_res["reasoning"],
                feedback_for_rewrite=eval_res["feedback_for_rewrite"],
                eval_latency_ms=eval_latency,
            )

            if eval_res["sufficient"]:
                # Satisfied!
                break
            else:
                if retries_remaining > 0:
                    retries_remaining -= 1
                    feedback = eval_res["feedback_for_rewrite"]
                    logger.info(
                        "retry_loop_triggered",
                        retries_remaining=retries_remaining,
                        feedback=feedback,
                    )
                else:
                    logger.warn(
                        "retries_exhausted_fallback",
                        original_query=query,
                        final_query=current_query,
                    )
                    insufficient_information = True
                    fallback_response = (
                        "I'm sorry, I cannot find enough relevant information in the "
                        "documents to answer your question."
                    )
                    retrieved_chunks = []
                    break
    elif route == "web_search":
        # Live web search placeholder (Module 7)
        pass
    else:  # direct
        pass

    total_latency = int((time.perf_counter() - total_start) * 1000)

    logger.info(
        "query_orchestration_complete",
        original_query=query,
        rewritten_query=current_query,
        routing_decision=route,
        num_retrieved_chunks=len(retrieved_chunks),
        insufficient_information=insufficient_information,
        rewrite_latency_ms=rewrite_latency,
        route_latency_ms=route_latency,
        total_latency_ms=total_latency,
    )

    return {
        "rewritten_query": current_query,
        "route": route,
        "retrieved_chunks": retrieved_chunks,
        "insufficient_information": insufficient_information,
        "fallback_response": fallback_response,
    }
