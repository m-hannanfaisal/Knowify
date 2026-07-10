import time
from typing import Any, Literal, Optional, TypedDict
import structlog
from langgraph.graph import END, StateGraph
from langsmith import traceable

from app.core.metrics import metrics_tracker
from app.evaluation.relevance_evaluator import evaluate_relevance
from app.ingestion.embeddings import BaseEmbeddingProvider
from app.orchestrator.rewriter import query_rewriter
from app.orchestrator.router import route_query
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import BaseReranker
from app.retrieval.service import retrieve

logger = structlog.get_logger(__name__)


class AgentState(TypedDict):
    """The state representation object passed between LangGraph nodes."""

    # Input parameters and internal state
    query: str
    conversation_history: list[dict[str, str]]
    long_term_memory: list[str]
    rewritten_query: str
    route: Literal["rag", "direct", "web_search"]
    retrieved_chunks: list[RetrievedChunk]
    evaluator_result: Optional[dict]
    retry_count: int
    insufficient_information: bool
    fallback_response: Optional[str]

    # Injected request-scoped dependencies
    collection_name: str
    embedding_provider: BaseEmbeddingProvider
    reranker: BaseReranker
    api_key: Optional[str]
    qdrant_url: Optional[str]


# ----------------------------------------------------------------------
# Node Definitions (Declared globally to compile the graph once)
# ----------------------------------------------------------------------


@traceable(run_type="chain", name="Knowify Rewrite Node")
async def rewrite_step(state: AgentState) -> dict:
    """Query rewriter node: refines user query for retrieval context."""
    rewrite_start = time.perf_counter()
    feedback = None
    if state.get("evaluator_result") and not state["evaluator_result"].get("sufficient"):
        feedback = state["evaluator_result"].get("feedback_for_rewrite")

    current_query = await query_rewriter(
        query=state["query"],
        conversation_history=state["conversation_history"],
        long_term_memory=state["long_term_memory"],
        feedback=feedback,
        api_key=state.get("api_key"),
    )
    rewrite_latency = int((time.perf_counter() - rewrite_start) * 1000)

    if feedback:
        logger.info(
            "retry_query_rewritten",
            query=current_query,
            feedback=feedback,
            latency_ms=rewrite_latency,
        )
        metrics_tracker.record_retry()

    return {
        "rewritten_query": current_query,
        "retry_count": state["retry_count"] + 1 if feedback else state["retry_count"],
    }


@traceable(run_type="chain", name="Knowify Route Node")
async def route_step(state: AgentState) -> dict:
    """Router node: Classifies query targets to RAG, Direct, or Web Search."""
    route_start = time.perf_counter()
    route_decision = await route_query(
        rewritten_query=state["rewritten_query"], api_key=state.get("api_key")
    )
    route_latency = int((time.perf_counter() - route_start) * 1000)
    logger.info("query_routed", route=route_decision, latency_ms=route_latency)
    return {"route": route_decision}


@traceable(run_type="retriever", name="Knowify Retrieve Node")
async def retrieve_step(state: AgentState) -> dict:
    """Retrieval node: Queries documents from Qdrant vector database."""
    retrieve_start = time.perf_counter()
    chunks = await retrieve(
        query=state["rewritten_query"],
        collection_name=state["collection_name"],
        embedding_provider=state["embedding_provider"],
        reranker=state["reranker"],
        qdrant_url=state.get("qdrant_url"),
    )
    retrieve_latency = int((time.perf_counter() - retrieve_start) * 1000)
    logger.info("retrieval_step_complete", count=len(chunks), latency_ms=retrieve_latency)

    # Record retrieval hit rate metrics
    metrics_tracker.record_retrieval(1 if chunks else 0, 1)

    return {"retrieved_chunks": chunks}


@traceable(run_type="chain", name="Knowify Evaluate Node")
async def evaluate_step(state: AgentState) -> dict:
    """Evaluator node: Evaluates document sufficiency and relevance."""
    eval_start = time.perf_counter()
    eval_res = await evaluate_relevance(
        original_query=state["query"],
        rewritten_query=state["rewritten_query"],
        retrieved_chunks=state["retrieved_chunks"],
        api_key=state.get("api_key"),
    )
    eval_latency = int((time.perf_counter() - eval_start) * 1000)
    logger.info(
        "relevance_evaluated",
        retry_count=state["retry_count"],
        sufficient=eval_res["sufficient"],
        reasoning=eval_res["reasoning"],
        feedback_for_rewrite=eval_res["feedback_for_rewrite"],
        eval_latency_ms=eval_latency,
    )

    # Record faithfulness evaluation score
    metrics_tracker.record_faithfulness(1.0 if eval_res["sufficient"] else 0.0)

    return {"evaluator_result": eval_res}


@traceable(run_type="chain", name="Knowify Generate Node")
async def generate_step(state: AgentState) -> dict:
    """Generator placeholder node: Prepares final outputs and handles retry exhaustion fallbacks."""
    eval_res = state.get("evaluator_result")
    insufficient = False
    fallback = None
    chunks = state["retrieved_chunks"]

    if state["route"] == "rag" and eval_res and not eval_res.get("sufficient"):
        logger.warn(
            "retries_exhausted_fallback",
            original_query=state["query"],
            final_query=state["rewritten_query"],
        )
        insufficient = True
        fallback = (
            "I'm sorry, I cannot find enough relevant information in the "
            "documents to answer your question."
        )
        chunks = []  # Clear chunks on failure fallback

    return {
        "insufficient_information": insufficient,
        "fallback_response": fallback,
        "retrieved_chunks": chunks,
    }


# ----------------------------------------------------------------------
# Conditional Edge Functions
# ----------------------------------------------------------------------


def should_retrieve(state: AgentState) -> str:
    """Decides if the workflow should query document store or proceed to generation."""
    if state["route"] == "rag":
        return "retrieve"
    return "generate"


def should_retry(state: AgentState) -> str:
    """Conditional edge checking sufficiency and retry limits for feedback loop routing."""
    eval_res = state.get("evaluator_result")
    if eval_res and not eval_res.get("sufficient") and state["retry_count"] < 2:
        logger.info(
            "retry_loop_triggered",
            retry_count=state["retry_count"] + 1,
            feedback=eval_res.get("feedback_for_rewrite"),
        )
        return "rewrite"
    return "generate"


# ----------------------------------------------------------------------
# Compiled Graph Assembly (Constructed once at module load)
# ----------------------------------------------------------------------

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("rewrite", rewrite_step)
workflow.add_node("route", route_step)
workflow.add_node("retrieve", retrieve_step)
workflow.add_node("evaluate", evaluate_step)
workflow.add_node("generate", generate_step)

# Add Edges
workflow.set_entry_point("rewrite")
workflow.add_edge("rewrite", "route")

# Conditional routing edge
workflow.add_conditional_edges(
    "route", should_retrieve, {"retrieve": "retrieve", "generate": "generate"}
)

workflow.add_edge("retrieve", "evaluate")

# Conditional retry loop edge
workflow.add_conditional_edges(
    "evaluate", should_retry, {"rewrite": "rewrite", "generate": "generate"}
)

workflow.add_edge("generate", END)

# Compiled Graph reusable instance
compiled_graph = workflow.compile()


# ----------------------------------------------------------------------
# handle_query Orchestrator Entry Point
# ----------------------------------------------------------------------


@traceable(run_type="chain", name="Knowify Orchestration Pipeline")
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
    """Orchestrates query processing by executing the pre-compiled LangGraph StateGraph.

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
    metrics_tracker.record_request()
    total_start = time.perf_counter()

    # Build initial state with parameters and runtime dependencies injected
    initial_state = AgentState(
        query=query,
        conversation_history=conversation_history,
        long_term_memory=long_term_memory,
        rewritten_query="",
        route="direct",
        retrieved_chunks=[],
        evaluator_result=None,
        retry_count=0,
        insufficient_information=False,
        fallback_response=None,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        reranker=reranker,
        api_key=api_key,
        qdrant_url=qdrant_url,
    )

    # Run the pre-compiled graph asynchronously
    final_output_state = await compiled_graph.ainvoke(initial_state)

    total_latency_seconds = time.perf_counter() - total_start
    total_latency = int(total_latency_seconds * 1000)

    # Record request latency in tracker
    metrics_tracker.record_latency(total_latency_seconds)

    logger.info(
        "query_orchestration_complete",
        original_query=query,
        rewritten_query=final_output_state["rewritten_query"],
        routing_decision=final_output_state["route"],
        num_retrieved_chunks=len(final_output_state["retrieved_chunks"]),
        insufficient_information=final_output_state["insufficient_information"],
        total_latency_ms=total_latency,
    )

    return {
        "rewritten_query": final_output_state["rewritten_query"],
        "route": final_output_state["route"],
        "retrieved_chunks": final_output_state["retrieved_chunks"],
        "insufficient_information": final_output_state["insufficient_information"],
        "fallback_response": final_output_state["fallback_response"],
    }
