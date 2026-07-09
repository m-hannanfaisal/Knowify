import json
import structlog
from openai import AsyncOpenAI

from app.core.config import settings
from app.retrieval.models import RetrievedChunk

logger = structlog.get_logger(__name__)


async def evaluate_relevance(
    original_query: str,
    rewritten_query: str,
    retrieved_chunks: list[RetrievedChunk],
    api_key: str | None = None,
) -> dict:
    """Evaluates whether the retrieved chunks are relevant and sufficient to answer the query.

    Args:
        original_query (str): The user's original query.
        rewritten_query (str): The search query used for retrieval.
        retrieved_chunks (list[RetrievedChunk]): Chunks retrieved from Qdrant.
        api_key (str | None): Optional OpenAI API key override.

    Returns:
        dict: {
            "sufficient": bool,
            "reasoning": str,
            "feedback_for_rewrite": str
        }
    """
    api_key = api_key or settings.LLM_API_KEY
    if api_key == "placeholder_key" or not api_key:
        logger.info("skip_relevance_evaluation_llm_call", reason="No valid LLM API key provided")
        
        # Rule-based mock evaluator for testing retry branches offline
        q = original_query.lower()
        if "force_insufficient" in q:
            return {
                "sufficient": False,
                "reasoning": "Mock: Forced insufficiency for testing.",
                "feedback_for_rewrite": "needs X",
            }
        elif "sufficient_on_retry" in q:
            # If the rewritten query contains the feedback "needs X", it means we've retried
            if "needs x" in rewritten_query.lower():
                return {
                    "sufficient": True,
                    "reasoning": "Mock: Sufficient now after query adjustment.",
                    "feedback_for_rewrite": "",
                }
            else:
                return {
                    "sufficient": False,
                    "reasoning": "Mock: Insufficient on first attempt.",
                    "feedback_for_rewrite": "needs X",
                }
        elif "always_insufficient" in q:
            return {
                "sufficient": False,
                "reasoning": "Mock: Always insufficient for testing fallback.",
                "feedback_for_rewrite": "needs Y",
            }
        
        # Default mock output
        return {
            "sufficient": len(retrieved_chunks) > 0,
            "reasoning": "Mock: Chunks retrieved are sufficient.",
            "feedback_for_rewrite": "" if len(retrieved_chunks) > 0 else "needs search term adjustment",
        }

    client = AsyncOpenAI(api_key=api_key)

    chunks_text = "\n\n".join(
        [
            f"Chunk {i+1} (Source: {c.source_filename}, Page: {c.page_number}):\n{c.text}"
            for i, c in enumerate(retrieved_chunks)
        ]
    )

    system_prompt = (
        "You are an expert relevance evaluator in a RAG chatbot pipeline.\n"
        "Your task is to judge whether the retrieved document chunks contain sufficient and relevant information "
        "to completely and accurately answer the user's original query and the search-optimized query.\n"
        "You must respond with a JSON object containing exactly these three keys:\n"
        "- 'sufficient': a boolean (true or false) indicating if the retrieved information is enough to answer the query.\n"
        "- 'reasoning': a brief string explaining why the information is or is not sufficient.\n"
        "- 'feedback_for_rewrite': if 'sufficient' is false, a string with actionable feedback describing what information "
        "is missing and how to rewrite the search query to find it. If 'sufficient' is true, this must be an empty string.\n"
        "Do not write any markdown styling or extra text. Output only the JSON object."
    )

    user_content = (
        f"Original User Query: {original_query}\n"
        f"Search-Optimized Query: {rewritten_query}\n\n"
        f"Retrieved Document Chunks:\n{chunks_text}\n\n"
        "Evaluation JSON:"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=250,
        )
        result_str = response.choices[0].message.content or "{}"
        result = json.loads(result_str)
        
        # Clean/Format types
        return {
            "sufficient": bool(result.get("sufficient", False)),
            "reasoning": str(result.get("reasoning", "")),
            "feedback_for_rewrite": str(result.get("feedback_for_rewrite", "")),
        }
    except Exception as e:
        logger.error("relevance_evaluator_failed", error=str(e))
        return {
            "sufficient": len(retrieved_chunks) > 0,
            "reasoning": f"Fallback due to evaluator error: {e}",
            "feedback_for_rewrite": "",
        }
