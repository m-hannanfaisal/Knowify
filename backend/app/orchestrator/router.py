from typing import Literal
import structlog
from openai import AsyncOpenAI

from app.core.config import settings

logger = structlog.get_logger(__name__)


async def route_query(
    rewritten_query: str,
    api_key: str | None = None,
) -> Literal["rag", "direct", "web_search"]:
    """Classifies a query to determine if it requires RAG database lookup, direct completion, or live web search.

    Args:
        rewritten_query (str): Standalone context-resolved search query.
        api_key (str | None): Optional OpenAI API key override.

    Returns:
        Literal["rag", "direct", "web_search"]: The chosen query pipeline routing.
    """
    api_key = api_key or settings.LLM_API_KEY
    if api_key == "placeholder_key" or not api_key:
        logger.info("skip_query_routing_llm_call", reason="No valid LLM API key provided")
        # Deterministic rules-based router for testing
        q = rewritten_query.lower()
        if any(w in q for w in ["weather", "news", "scores", "current", "web"]):
            return "web_search"
        if any(w in q for w in ["docs", "documents", "documentation", "rag", "framework", "vector"]):
            return "rag"
        return "direct"

    client = AsyncOpenAI(api_key=api_key)

    system_prompt = (
        "You are an intelligent query router in a multi-step RAG chatbot pipeline.\n"
        "Analyze the user query and classify it into exactly one of three categories:\n"
        "1. 'rag': The query requires looking up specific documents, technical manuals, codebase docs, or context stored in the knowledge base.\n"
        "2. 'direct': General knowledge questions, chit-chat, greetings, generic coding snippets, or questions answerable without external database or internet queries.\n"
        "3. 'web_search': Queries about current events, real-time facts, current news, sports scores, or information requiring live internet access.\n"
        "Provide exactly one of the three options: 'rag', 'direct', or 'web_search'. Do not include any other words."
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rewritten_query},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        decision = (response.choices[0].message.content or "direct").strip().lower()
        if decision in ["rag", "direct", "web_search"]:
            return decision  # type: ignore

        # Flexible backup regex parsing
        if "rag" in decision:
            return "rag"
        elif "web" in decision:
            return "web_search"
        return "direct"
    except Exception as e:
        logger.error("query_router_failed", error=str(e))
        return "direct"
