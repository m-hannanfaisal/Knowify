import structlog
from openai import AsyncOpenAI

from app.core.config import settings

logger = structlog.get_logger(__name__)


async def query_rewriter(
    query: str,
    conversation_history: list[dict[str, str]],
    long_term_memory: list[str],
    feedback: str | None = None,
    api_key: str | None = None,
) -> str:
    """Rewrites the user query using conversation history, long-term memory, and retry feedback.

    Args:
        query (str): The original user query.
        conversation_history (list[dict[str, str]]): List of previous messages in OpenAI format.
        long_term_memory (list[str]): Relevant context facts retrieved from long-term memory.
        feedback (str | None): Constructive feedback from a previous failed retrieval evaluation.
        api_key (str | None): Optional OpenAI API key override.

    Returns:
        str: Context-resolved search-optimized query string.
    """
    api_key = api_key or settings.LLM_API_KEY
    if api_key == "placeholder_key" or not api_key:
        logger.info("skip_query_rewrite_llm_call", reason="No valid LLM API key provided")
        if feedback:
            return f"Mock Rewritten with feedback: {query} (Feedback: {feedback})"
        return f"Mock Rewritten: {query}"

    client = AsyncOpenAI(api_key=api_key)

    history_str = ""
    for turn in conversation_history:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_str += f"{role}: {turn['content']}\n"

    memory_str = "\n".join([f"- {fact}" for fact in long_term_memory])

    system_prompt = (
        "You are an AI assistant that refines user queries for search engines in a RAG pipeline.\n"
        "Your task is to rewrite the user's latest query into a standalone, search-optimized query.\n"
        "Integrate relevant context from the conversation history and long-term memories provided.\n"
        "Resolve all pronouns (e.g. 'it', 'them', 'that') and referential phrases.\n"
        "Do not answer the query. Do not add any preamble or explanation. Return ONLY the rewritten query text."
    )

    feedback_str = f"Feedback from previous attempt: {feedback}\n" if feedback else ""

    user_content = (
        f"Conversation History:\n{history_str}\n"
        f"Long-term Memories:\n{memory_str}\n"
        f"{feedback_str}"
        f"Original User Query: {query}\n"
        f"Rewritten standalone query:"
    )


    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        rewritten = response.choices[0].message.content or query
        return rewritten.strip()
    except Exception as e:
        logger.error("query_rewriter_failed", error=str(e))
        return query
