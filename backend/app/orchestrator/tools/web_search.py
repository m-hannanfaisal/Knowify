import time
from typing import Optional
import httpx
import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.core.config import settings

logger = structlog.get_logger(__name__)


class SearchResult(BaseModel):
    """Data model representing a web search result item."""

    title: str
    url: str
    snippet: str


async def web_search(query: str, api_key: Optional[str] = None) -> list[SearchResult]:
    """Queries the Tavily Search API for the given query.

    Args:
        query (str): The search query.
        api_key (Optional[str]): Optional Tavily API key override.

    Returns:
        list[SearchResult]: List of search results.
    """
    start_time = time.perf_counter()
    api_key = api_key or settings.TAVILY_API_KEY

    # Rule-based offline mock search results for development and testing
    if api_key == "placeholder_key" or not api_key:
        logger.info("skip_web_search_api_call", query=query, reason="No valid Tavily API key")
        q = query.lower()
        if "weather" in q:
            results = [
                SearchResult(
                    title="Current Weather in Paris - BBC Weather",
                    url="https://www.bbc.com/weather/paris",
                    snippet="Current weather in Paris is 22 degrees Celsius and partly cloudy with light breeze.",
                )
            ]
        elif "sports" in q or "score" in q:
            results = [
                SearchResult(
                    title="Real Madrid vs Barcelona Live Scores - ESPN",
                    url="https://www.espn.com/soccer",
                    snippet="FT: Real Madrid 2 - 1 Barcelona. Goals by Vinicius Jr and Bellingham.",
                )
            ]
        else:
            results = [
                SearchResult(
                    title="General Search Result 1",
                    url="https://example.com/result1",
                    snippet="This is a general mock search snippet relevant to " + query,
                )
            ]
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info("web_search_complete", query=query, num_results=len(results), latency_ms=latency_ms)
        return results

    # Live Tavily HTTP post query
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 5,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            raw_results = data.get("results", [])

            results = []
            for item in raw_results:
                results.append(
                    SearchResult(
                        title=item.get("title", "No Title"),
                        url=item.get("url", ""),
                        snippet=item.get("content", ""),
                    )
                )

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info("web_search_complete", query=query, num_results=len(results), latency_ms=latency_ms)
            return results
    except Exception as e:
        logger.error("web_search_api_failed", query=query, error=str(e))
        return []


async def summarize_web_results(
    query: str, results: list[SearchResult], api_key: Optional[str] = None
) -> str:
    """Synthesizes search results into a grounded, markdown cited answer.

    Args:
        query (str): The search query context.
        results (list[SearchResult]): The retrieved search result items.
        api_key (Optional[str]): Optional OpenAI API key override.

    Returns:
        str: Grounded cited summary response.
    """
    start_time = time.perf_counter()
    if not results:
        return "I'm sorry, I could not retrieve any web search results for your query."

    openai_api_key = api_key or settings.LLM_API_KEY
    if openai_api_key == "placeholder_key" or not openai_api_key:
        logger.info("skip_web_summary_llm_call", query=query, reason="No valid OpenAI API key")
        # Offline mock synthesis
        mock_summary = f"Mock Web Summary for query '{query}':\n"
        for i, res in enumerate(results):
            mock_summary += f"- According to [{res.title}]({res.url}), {res.snippet}\n"
        return mock_summary.strip()

    client = AsyncOpenAI(api_key=openai_api_key)

    search_results_text = "\n\n".join(
        [
            f"Source [{i+1}] (Title: {res.title}, URL: {res.url}):\n{res.snippet}"
            for i, res in enumerate(results)
        ]
    )

    system_prompt = (
        "You are an expert information synthesizer.\n"
        "Formulate a detailed, grounded, and accurate answer to the user's query using only the search results provided.\n"
        "For every claim you make, cite the corresponding source URL using markdown link format (e.g. '[Source Title](url)').\n"
        "If the search results do not contain enough information to answer the query, state that you cannot answer "
        "based on current search results. Do not make up any facts outside the context."
    )

    user_content = (
        f"User Query: {query}\n\n"
        f"Search Results:\n{search_results_text}\n\n"
        "Grounded cited answer:"
    )

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
        summary = (response.choices[0].message.content or "").strip()
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info("summarize_web_results_complete", query=query, latency_ms=latency_ms)
        return summary
    except Exception as e:
        logger.error("summarize_web_results_failed", query=query, error=str(e))
        return "Failed to synthesize web search results due to an internal system error."
