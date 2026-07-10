import pytest
from unittest.mock import AsyncMock, patch
import httpx

from app.orchestrator.tools.web_search import web_search, summarize_web_results, SearchResult


@pytest.mark.asyncio
async def test_web_search_offline() -> None:
    """Test web_search rules-based fallbacks when TAVILY_API_KEY is placeholder."""
    # 1. Weather query
    weather_results = await web_search("Paris weather", api_key="placeholder_key")
    assert len(weather_results) == 1
    assert "bbc" in weather_results[0].url
    assert "Celsius" in weather_results[0].snippet

    # 2. Sports query
    sports_results = await web_search("Barcelona score", api_key="placeholder_key")
    assert len(sports_results) == 1
    assert "espn" in sports_results[0].url
    assert "Real Madrid" in sports_results[0].snippet

    # 3. General query
    general_results = await web_search("FastAPI framework", api_key="placeholder_key")
    assert len(general_results) == 1
    assert "example.com" in general_results[0].url
    assert "FastAPI" in general_results[0].snippet


@pytest.mark.asyncio
async def test_web_search_mock_api() -> None:
    """Test web_search Tavily HTTP API parsing by mocking httpx AsyncClient."""
    mock_tavily_response = {
        "results": [
            {
                "title": "Tavily Title 1",
                "url": "https://tavily.com/1",
                "content": "This is content retrieved from Tavily search endpoint."
            }
        ]
    }

    # Patch httpx.AsyncClient.post
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # Construct mock response object
        from unittest.mock import MagicMock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_tavily_response
        mock_post.return_value = mock_response


        results = await web_search("search query", api_key="valid_tavily_key")
        assert len(results) == 1
        assert results[0].title == "Tavily Title 1"
        assert results[0].url == "https://tavily.com/1"
        assert "retrieved" in results[0].snippet


@pytest.mark.asyncio
async def test_summarize_web_results_offline() -> None:
    """Test summarization falls back to mock cited list when LLM API key is placeholder."""
    results = [
        SearchResult(title="Docs", url="https://docs.org", snippet="FastAPI is async.")
    ]
    summary = await summarize_web_results("query", results, api_key="placeholder_key")
    assert "Mock Web Summary" in summary
    assert "FastAPI is async" in summary
    assert "https://docs.org" in summary


@pytest.mark.asyncio
async def test_summarize_web_results_mock_llm() -> None:
    """Test web summarization calling OpenAI completions with grounded citations."""
    results = [
        SearchResult(title="Docs", url="https://docs.org", snippet="FastAPI is async.")
    ]

    with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value.choices = [
            AsyncMock(message=AsyncMock(content="According to [Docs](https://docs.org), FastAPI is async."))
        ]

        summary = await summarize_web_results("query", results, api_key="valid_openai_key")
        assert "According to [Docs]" in summary
        assert "https://docs.org" in summary
