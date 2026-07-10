import pytest
from unittest.mock import AsyncMock, patch

from app.retrieval.models import RetrievedChunk
from app.orchestrator.generation import generate_response


class MockAsyncIterator:
    """Helper to mock an asynchronous stream generator from OpenAI Completions."""

    def __init__(self, items: list) -> None:
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.items):
            res = self.items[self.index]
            self.index += 1
            return res
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_generate_response_offline_mock() -> None:
    """Verifies offline rules-based response generation under placeholder configs."""
    retrieved_chunks = [
        RetrievedChunk("FastAPI features.", "doc1.txt", "text", None, 0, 1.0),
        RetrievedChunk("Advanced config details.", "doc2.pdf", "pdf", 5, 2, 0.9),
    ]

    events = []
    async for event in generate_response(
        query="What is FastAPI?",
        rewritten_query="What is FastAPI?",
        retrieved_chunks=retrieved_chunks,
        conversation_history=[],
        long_term_memory=[],
        api_key="placeholder_key",
    ):
        events.append(event)

    # 1. Assert tokens were streamed
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) > 0
    full_text = "".join([t["text"] for t in token_events])
    assert "FastAPI" in full_text
    assert "[source: doc1.txt, chunk 0]" in full_text
    assert "[source: doc2.pdf, page 5]" in full_text

    # 2. Assert citations metadata was yielded and mapped correctly
    citation_events = [e for e in events if e["type"] == "citations"]
    assert len(citation_events) == 1
    citations = citation_events[0]["citations"]
    assert len(citations) == 2
    assert citations[0]["filename"] == "doc1.txt"
    assert citations[0]["location"] == "chunk 0"
    assert citations[1]["filename"] == "doc2.pdf"
    assert citations[1]["location"] == "page 5"
    assert citations[1]["page_number"] == 5

    # 3. Assert metrics event was yielded
    metrics_events = [e for e in events if e["type"] == "metrics"]
    assert len(metrics_events) == 1
    assert metrics_events[0]["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_generate_response_mock_openai() -> None:
    """Verifies live streaming and citation extraction by mocking OpenAI AsyncCompletions client."""
    retrieved_chunks = [
        RetrievedChunk("Mock index text.", "test_doc.txt", "txt", 1, 0, 1.0),
    ]

    mock_stream_chunks = [
        AsyncMock(choices=[AsyncMock(delta=AsyncMock(content="FastAPI is async "))]),
        AsyncMock(choices=[AsyncMock(delta=AsyncMock(content="[source: test_doc.txt, page 1]."))]),
    ]

    with patch("openai.resources.chat.completions.AsyncCompletions.create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = MockAsyncIterator(mock_stream_chunks)

        events = []
        async for event in generate_response(
            query="query",
            rewritten_query="query",
            retrieved_chunks=retrieved_chunks,
            conversation_history=[],
            long_term_memory=[],
            api_key="valid_openai_key",
        ):
            events.append(event)

        # Confirm that OpenAI completions was called
        mock_create.assert_called_once()

        # Check streamed text tokens
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 2
        assert token_events[0]["text"] == "FastAPI is async "
        assert token_events[1]["text"] == "[source: test_doc.txt, page 1]."

        # Check parsed citations metadata
        citation_events = [e for e in events if e["type"] == "citations"]
        assert len(citation_events) == 1
        citations = citation_events[0]["citations"]
        assert len(citations) == 1
        assert citations[0]["filename"] == "test_doc.txt"
        assert citations[0]["location"] == "page 1"
        assert citations[0]["page_number"] == 1
