import os
import pytest
import pandas as pd
from unittest.mock import MagicMock

from app.core.config import settings
from app.ingestion.parser import DocumentParser, ParsedPage, ParsedTable
from app.ingestion.splitter import RecursiveCharacterTextSplitter, TabularSplitter
from app.ingestion.embeddings import MockEmbeddingProvider
from app.ingestion.store import QdrantStore
from app.ingestion.service import embed_and_store



@pytest.fixture(scope="module")
def fixtures_dir() -> str:
    """Returns the directory of test fixtures, generating them if not present."""
    current_dir = os.path.dirname(__file__)
    fixtures_path = os.path.join(current_dir, "fixtures")
    if not os.path.exists(os.path.join(fixtures_path, "sample.txt")):
        # If running inside python shell/terminal before generator was called
        from tests.generate_fixtures import main as gen_main
        gen_main()
    return fixtures_path


def test_parse_txt(fixtures_dir: str) -> None:
    """Test plain text parser."""
    parser = DocumentParser()
    txt_path = os.path.join(fixtures_dir, "sample.txt")
    output = parser.parse(txt_path)
    assert isinstance(output, list)
    assert len(output) == 1
    assert "Hello World from a plain text file." in output[0].content


def test_parse_markdown(fixtures_dir: str) -> None:
    """Test markdown parser."""
    parser = DocumentParser()
    md_path = os.path.join(fixtures_dir, "sample.md")
    output = parser.parse(md_path)
    assert isinstance(output, list)
    assert len(output) == 1
    assert "# Sample Title" in output[0].content


def test_parse_json(fixtures_dir: str) -> None:
    """Test JSON parser."""
    parser = DocumentParser()
    json_path = os.path.join(fixtures_dir, "sample.json")
    output = parser.parse(json_path)
    assert isinstance(output, list)
    assert len(output) == 1
    assert "title" in output[0].content


def test_parse_html(fixtures_dir: str) -> None:
    """Test HTML parser."""
    parser = DocumentParser()
    html_path = os.path.join(fixtures_dir, "sample.html")
    output = parser.parse(html_path)
    assert isinstance(output, list)
    assert len(output) == 1
    assert "Hello HTML" in output[0].content


def test_parse_docx(fixtures_dir: str) -> None:
    """Test DOCX parser."""
    parser = DocumentParser()
    docx_path = os.path.join(fixtures_dir, "sample.docx")
    output = parser.parse(docx_path)
    assert isinstance(output, list)
    assert len(output) == 1
    assert "docx" in output[0].content.lower()
    assert "Header A | Header B" in output[0].content


def test_parse_pdf(fixtures_dir: str) -> None:
    """Test PDF parser page content and numbering."""
    parser = DocumentParser()
    pdf_path = os.path.join(fixtures_dir, "sample.pdf")
    output = parser.parse(pdf_path)
    assert isinstance(output, list)
    assert len(output) == 1
    assert output[0].page_number == 1
    assert "Hello PDF World" in output[0].content


def test_parse_csv_and_xlsx(fixtures_dir: str) -> None:
    """Test tabular parsers."""
    parser = DocumentParser()
    
    csv_path = os.path.join(fixtures_dir, "sample.csv")
    csv_output = parser.parse(csv_path)
    assert isinstance(csv_output, ParsedTable)
    assert list(csv_output.df.columns) == ["Name", "Age", "City"]
    assert len(csv_output.df) == 3

    xlsx_path = os.path.join(fixtures_dir, "sample.xlsx")
    xlsx_output = parser.parse(xlsx_path)
    assert isinstance(xlsx_output, ParsedTable)
    assert list(xlsx_output.df.columns) == ["Name", "Age", "City"]
    assert len(xlsx_output.df) == 3


def test_recursive_splitter() -> None:
    """Test RecursiveCharacterTextSplitter with size and overlap."""
    text = "Paragraph 1 is here.\n\nParagraph 2 is here. It is somewhat longer text."
    splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=5)
    chunks = splitter.split_text(text)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 30


def test_tabular_splitter() -> None:
    """Test TabularSplitter chunks by row-groups preserving headers."""
    df = pd.DataFrame({
        "Col1": [1, 2, 3, 4, 5],
        "Col2": ["A", "B", "C", "D", "E"]
    })
    splitter = TabularSplitter(row_group_size=2)
    chunks = splitter.split_dataframe(df)
    
    # 5 rows total, group size 2 -> 3 chunks
    assert len(chunks) == 3
    # Check that headers are present in chunks
    assert "Col1" in chunks[0] and "Col2" in chunks[0]
    assert "Col1" in chunks[1] and "Col2" in chunks[1]
    assert "Col1" in chunks[2] and "Col2" in chunks[2]


@pytest.mark.asyncio
async def test_mock_embedding_provider() -> None:
    """Test deterministic mock embedding outputs."""
    provider = MockEmbeddingProvider(dimension=64)
    assert provider.dimension == 64
    
    text1 = "sample text chunk"
    text2 = "another text chunk"
    
    vec1 = await provider.embed_documents([text1])
    vec2 = await provider.embed_documents([text1])
    vec3 = await provider.embed_documents([text2])
    
    assert len(vec1[0]) == 64
    assert vec1 == vec2  # Must be deterministic
    assert vec1 != vec3  # Different text must produce different vector


@pytest.mark.asyncio
async def test_embed_and_store_integration(fixtures_dir: str) -> None:
    """Integration test: end-to-end ingestion pipeline writing to Qdrant."""
    txt_path = os.path.join(fixtures_dir, "sample.txt")
    collection_name = "test_ingestion_collection"
    embedding_provider = MockEmbeddingProvider(dimension=64)
    
    # Let's run embed_and_store. We use settings.QDRANT_URL to resolve
    # the correct address inside/outside Docker Compose.
    count = await embed_and_store(
        file_path=txt_path,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        qdrant_url=settings.QDRANT_URL
    )
    
    assert count > 0

    # Let's clean up collection via a quick store call
    store = QdrantStore(url=settings.QDRANT_URL)
    collections_response = await store.client.get_collections()
    exists = any(c.name == collection_name for c in collections_response.collections)
    assert exists
    
    # Delete the collection after test
    await store.client.delete_collection(collection_name)
