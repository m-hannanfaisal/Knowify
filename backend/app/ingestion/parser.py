import json
import os
from typing import Union
import docx
import pandas as pd
import pypdf
from bs4 import BeautifulSoup


class ParsedPage:
    """Represents a single parsed page of a text-based document.

    Attributes:
        content (str): The text content of the page.
        page_number (int | None): The page number, if applicable.
    """

    def __init__(self, content: str, page_number: int | None = None) -> None:
        self.content = content
        self.page_number = page_number


class ParsedTable:
    """Represents a parsed tabular document.

    Attributes:
        df (pd.DataFrame): The parsed data as a pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


ParsedDocumentOutput = Union[list[ParsedPage], ParsedTable]


def parse_pdf(file_path: str) -> list[ParsedPage]:
    """Parses a PDF file page-by-page.

    Args:
        file_path (str): Absolute path to the PDF.

    Returns:
        list[ParsedPage]: Parsed page objects.
    """
    pages: list[ParsedPage] = []
    reader = pypdf.PdfReader(file_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(ParsedPage(content=text, page_number=i + 1))
    return pages


def parse_docx(file_path: str) -> list[ParsedPage]:
    """Parses a DOCX file and returns its paragraphs and table content.

    Args:
        file_path (str): Absolute path to the DOCX.

    Returns:
        list[ParsedPage]: A list containing a single ParsedPage with the document text.
    """
    doc = docx.Document(file_path)
    full_text: list[str] = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                full_text.append(" | ".join(row_text))
    return [ParsedPage(content="\n".join(full_text), page_number=None)]


def parse_html(file_path: str) -> list[ParsedPage]:
    """Parses an HTML file extracting clean readable text.

    Args:
        file_path (str): Absolute path to the HTML.

    Returns:
        list[ParsedPage]: HTML text content page.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n")
    cleaned_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return [ParsedPage(content=cleaned_text, page_number=None)]


def parse_csv(file_path: str) -> ParsedTable:
    """Parses a CSV file into a ParsedTable.

    Args:
        file_path (str): Absolute path to the CSV.

    Returns:
        ParsedTable: DataFrame wrapper.
    """
    df = pd.read_csv(file_path)
    return ParsedTable(df=df)


def parse_xlsx(file_path: str) -> ParsedTable:
    """Parses an Excel spreadsheet into a ParsedTable.

    Args:
        file_path (str): Absolute path to the XLSX.

    Returns:
        ParsedTable: DataFrame wrapper.
    """
    df = pd.read_excel(file_path)
    return ParsedTable(df=df)


def parse_json(file_path: str) -> list[ParsedPage]:
    """Parses a JSON file formatting it as a pretty string.

    Args:
        file_path (str): Absolute path to the JSON.

    Returns:
        list[ParsedPage]: Format-indented JSON string representation.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = json.dumps(data, indent=2)
    return [ParsedPage(content=text, page_number=None)]


def parse_text_file(file_path: str) -> list[ParsedPage]:
    """Parses a raw text or markdown file.

    Args:
        file_path (str): Absolute path to the text file.

    Returns:
        list[ParsedPage]: File content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [ParsedPage(content=text, page_number=None)]


class DocumentParser:
    """Handles parsing logic delegation based on file extensions."""

    def parse(self, file_path: str) -> ParsedDocumentOutput:
        """Parses the document at file_path based on its file extension.

        Args:
            file_path (str): Path to the document.

        Returns:
            ParsedDocumentOutput: Parsed data structure.

        Raises:
            ValueError: If extension is unsupported.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return parse_pdf(file_path)
        elif ext == ".docx":
            return parse_docx(file_path)
        elif ext in (".html", ".htm"):
            return parse_html(file_path)
        elif ext == ".csv":
            return parse_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            return parse_xlsx(file_path)
        elif ext == ".json":
            return parse_json(file_path)
        elif ext in (".md", ".markdown", ".txt"):
            return parse_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
