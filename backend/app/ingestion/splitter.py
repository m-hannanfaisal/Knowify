import pandas as pd


class DocumentChunk:
    """Represents a text chunk extracted from a document with associated metadata.

    Attributes:
        text (str): The text content of the chunk.
        source_filename (str): Name of the source file.
        file_type (str): Type of the file (e.g. pdf, csv).
        page_number (int | None): Page number, if applicable.
        chunk_index (int): Index of the chunk in the document.
    """

    def __init__(
        self,
        text: str,
        source_filename: str,
        file_type: str,
        page_number: int | None,
        chunk_index: int,
        metadata: dict | None = None,
    ) -> None:
        self.text = text
        self.source_filename = source_filename
        self.file_type = file_type
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.metadata = metadata or {}



class RecursiveCharacterTextSplitter:
    """Splits raw text recursively using a hierarchy of separators to fit chunk sizes."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []
        separator = separators[-1] if separators else ""
        new_separators = []
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1 :]
                break

        if separator != "":
            splits = text.split(separator)
        else:
            splits = list(text)

        good_splits: list[str] = []
        for s in splits:
            if len(s) <= self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                rec_splits = self._split_text(s, new_separators)
                final_chunks.extend(rec_splits)

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))

        return final_chunks

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        docs: list[str] = []
        current_doc: list[str] = []
        total = 0
        for s in splits:
            len_s = len(s)
            if total + len_s + (len(separator) if current_doc else 0) > self.chunk_size:
                if current_doc:
                    docs.append(separator.join(current_doc))
                    overlap_doc = []
                    overlap_len = 0
                    for prev_s in reversed(current_doc):
                        if overlap_len + len(prev_s) + (len(separator) if overlap_doc else 0) <= self.chunk_overlap:
                            overlap_doc.insert(0, prev_s)
                            overlap_len += len(prev_s) + (len(separator) if len(overlap_doc) > 1 else 0)
                        else:
                            break
                    current_doc = overlap_doc
                    total = overlap_len
                else:
                    docs.append(s)
                    continue

            current_doc.append(s)
            total += len_s + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            docs.append(separator.join(current_doc))
        return docs

    def split_text(self, text: str) -> list[str]:
        """Splits text into chunks.

        Args:
            text (str): Plain text to split.

        Returns:
            list[str]: Split text chunks.
        """
        if not text.strip():
            return []
        return self._split_text(text, self.separators)


class TabularSplitter:
    """Splits tabular DataFrames into row groups formatted as Markdown tables."""

    def __init__(self, row_group_size: int = 10) -> None:
        self.row_group_size = row_group_size

    def split_dataframe(self, df: pd.DataFrame) -> list[str]:
        """Slices df into row groups and converts each group into a Markdown table.

        Args:
            df (pd.DataFrame): Dataframe.

        Returns:
            list[str]: Row group Markdown strings.
        """
        chunks: list[str] = []
        num_rows = len(df)
        if num_rows == 0:
            return []

        for i in range(0, num_rows, self.row_group_size):
            row_group = df.iloc[i : i + self.row_group_size]
            # Convert to markdown table format preserving headers
            md_table = row_group.to_markdown(index=False)
            chunks.append(str(md_table))

        return chunks
