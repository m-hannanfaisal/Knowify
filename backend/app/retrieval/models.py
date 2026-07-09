class RetrievedChunk:
    """Represents a chunk retrieved from storage with a ranking score.

    Attributes:
        text (str): Content text of the chunk.
        source_filename (str): Source file name.
        file_type (str): Format type of the source file.
        page_number (int | None): Source page number, if applicable.
        chunk_index (int): Index sequence number of the chunk.
        score (float): Similarity/ranking score assigned by retrieval or reranking.
    """

    def __init__(
        self,
        text: str,
        source_filename: str,
        file_type: str,
        page_number: int | None,
        chunk_index: int,
        score: float,
    ) -> None:
        self.text = text
        self.source_filename = source_filename
        self.file_type = file_type
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.score = score

    def to_dict(self) -> dict:
        """Converts the RetrievedChunk to a dictionary format."""
        return {
            "text": self.text,
            "source_filename": self.source_filename,
            "file_type": self.file_type,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "score": self.score,
        }
