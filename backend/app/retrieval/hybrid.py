import re
from rank_bm25 import BM25Okapi

from app.ingestion.store import QdrantStore
from app.retrieval.models import RetrievedChunk


def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer for BM25 text splitting.

    Args:
        text (str): Input text.

    Returns:
        list[str]: Tokenized words.
    """
    return re.findall(r"\w+", text.lower())


async def dense_retrieval(
    query_vector: list[float],
    collection_name: str,
    top_k: int,
    qdrant_url: str | None = None,
) -> list[RetrievedChunk]:
    """Queries Qdrant using dense vector cosine similarity.

    Args:
        query_vector (list[float]): Embedded search query.
        collection_name (str): Destination collection name.
        top_k (int): Number of dense results to retrieve.
        qdrant_url (str | None): Custom Qdrant endpoint.

    Returns:
        list[RetrievedChunk]: Retrieved chunks with cosine scores.
    """
    store = QdrantStore(url=qdrant_url)
    response = await store.client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    hits = response.points


    results: list[RetrievedChunk] = []
    for hit in hits:
        payload = hit.payload
        if payload and "text" in payload:
            results.append(
                RetrievedChunk(
                    text=payload["text"],
                    source_filename=payload["source_filename"],
                    file_type=payload["file_type"],
                    page_number=payload.get("page_number"),
                    chunk_index=payload["chunk_index"],
                    score=float(hit.score),
                )
            )
    return results


async def sparse_retrieval(
    query: str,
    collection_name: str,
    top_k: int,
    qdrant_url: str | None = None,
) -> list[RetrievedChunk]:
    """Fetches text chunks from Qdrant and ranks them using BM25.

    Args:
        query (str): The search query.
        collection_name (str): Destination collection name.
        top_k (int): Number of sparse results to retrieve.
        qdrant_url (str | None): Custom Qdrant endpoint.

    Returns:
        list[RetrievedChunk]: Top ranked chunks with BM25 scores.
    """
    store = QdrantStore(url=qdrant_url)
    response = await store.client.scroll(
        collection_name=collection_name,
        limit=10000,
        with_payload=True,
    )
    points = response[0]

    if not points:
        return []

    corpus: list[list[str]] = []
    chunk_map = []
    for p in points:
        payload = p.payload
        if payload and "text" in payload:
            corpus.append(tokenize(payload["text"]))
            chunk_map.append(p)

    if not corpus:
        return []

    bm25 = BM25Okapi(corpus)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    results: list[RetrievedChunk] = []
    for idx, score in enumerate(scores):
        if score <= 0.0:
            continue
        p = chunk_map[idx]
        payload = p.payload
        results.append(
            RetrievedChunk(
                text=payload["text"],
                source_filename=payload["source_filename"],
                file_type=payload["file_type"],
                page_number=payload.get("page_number"),
                chunk_index=payload["chunk_index"],
                score=float(score),
            )
        )

    # Sort descending by BM25 score
    results.sort(key=lambda c: c.score, reverse=True)
    return results[:top_k]


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Merges two ranked lists using Reciprocal Rank Fusion (RRF).

    Args:
        dense_results (list[RetrievedChunk]): Chunks sorted by dense retrieval.
        sparse_results (list[RetrievedChunk]): Chunks sorted by sparse retrieval.
        k (int): Constant parameter for RRF smoothing.

    Returns:
        list[RetrievedChunk]: Merged and sorted list with RRF score.
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, RetrievedChunk] = {}

    def _get_key(c: RetrievedChunk) -> str:
        return f"{c.source_filename}_{c.chunk_index}"

    for rank, chunk in enumerate(dense_results):
        key = _get_key(chunk)
        doc_map[key] = chunk
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank + 1))

    for rank, chunk in enumerate(sparse_results):
        key = _get_key(chunk)
        doc_map[key] = chunk
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank + 1))

    fused_results: list[RetrievedChunk] = []
    for key, score in rrf_scores.items():
        original = doc_map[key]
        fused_results.append(
            RetrievedChunk(
                text=original.text,
                source_filename=original.source_filename,
                file_type=original.file_type,
                page_number=original.page_number,
                chunk_index=original.chunk_index,
                score=score,
            )
        )

    fused_results.sort(key=lambda c: c.score, reverse=True)
    return fused_results
