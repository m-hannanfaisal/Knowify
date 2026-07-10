import time
import structlog
from langsmith import traceable

from app.ingestion.embeddings import BaseEmbeddingProvider
from app.retrieval.hybrid import dense_retrieval, reciprocal_rank_fusion, sparse_retrieval
from app.retrieval.models import RetrievedChunk
from app.retrieval.reranker import BaseReranker

logger = structlog.get_logger(__name__)


@traceable(run_type="retriever", name="Knowify Hybrid Retrieval")



async def retrieve(
    query: str,
    collection_name: str,
    embedding_provider: BaseEmbeddingProvider,
    reranker: BaseReranker,
    top_k: int = 5,
    rrf_k: int = 60,
    rerank_top_n: int = 10,
    qdrant_url: str | None = None,
) -> list[RetrievedChunk]:
    """Runs the orchestrated hybrid retrieval and cross-encoder reranking pipeline.

    Args:
        query (str): The search query.
        collection_name (str): Search collection target.
        embedding_provider (BaseEmbeddingProvider): Query embedding service.
        reranker (BaseReranker): Reranker cross-encoder service.
        top_k (int): Number of final results to return.
        rrf_k (int): Constant smoothing factor for Reciprocal Rank Fusion.
        rerank_top_n (int): Slice size from RRF output sent to the reranker.
        qdrant_url (str | None): Custom Qdrant endpoint address.

    Returns:
        list[RetrievedChunk]: Rescored and filtered top results.
    """
    total_start = time.perf_counter()

    # 1. Embed query
    embed_start = time.perf_counter()
    query_vectors = await embedding_provider.embed_documents([query])
    query_vector = query_vectors[0]
    embed_latency = int((time.perf_counter() - embed_start) * 1000)

    # 2. Dense retrieval
    dense_start = time.perf_counter()
    dense_results = await dense_retrieval(
        query_vector=query_vector,
        collection_name=collection_name,
        top_k=rerank_top_n,
        qdrant_url=qdrant_url,
    )
    dense_latency = int((time.perf_counter() - dense_start) * 1000)

    # 3. Sparse retrieval
    sparse_start = time.perf_counter()
    sparse_results = await sparse_retrieval(
        query=query,
        collection_name=collection_name,
        top_k=rerank_top_n,
        qdrant_url=qdrant_url,
    )
    sparse_latency = int((time.perf_counter() - sparse_start) * 1000)

    # 4. Reciprocal Rank Fusion (RRF)
    rrf_start = time.perf_counter()
    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results, sparse_results=sparse_results, k=rrf_k
    )
    rrf_latency = int((time.perf_counter() - rrf_start) * 1000)

    # Slice the top RRF outputs for reranking
    top_fused = fused_results[:rerank_top_n]

    # 5. Cross-Encoder Rerank
    rerank_start = time.perf_counter()
    final_results = await reranker.rerank(query=query, chunks=top_fused, top_k=top_k)
    rerank_latency = int((time.perf_counter() - rerank_start) * 1000)

    total_latency = int((time.perf_counter() - total_start) * 1000)

    logger.info(
        "retrieval_complete",
        query=query,
        num_dense_results=len(dense_results),
        num_sparse_results=len(sparse_results),
        num_after_rrf=len(fused_results),
        num_after_rerank=len(final_results),
        embed_latency_ms=embed_latency,
        dense_latency_ms=dense_latency,
        sparse_latency_ms=sparse_latency,
        rrf_latency_ms=rrf_latency,
        rerank_latency_ms=rerank_latency,
        total_latency_ms=total_latency,
    )

    return final_results
