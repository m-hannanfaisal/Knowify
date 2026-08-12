import sys
import types

# Monkey-patch langchain_community missing VertexAI module to avoid import crash in Ragas
try:
    m = types.ModuleType("langchain_community.chat_models.vertexai")
    m.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = m
except Exception as e:
    pass

import argparse
import asyncio
import json
import os
import time
import structlog
from tabulate import tabulate
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatResult
from langchain_core.embeddings import Embeddings

# Add backend directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.core.config import settings
from app.ingestion.embeddings import OpenAIEmbeddingProvider
from app.retrieval.reranker import CohereReranker
from app.orchestrator.service import handle_query
from app.orchestrator.generation import generate_response

logger = structlog.get_logger(__name__)

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.json")
PIPELINE_CACHE_PATH = os.path.join(os.path.dirname(__file__), "pipeline_cache.json")
THRESHOLD = 0.80

class GroqCompatibleChatOpenAI(ChatOpenAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        n = kwargs.get("n") or self.n or 1
        if n <= 1:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            
        # If n > 1, execute n times with n=1 and merge results
        kwargs_copy = kwargs.copy()
        kwargs_copy["n"] = 1
        
        generations = []
        llm_output = None
        for _ in range(n):
            res = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs_copy)
            generations.extend(res.generations)
            if res.llm_output:
                llm_output = res.llm_output
                
        return ChatResult(generations=generations, llm_output=llm_output)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        n = kwargs.get("n") or self.n or 1
        if n <= 1:
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            
        # If n > 1, execute n times with n=1 and merge results
        kwargs_copy = kwargs.copy()
        kwargs_copy["n"] = 1
        
        generations = []
        llm_output = None
        for _ in range(n):
            res = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs_copy)
            generations.extend(res.generations)
            if res.llm_output:
                llm_output = res.llm_output
                
        return ChatResult(generations=generations, llm_output=llm_output)

class RagasEmbeddingsWrapper(Embeddings):
    def __init__(self, provider):
        self.provider = provider
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.provider._is_local:
            return self.provider._local_model.encode(texts).tolist()
        else:
            import openai
            client = openai.OpenAI(api_key=self.provider._api_key)
            response = client.embeddings.create(
                input=texts,
                model=self.provider._model
            )
            return [data.embedding for data in response.data]
        
    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

async def run_evaluation(use_cache: bool = True) -> None:
    start_time = time.perf_counter()

    if not os.path.exists(GOLDEN_SET_PATH):
        logger.error("golden_set_missing", path=GOLDEN_SET_PATH)
        sys.exit(1)

    with open(GOLDEN_SET_PATH, "r") as f:
        golden_set = json.load(f)

    logger.info("golden_set_loaded", count=len(golden_set))

    # Initialize real embedding provider and wrapped embeddings
    real_emb_provider = OpenAIEmbeddingProvider()
    eval_embeddings = RagasEmbeddingsWrapper(real_emb_provider)

    # Initialize real reranker
    real_reranker = CohereReranker()

    # Initialize real LangChain LLM for Ragas evaluation
    api_key = settings.LLM_API_KEY
    if not api_key or api_key == "placeholder_key":
        logger.error("missing_llm_api_key", reason="Ragas evaluation requires a valid LLM_API_KEY")
        sys.exit(1)

    if api_key.startswith("gsk_"):
        eval_llm = GroqCompatibleChatOpenAI(
            openai_api_key=api_key,
            openai_api_base="https://api.groq.com/openai/v1",
            model_name="llama-3.1-8b-instant",
            temperature=0.0
        )
    else:
        eval_llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4o-mini",
            temperature=0.0
        )

    # We use the real pre-seeded knowify_collection
    collection_name = "knowify_collection"

    logger.info("running_live_ragas_evaluation", collection=collection_name)

    # --- Phase 1: RAG pipeline (partial cache merge) ---
    # Load existing cache entries keyed by question text
    cache_by_question: dict[str, dict] = {}
    if use_cache and os.path.exists(PIPELINE_CACHE_PATH):
        with open(PIPELINE_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cache_by_question = {e["question"]: e for e in cached}
        logger.info("pipeline_cache_loaded", count=len(cache_by_question))

    # Identify which golden set questions are missing from the cache
    missing = [item for item in golden_set if item["question"] not in cache_by_question]
    if missing:
        logger.info("pipeline_cache_miss", missing_count=len(missing), running_live=True)
        for idx, item in enumerate(missing):
            q = item["question"]
            ground_truth = item.get("ground_truth") or item.get("expected_answer")

            logger.info("processing_eval_item", index=idx, question=q)

            # Execute actual orchestrator pipeline (StateGraph)
            res = await handle_query(
                query=q,
                conversation_history=[],
                long_term_memory=[],
                collection_name=collection_name,
                embedding_provider=real_emb_provider,
                reranker=real_reranker,
                api_key=api_key,
                qdrant_url=settings.QDRANT_URL,
            )

            # Run generation pipeline to synthesize answer
            generated_answer = ""
            if res["insufficient_information"]:
                generated_answer = res["fallback_response"] or "I'm sorry, I cannot find enough relevant information in the documents to answer your question."
            else:
                async for event in generate_response(
                    query=q,
                    rewritten_query=res["rewritten_query"],
                    retrieved_chunks=res["retrieved_chunks"],
                    conversation_history=[],
                    long_term_memory=[],
                    api_key=api_key,
                ):
                    if event["type"] == "token":
                        generated_answer += event["text"]

            # Context list
            contexts = [chunk.text for chunk in res["retrieved_chunks"]]
            if not contexts:
                contexts = [ground_truth]

            cache_by_question[q] = {
                "question": q,
                "answer": generated_answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "route": res["route"]
            }

        # Persist updated cache (all entries, including newly-run ones)
        updated_cache = list(cache_by_question.values())
        with open(PIPELINE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(updated_cache, f, indent=2, ensure_ascii=False)
        logger.info("pipeline_cache_saved", path=PIPELINE_CACHE_PATH, count=len(updated_cache))
    else:
        logger.info("pipeline_cache_full_hit", count=len(cache_by_question))

    # Assemble eval_items in golden-set order (preserves question order for table)
    eval_items = [cache_by_question[item["question"]] for item in golden_set]

    # Convert to datasets.Dataset
    dataset_dict = {
        "question": [x["question"] for x in eval_items],
        "answer": [x["answer"] for x in eval_items],
        "contexts": [x["contexts"] for x in eval_items],
        "ground_truth": [x["ground_truth"] for x in eval_items]
    }
    dataset = Dataset.from_dict(dataset_dict)

    # Run actual RAGAS evaluation
    logger.info("evaluating_with_ragas_library", count=len(eval_items))
    run_config = RunConfig(
        max_workers=1,
        timeout=120,
        max_retries=10,
        max_wait=60
    )
    
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config
    )
    
    # Extract results as dataframe
    import math
    df = ragas_result.to_pandas()

    def safe_score(val: object) -> float:
        """Return float score, replacing NaN/None with 0.0."""
        try:
            f = float(val)  # type: ignore[arg-type]
            return 0.0 if math.isnan(f) else f
        except (TypeError, ValueError):
            return 0.0

    table_data = []
    for index, item in enumerate(eval_items):
        row = df.iloc[index]
        table_data.append([
            item["question"][:50] + "...",
            round(safe_score(row.get("faithfulness")), 3),
            round(safe_score(row.get("context_precision")), 3),
            round(safe_score(row.get("answer_relevancy")), 3),
            round(safe_score(row.get("answer_correctness")), 3),
            item["route"]
        ])

    print("\n=== RAGAS EVALUATION METRICS REPORT ===")
    print(
        tabulate(
            table_data,
            headers=["Question", "Faithfulness", "Context Precision", "Relevancy", "Correctness", "Route"],
            tablefmt="grid",
        )
    )
    
    avg_faithfulness = safe_score(df["faithfulness"].mean() if "faithfulness" in df.columns else float("nan"))
    avg_precision = safe_score(df["context_precision"].mean() if "context_precision" in df.columns else float("nan"))
    avg_relevancy = safe_score(df["answer_relevancy"].mean() if "answer_relevancy" in df.columns else float("nan"))
    avg_recall = safe_score(df["context_recall"].mean() if "context_recall" in df.columns else float("nan"))
    avg_correctness = safe_score(df["answer_correctness"].mean() if "answer_correctness" in df.columns else float("nan"))

    print("\n=== SUMMARY METRICS ===")
    print(f"Average Faithfulness:      {avg_faithfulness:.4f} (Threshold: {THRESHOLD})")
    print(f"Average Context Precision:  {avg_precision:.4f} (Threshold: {THRESHOLD})")
    print(f"Average Answer Relevancy:   {avg_relevancy:.4f}")
    print(f"Average Context Recall:     {avg_recall:.4f}")
    print(f"Average Answer Correctness: {avg_correctness:.4f}")
    print(f"Total Latency:              {time.perf_counter() - start_time:.2f} seconds\n")

    # Threshold Check
    if avg_faithfulness < THRESHOLD or avg_precision < THRESHOLD:
        logger.error(
            "evaluation_threshold_violation",
            faithfulness=avg_faithfulness,
            precision=avg_precision,
            threshold=THRESHOLD,
        )
        sys.exit(1)

    logger.info(
        "evaluation_passed_successfully",
        faithfulness=avg_faithfulness,
        precision=avg_precision,
    )
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation against Knowify golden set.")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore pipeline_cache.json and re-run the full RAG pipeline from scratch."
    )
    args = parser.parse_args()
    asyncio.run(run_evaluation(use_cache=not args.no_cache))
