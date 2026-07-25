import os
import sys
import asyncio
import structlog

# Add backend directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import settings
from app.ingestion.embeddings import OpenAIEmbeddingProvider
from app.ingestion.service import embed_and_store
from app.ingestion.image_handler import ingest_image

logger = structlog.get_logger(__name__)

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample_data", "acmecrm", "docs"))
COLLECTION_NAME = "knowify_collection"

async def main():
    print(f"Starting seeding from directory: {DOCS_DIR}")
    if not os.path.exists(DOCS_DIR):
        print(f"Error: Directory {DOCS_DIR} does not exist.")
        sys.exit(1)

    # Initialize local embedding provider
    emb = OpenAIEmbeddingProvider()
    print(f"Using embedding model: {emb._model} (Local: {emb._is_local})")

    files = sorted(os.listdir(DOCS_DIR))
    print(f"Found {len(files)} files to process.")

    summary = {
        "processed": [],
        "failures": []
    }

    for filename in files:
        file_path = os.path.join(DOCS_DIR, filename)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower().replace(".", "")
        print(f"\nIngesting file: {filename} (extension: {ext})...")

        try:
            if ext in ["png", "jpg", "jpeg"]:
                chunks_count = await ingest_image(
                    image_path=file_path,
                    collection_name=COLLECTION_NAME,
                    embedding_provider=emb,
                    api_key=settings.LLM_API_KEY,
                    qdrant_url=settings.QDRANT_URL
                )
            else:
                chunks_count = await embed_and_store(
                    file_path=file_path,
                    collection_name=COLLECTION_NAME,
                    embedding_provider=emb,
                    qdrant_url=settings.QDRANT_URL
                )
            
            print(f"SUCCESS: {filename} -> {chunks_count} chunks.")
            summary["processed"].append({
                "filename": filename,
                "chunks": chunks_count
            })
        except Exception as e:
            print(f"FAILURE: {filename} -> Error: {str(e)}")
            summary["failures"].append({
                "filename": filename,
                "error": str(e)
            })

    print("\n" + "="*50)
    print("SEEDING SUMMARY")
    print("="*50)
    print(f"Total files successfully processed: {len(summary['processed'])}")
    for item in summary["processed"]:
        print(f" - {item['filename']}: {item['chunks']} chunks")
        
    print(f"\nTotal failures: {len(summary['failures'])}")
    for item in summary["failures"]:
        print(f" - {item['filename']}: {item['error']}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
