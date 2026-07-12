# Knowify - RAG Chatbot Backend

A production-grade, highly optimized RAG chatbot backend featuring custom document ingestion splitters, hybrid vector + sparse search retrieval, cross-encoder reranking, and compiled LangGraph orchestrators.

---

## 🛠️ MCP Server Integration

This project exposes its own RAG pipeline as a Model Context Protocol (MCP) server stdio tool.

### Claude Desktop Integration

To add this RAG knowledge base search tool to Claude Desktop, edit your `claude_desktop_config.json` configuration file:
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

Add the following block to your `mcpServers` section:

```json
{
  "mcpServers": {
    "knowify-rag": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "knowify-api-1",
        "python",
        "-m",
        "app.orchestrator.mcp.server"
      ]
    }
  }
}
```

*Note: Ensure the docker-compose containers are running (`docker-compose up -d`) so that Claude Desktop can communicate with the backend process over stdio.*

---

## 🚀 Getting Started

### 1. Configure Environment Variables
Copy the template `.env` file to initialize configurations:
```bash
cp .env.example backend/.env
```
Open `backend/.env` and update the parameters:
- To run with live OpenAI and Tavily connections, replace `placeholder_key` with your active API tokens.
- To switch from local embedded mode to Docker-compose network mode, set `QDRANT_MODE=docker` and `CACHE_MODE=redis`.
- To authorize users for admin endpoints, define their email addresses in the comma-separated `ADMIN_EMAILS` variable (e.g. `ADMIN_EMAILS="admin@example.com"`).


### 2. Run Services via Docker
Start the vector store, cache, and FastAPI backend services:
```bash
docker-compose up -d
```

### 3. Run Tests
Verify all system logic by running the test suite inside the container:
```bash
docker-compose exec api pytest
```

### 4. Run RAGAS Evaluations
Run the golden set verification script inside the container:
```bash
docker-compose exec api python eval/run_ragas.py
```

