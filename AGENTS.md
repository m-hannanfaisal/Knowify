# AI Agent Coding Standards and Rules

Every AI agent and developer working in this repository must adhere to the following standards:

- **Language and Framework**: Python 3.11+, FastAPI for all backend services.
- **Type Hinting and Documentation**: Fully specify type hints on all functions and write descriptive docstrings on all public methods.
- **Single Responsibility Principle**: Ensure each file handles exactly one responsibility. No monolithic or "god-modules".
- **Secrets Management**: Never hardcode API keys, passwords, or secrets. Always use environment variables managed via `pydantic-settings`.
- **Testing Requirements**: Every new module must include a `tests/` folder with `pytest` unit/integration coverage for all core logic.
- **Structured Logging**: Use structured JSON logging via `structlog` in every module. Do not use print statements.
- **Asynchronous I/O**: Use `async`/`await` for all input/output operations, including database transactions, vector database calls, LLM requests, and HTTP client requests.
