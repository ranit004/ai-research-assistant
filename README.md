# AI Research Assistant

Backend-only multi-step research assistant API. The finished system will ingest documents into a knowledge base, answer questions from it with cited evidence, and handle compound, ambiguous, and follow-up questions through multi-step LangGraph reasoning.

**Status: foundation milestone.** The service boots, exposes `GET /health`, and is Docker-ready. Document ingestion, RAG, and LangGraph reasoning are not implemented yet (see [Roadmap](#roadmap)).

## Stack

- Python 3.11+
- FastAPI, served by Uvicorn
- LangChain and LangGraph (installed, integration pending)
- Qdrant vector database (client installed, integration pending)
- pytest for tests

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .                 # makes the package importable for local uvicorn

cp .env.example .env             # adjust if needed; .env is gitignored

uvicorn research_assistant.main:app --reload
```

Then open:

- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Tests

```bash
pytest
```

## Docker

```bash
docker build -t research-assistant .
docker run -p 8000:8000 research-assistant
```

## Configuration

All settings are read from environment variables or a local `.env` file (copy `.env.example`). Secrets are never stored in source code.

| Variable      | Default                  | Description                              |
| ------------- | ------------------------ | ---------------------------------------- |
| `APP_NAME`    | `AI Research Assistant`  | Service name shown in `/docs` and `/health` |
| `ENVIRONMENT` | `development`            | Deployment environment label             |
| `DEBUG`       | `false`                  | FastAPI debug mode                       |
| `LOG_LEVEL`   | `INFO`                   | Root log level                           |
| `API_HOST`    | `0.0.0.0`                | Uvicorn bind host                        |
| `API_PORT`    | `8000`                   | Uvicorn bind port                        |

## Project structure

```
src/research_assistant/
├── api/
│   └── health.py        # GET /health
├── config.py            # Environment-driven settings (pydantic-settings)
├── logging_config.py    # Console logging setup
└── main.py              # App factory, lifespan, router registration
tests/
├── conftest.py          # TestClient fixture (runs lifespan)
└── test_health.py
Dockerfile               # Slim image, non-root user, cached dependency layer
requirements.txt         # Runtime dependencies (pinned)
requirements-dev.txt     # Test dependencies
```

## Roadmap

Not implemented yet; each item will land in a later milestone:

1. Document ingestion into a knowledge base.
2. Question answering over the knowledge base with evidence and sources.
3. LangGraph graph for multi-step reasoning (compound and ambiguous questions).
4. Conversation context for follow-up questions.
5. Grounded refusal behavior for unsupported questions.
6. Qdrant integration and deployment configuration.
