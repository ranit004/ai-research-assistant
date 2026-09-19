# AI Research Assistant

A backend-only REST API that answers questions grounded in an uploaded knowledge base. The service uses **FastAPI**, **LangChain**, **LangGraph**, **Groq**, and **Qdrant** to execute a genuine multi-step research workflow before generating any answer.

---

## What it does

1. You upload documents (PDF, plain text, Markdown).
2. The service chunks and embeds them locally into a Qdrant vector store using Sentence Transformers.
3. When you ask a question, an eight-node LangGraph pipeline runs: it classifies the question, rewrites it, decomposes it into sub-questions, retrieves and evaluates evidence for each sub-question, refines the search if evidence is insufficient (up to a bounded limit), synthesises an answer from retrieved chunks only, then verifies the answer before returning it.
4. Every response includes the source chunks that grounded the answer.

The system does not generate answers from prior knowledge. If the knowledge base does not contain enough information, it says so explicitly.

---

## Architecture

```
src/research_assistant/
├── main.py                  # App factory, router registration, lifespan
├── config.py                # All settings from environment / .env (pydantic-settings)
├── logging_config.py        # Console logger setup
├── api/
│   ├── health.py            # GET /health
│   ├── documents.py         # POST /documents (upload alias)
│   ├── conversations.py     # POST /conversations, GET /{id}, POST /{id}/ask
│   ├── ingest.py            # POST /ingest/document (core ingest logic)
│   ├── research.py          # POST /research (stateless research endpoint)
│   └── retrieve.py          # POST /retrieve (raw similarity search)
├── agent/
│   ├── graph.py             # LangGraph StateGraph assembly
│   ├── nodes.py             # Eight node functions powered by Groq Chat LLM
│   ├── prompts.py           # All LLM prompt templates
│   └── state.py             # Typed ResearchState, EvidenceItem, SubQuestionResult
├── conversations/
│   └── store.py             # Thread-safe in-memory conversation store
├── ingestion/
│   ├── chunker.py           # Fixed-size character chunking with overlap
│   ├── models.py            # DocumentChunk Pydantic model
│   └── parser.py            # MIME detection, text extraction, text cleaning
└── vectorstore/
    ├── client.py            # Qdrant client (lazy, cached), upsert, query_points search
    └── embedder.py          # Local Sentence Transformers embedding model (all-MiniLM-L6-v2)
```

---

## LangGraph workflow

```
START
  │
  ▼
understand_query      ← classifies as standalone / ambiguous / follow_up / compound
  │
  ▼
rewrite_query         ← rewrites to a standalone, unambiguous form using conversation context
  │
  ▼
decompose_query       ← splits compound questions into focused sub-questions (max 4)
  │
  ▼
retrieve_evidence     ← embeds and searches Qdrant for each sub-question independently
  │
  ▼
evaluate_evidence     ← LLM judges whether retrieved chunks actually support an answer
  │
  ├─── evidence sufficient ────────────────────────────────────────┐
  │                                                                │
  └─── insufficient AND iterations < max_research_iterations ──►  │
         │                                                         │
         ▼                                                         │
       refine_query   ← generates an alternative search query      │
         │                                                         │
         └───► retrieve_evidence  (loop back)                      │
                                                                    │
                                                            ◄──────┘
                                                            synthesize_answer
                                                                    │
                                                                    ▼
                                                            verify_answer   ← fact-checks against retrieved evidence
                                                                    │
                                                                   END
```

**Loop termination:** `MAX_RESEARCH_ITERATIONS` (default 3) caps the refine-retrieve cycle unconditionally. When evidence is still insufficient at that limit, the synthesiser returns the canonical refusal message.

---

## Knowledge base

Documents are chunked at 512 characters with a 64-character overlap. Each chunk is embedded into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2` and stored in Qdrant with the following metadata:

| Field | Description |
|---|---|
| `document_id` | UUID assigned on upload |
| `chunk_id` | UUID for this chunk |
| `chunk_index` | Position within the document |
| `title` | Filename stem (sanitised, display only) |
| `section` | Nearest Markdown heading before this chunk |
| `source` | Original filename (sanitised, never used as a path) |
| `text` | Chunk content (bidi overrides and control characters stripped) |

---

## Setup

### Prerequisites

- Python 3.11+
- A running Qdrant instance (`docker run -p 6333:6333 qdrant/qdrant`)
- A Groq API key (get a free key at [console.groq.com](https://console.groq.com))

### Local development

```bash
git clone https://github.com/ranit004/ai-research-assistant.git
cd ai-research-assistant

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install runtime and dev dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

# Configure
cp .env.example .env
# Edit .env and set GROQ_API_KEY

# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Start the API
uvicorn research_assistant.main:app --reload
```

**Swagger UI:** http://127.0.0.1:8000/docs  
**Health check:** http://127.0.0.1:8000/health

Alternatively, with `uv`:

```bash
uv run uvicorn research_assistant.main:app --reload
```

---

## Environment variables

Copy `.env.example` to `.env`. Every variable has a safe default except `GROQ_API_KEY`.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** API key for Groq LLM inference. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq chat model for graph nodes. |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant base URL. |
| `QDRANT_COLLECTION` | `research_docs` | Collection name. |
| `QDRANT_API_KEY` | — | Set for authenticated Qdrant Cloud instances. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model. |
| `EMBEDDING_DIM` | `384` | Embedding vector dimensions. |
| `MAX_UPLOAD_BYTES` | `10485760` | Maximum file size (10 MB). |
| `CHUNK_SIZE` | `512` | Characters per chunk. |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks. |
| `MAX_SUB_QUESTIONS` | `4` | Maximum sub-questions from decomposition. |
| `MAX_RESEARCH_ITERATIONS` | `3` | Maximum refine-retrieve loops. |
| `EVIDENCE_MIN_SCORE` | `0.35` | Minimum cosine similarity to consider a hit. |
| `EVIDENCE_TOP_K` | `5` | Retrieved chunks per sub-question. |
| `MAX_QUESTION_LENGTH` | `2000` | Maximum question characters. |
| `MAX_HISTORY_TURNS` | `10` | Conversation turns loaded into the graph. |

---

## API endpoints

### Health

```
GET /health
```

Returns service liveness and version.

---

### Document ingestion

```
POST /documents
Content-Type: multipart/form-data
Body: file=<upload>
```

Also available at the path `POST /ingest/document`.

Accepted types: `application/pdf`, `text/plain`, `text/markdown` (max 10 MB, max 500 chunks).

**Response:**

```json
{
  "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "chunk_count": 42,
  "title": "kubernetes-concepts"
}
```

---

### Conversations

```
POST /conversations
```

Creates a new conversation session. Returns `conversation_id` and `created_at`.

```
GET /conversations/{conversation_id}
```

Returns the full message history for a conversation.

```
POST /conversations/{conversation_id}/ask
Content-Type: application/json
Body: {"question": "What is a ReplicaSet?"}
```

Runs the full multi-step research workflow with conversation history context.

**Response:**

```json
{
  "answer": "A ReplicaSet ensures that a specified number of Pod replicas are running at any given time. [kubernetes-concepts]",
  "supported": true,
  "sources": [
    {
      "document_id": "3fa85f64-...",
      "chunk_id": "7c9e6679-...",
      "title": "kubernetes-concepts",
      "section": "Workloads",
      "source": "kubernetes-concepts.md"
    }
  ],
  "conversation_id": "09892583-...",
  "metadata": {
    "sub_question_count": 1,
    "research_iteration_count": 0,
    "query_type": "standalone"
  }
}
```

---

### Research (stateless)

```
POST /research
Content-Type: application/json
Body: {
  "query": "What is a pod?",
  "conversation_history": []
}
```

Runs the research workflow without creating or loading a conversation session.

---

### Similarity search (debug)

```
POST /retrieve
Content-Type: application/json
Body: {"query": "ReplicaSet", "top_k": 5, "score_threshold": 0.4}
```

Raw Qdrant similarity search using `query_points()`. Useful for inspecting retrieval quality.

---

## Example requests

### Upload a document

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@kubernetes-concepts.md"
```

### Create a conversation and ask a question

```bash
# Create session
CID=$(curl -s -X POST http://localhost:8000/conversations | jq -r .conversation_id)

# Ask a question
curl -X POST http://localhost:8000/conversations/$CID/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a ReplicaSet?"}'

# Ask a follow-up (history is loaded automatically)
curl -X POST http://localhost:8000/conversations/$CID/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does it differ from a Deployment?"}'
```

---

## Example difficult questions

These question types exercise the multi-step research workflow explicitly:

| Type | Example |
|---|---|
| **Compound** | "What is a ReplicaSet and how does it differ from a Deployment?" |
| **Ambiguous** | "Tell me about controllers." (the graph will rewrite to the most plausible interpretation) |
| **Follow-up** | Ask "What is a pod?" then "How many containers can it hold?" in the same conversation |
| **Multi-hop** | "Who created Kubernetes, when was it open-sourced, and what was the original use case?" |
| **Unsupported** | "What is the capital of Mars?" (returns the canonical refusal, not a hallucinated answer) |

---

## Security considerations

**Secrets:** `GROQ_API_KEY` and `QDRANT_API_KEY` are typed as `SecretStr` via pydantic-settings. They are never logged or surfaced in any response, including error responses.

**File handling:** MIME type is detected from magic bytes, not from the filename. Filenames are sanitised and never used as filesystem paths. Files exceeding 10 MB or producing more than 500 chunks are rejected before embedding begins.

**Input validation:** All API inputs are validated by Pydantic. Questions above 2,000 characters or containing only whitespace are rejected with HTTP 422.

**Prompt injection:** Retrieved document text is wrapped in explicit XML-style delimiters and labelled as untrusted data in every prompt template. Unicode bidi override characters and invisible codepoints are stripped from all document text during ingestion and again before each LLM call. Evidence text is capped at 1,500 characters per chunk. Conversation history is capped at `MAX_HISTORY_TURNS * 2` messages and 4,000 total characters before injection.

**Resource limits:** The LangGraph loop terminates unconditionally at `MAX_RESEARCH_ITERATIONS`. The conversation store evicts oldest messages when a session exceeds `MAX_HISTORY_TURNS * 4` stored messages.

**Error handling:** Internal exceptions return a generic message. Stack traces and library error text are never surfaced to callers.

**Docker:** The image runs as a non-root `appuser`. The `.env` file is excluded from the Docker context via `.dockerignore`.

**SSRF:** The Qdrant URL is set by the operator at startup, not by any user request. No user-controlled URL is ever fetched.

---

## Testing

```bash
# Run all tests (no external services required — all I/O is mocked)
uv run pytest -v

# Or with a plain pip install
pytest -v
```

Test files:

| File | Coverage |
|---|---|
| `test_health.py` | `/health` endpoint |
| `test_api.py` | Document ingestion, retrieval endpoints, and Qdrant `query_points` compatibility |
| `test_ingestion.py` | Parser, chunker, and DocumentChunk model |
| `test_research_graph.py` | All eight LangGraph nodes and graph routing |
| `test_conversations.py` | Conversation store and `/conversations` endpoints |
| `test_security_and_eval.py` | Security controls and AI/RAG evaluation dataset |

**77 tests, 0 failures** (no network calls required).

---

## Deployment

### Docker

```bash
docker build -t research-assistant .
docker run -p 8000:8000 \
  -e GROQ_API_KEY=gsk_... \
  -e QDRANT_URL=http://qdrant:6333 \
  research-assistant
```

### Docker Compose (with Qdrant)

```yaml
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      GROQ_API_KEY: "${GROQ_API_KEY}"
      GROQ_MODEL: "${GROQ_MODEL:-openai/gpt-oss-120b}"
      QDRANT_URL: "http://qdrant:6333"
      ENVIRONMENT: "production"
    depends_on:
      - qdrant

volumes:
  qdrant_data:
```

```bash
GROQ_API_KEY=gsk_... docker compose up
```

### Production checklist

- Set `ENVIRONMENT=production` and `DEBUG=false`.
- Run behind a reverse proxy (nginx, Caddy) that enforces HTTPS and request-size limits.
- Consider rate-limiting the `/documents` and `/conversations/{id}/ask` endpoints at the proxy layer.
- The in-memory conversation store is per-process. Use a persistent store (Redis, PostgreSQL) for multi-worker or multi-replica deployments.

---

## Limitations and trade-offs

These are deliberate decisions given the assignment time limit:

**In-memory conversation store.** Conversations are stored in process memory. Restarting the server clears all history. A persistent store would require a database dependency not included in the assignment scope.

**Character-based chunking.** Chunks are split by character count, not by token count or semantic boundary. This is simpler and has zero extra dependencies. Token-aware splitting (e.g., `tiktoken`) would produce more consistent chunk sizes but adds complexity.

**Local embedding model.** Embeddings are generated locally using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). No external API key is needed for embeddings.

**No hallucination guarantee.** The system is designed to refuse when evidence is insufficient and to verify answers against retrieved chunks, but a sufficiently misleading knowledge base could still produce incorrect answers.

**No authentication.** The API has no authentication layer. Operator-level controls (API gateway, network isolation) are required before exposing this to untrusted clients.

**No streaming.** Responses are returned as complete JSON objects. Streaming LLM output would require a different response model.

**No document deletion.** There is no endpoint to remove a document from the vector store after ingestion.

**Single-process only.** Qdrant's `_get_client()` uses `@lru_cache`. Under multiple workers, each worker maintains its own connection.
