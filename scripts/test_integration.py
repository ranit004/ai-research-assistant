"""Integration tests for Groq + local embedding migration.

Run with:  uv run python scripts/test_integration.py

Requires:
  - GROQ_API_KEY set in .env
  - Qdrant running at QDRANT_URL
"""

from __future__ import annotations

import sys
import time

# ── 1. Embedding generation test (local, no API) ──────────────────────────────

print("=" * 60)
print("Test 1: Local embedding generation (sentence-transformers)")
print("=" * 60)
try:
    from research_assistant.vectorstore.embedder import get_embeddings

    texts = ["What is a Kubernetes pod?", "How does a ReplicaSet work?"]
    vecs = get_embeddings(texts)
    assert len(vecs) == 2, f"Expected 2 vectors, got {len(vecs)}"
    assert len(vecs[0]) == 384, f"Expected dim=384, got {len(vecs[0])}"
    print(f"  PASS: Generated {len(vecs)} vectors of dim {len(vecs[0])}")
    print(f"  Sample: vecs[0][:5] = {vecs[0][:5]}")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ── 2. Groq connectivity test ─────────────────────────────────────────────────

print()
print("=" * 60)
print("Test 2: Groq LLM connectivity")
print("=" * 60)
try:
    from research_assistant.config import settings

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment / .env")

    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatGroq(
        model=settings.groq_model,
        temperature=0.0,
        groq_api_key=settings.groq_api_key.get_secret_value(),
    )
    response = llm.invoke([
        SystemMessage(content="Reply with exactly one sentence."),
        HumanMessage(content="What is 2 + 2?"),
    ])
    answer = str(response.content).strip()
    print(f"  PASS: Groq responded: {answer!r}")
    assert answer, "Empty response from Groq"
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ── 3. Qdrant retrieval test ──────────────────────────────────────────────────

print()
print("=" * 60)
print("Test 3: Qdrant retrieval (embedding-to-search)")
print("=" * 60)
try:
    from research_assistant.config import settings
    from research_assistant.vectorstore.client import ensure_collection, similarity_search
    from research_assistant.vectorstore.embedder import get_embeddings

    ensure_collection()
    query_vec = get_embeddings(["test query"])[0]
    hits = similarity_search(query_vec, top_k=3, score_threshold=0.0)
    print(f"  PASS: Qdrant returned {len(hits)} hit(s) (may be 0 if collection is empty)")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ── 4. End-to-end research request (graph only, no HTTP) ─────────────────────

print()
print("=" * 60)
print("Test 4: End-to-end research graph (requires GROQ_API_KEY + Qdrant)")
print("=" * 60)
try:
    from research_assistant.agent.graph import research_graph
    from research_assistant.agent.state import QueryType

    initial = {
        "original_query": "What is a Kubernetes pod?",
        "conversation_history": [],
        "query_type": QueryType.standalone,
        "rewritten_query": "",
        "sub_questions": [],
        "sub_question_results": [],
        "evidence_sufficient": False,
        "iteration_count": 0,
        "refined_query": "",
        "answer": "",
        "answer_supported": False,
        "citations": [],
    }

    t0 = time.perf_counter()
    final = research_graph.invoke(initial)
    elapsed = time.perf_counter() - t0

    print(f"  PASS: Research graph completed in {elapsed:.2f}s")
    print(f"  Query type: {final.get('query_type')}")
    print(f"  Sub-questions: {final.get('sub_questions')}")
    print(f"  Supported: {final.get('answer_supported')}")
    print(f"  Answer: {str(final.get('answer', ''))[:200]!r}")
    if final.get("citations"):
        print(f"  Citations: {final['citations']}")
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("All integration tests passed.")
print("=" * 60)
