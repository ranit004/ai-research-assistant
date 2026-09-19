"""Tests for conversation management and POST /conversations/{id}/ask endpoints."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from research_assistant.conversations.store import conversation_store
from research_assistant.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_store():
    conversation_store.clear()
    yield
    conversation_store.clear()


def test_create_conversation():
    response = client.post("/conversations")
    assert response.status_code == 201
    data = response.json()
    assert "conversation_id" in data
    assert "created_at" in data
    assert len(data["conversation_id"]) > 0


def test_get_conversation_success():
    res = client.post("/conversations")
    cid = res.json()["conversation_id"]

    get_res = client.get(f"/conversations/{cid}")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["conversation_id"] == cid
    assert data["messages"] == []


def test_get_conversation_not_found():
    response = client.get("/conversations/non-existent-uuid")
    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."


def test_ask_question_not_found():
    response = client.post(
        "/conversations/non-existent-uuid/ask",
        json={"question": "What is Kubernetes?"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."


def test_ask_question_invalid_empty():
    res = client.post("/conversations")
    cid = res.json()["conversation_id"]

    response = client.post(f"/conversations/{cid}/ask", json={"question": "   "})
    assert response.status_code == 422


def test_ask_question_exceeds_max_length():
    res = client.post("/conversations")
    cid = res.json()["conversation_id"]

    long_q = "a" * 2001
    response = client.post(f"/conversations/{cid}/ask", json={"question": long_q})
    assert response.status_code == 422


@patch("research_assistant.api.conversations.research_graph")
def test_ask_question_single_turn_success(mock_graph: MagicMock):
    mock_graph.invoke.return_value = {
        "original_query": "What is a ReplicaSet?",
        "query_type": "standalone",
        "sub_questions": ["What is a ReplicaSet?"],
        "sub_question_results": [
            {
                "sub_question": "What is a ReplicaSet?",
                "supported": True,
                "evidence": [
                    {
                        "chunk_id": "chunk-101",
                        "document_id": "doc-1",
                        "title": "K8s Docs",
                        "section": "ReplicaSet",
                        "source": "k8s.txt",
                        "text": "A ReplicaSet maintains a stable set of replica Pods.",
                        "score": 0.9,
                    }
                ],
            }
        ],
        "iteration_count": 0,
        "answer": "A ReplicaSet ensures that a specified number of pod replicas are running.",
        "answer_supported": True,
        "citations": ["k8s.txt"],
    }

    res = client.post("/conversations")
    cid = res.json()["conversation_id"]

    ask_res = client.post(
        f"/conversations/{cid}/ask",
        json={"question": "What is a ReplicaSet?"},
    )
    assert ask_res.status_code == 200
    data = ask_res.json()
    assert data["conversation_id"] == cid
    assert data["answer"] == "A ReplicaSet ensures that a specified number of pod replicas are running."
    assert data["supported"] is True
    assert len(data["sources"]) == 1
    assert data["sources"][0]["document_id"] == "doc-1"
    assert data["sources"][0]["chunk_id"] == "chunk-101"
    assert data["metadata"]["sub_question_count"] == 1
    assert data["metadata"]["research_iteration_count"] == 0
    assert data["metadata"]["query_type"] == "standalone"

    # Check conversation message history stored
    get_res = client.get(f"/conversations/{cid}")
    msgs = get_res.json()["messages"]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "What is a ReplicaSet?"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "A ReplicaSet ensures that a specified number of pod replicas are running."


@patch("research_assistant.api.conversations.research_graph")
def test_ask_question_multi_turn_history(mock_graph: MagicMock):
    res = client.post("/conversations")
    cid = res.json()["conversation_id"]

    # First turn setup
    mock_graph.invoke.return_value = {
        "original_query": "What is a ReplicaSet?",
        "answer": "A ReplicaSet maintains pod replicas.",
        "answer_supported": True,
        "sub_questions": ["What is a ReplicaSet?"],
        "sub_question_results": [],
        "iteration_count": 0,
        "query_type": "standalone",
    }
    client.post(f"/conversations/{cid}/ask", json={"question": "What is a ReplicaSet?"})

    # Second turn setup
    mock_graph.invoke.return_value = {
        "original_query": "How does it differ from a Deployment?",
        "answer": "Deployments manage ReplicaSets automatically.",
        "answer_supported": True,
        "sub_questions": ["How does a Deployment differ from a ReplicaSet?"],
        "sub_question_results": [],
        "iteration_count": 0,
        "query_type": "follow_up",
    }
    ask2_res = client.post(
        f"/conversations/{cid}/ask",
        json={"question": "How does it differ from a Deployment?"},
    )
    assert ask2_res.status_code == 200

    # Verify conversation history was passed to graph on turn 2
    last_call_args = mock_graph.invoke.call_args[0][0]
    assert last_call_args["original_query"] == "How does it differ from a Deployment?"
    assert len(last_call_args["conversation_history"]) == 2
    assert last_call_args["conversation_history"][0]["role"] == "user"
    assert last_call_args["conversation_history"][0]["content"] == "What is a ReplicaSet?"
    assert last_call_args["conversation_history"][1]["role"] == "assistant"

    # Verify total history stored
    get_res = client.get(f"/conversations/{cid}")
    msgs = get_res.json()["messages"]
    assert len(msgs) == 4


def test_documents_upload_alias():
    with patch("research_assistant.api.ingest.validate_upload", return_value="text/plain"), \
         patch("research_assistant.api.ingest.extract_text", return_value="Document text content"), \
         patch("research_assistant.api.ingest.get_embeddings", return_value=[[0.1] * 1536]), \
         patch("research_assistant.api.ingest.ensure_collection"), \
         patch("research_assistant.api.ingest.upsert_chunks"):
        response = client.post(
            "/documents",
            files={"file": ("test.txt", b"Document text content", "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["chunk_count"] == 1
        assert data["title"] == "test"
