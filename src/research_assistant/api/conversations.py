"""Conversation management and multi-turn research endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from research_assistant.agent.graph import research_graph
from research_assistant.agent.state import ResearchState
from research_assistant.config import settings
from research_assistant.conversations.store import conversation_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])

_UNSUPPORTED_ANSWER = (
    "The available knowledge base does not contain sufficient information "
    "to answer this question."
)


class CreateConversationResponse(BaseModel):
    conversation_id: str
    created_at: str


class ConversationMessage(BaseModel):
    role: str
    content: str


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    created_at: str
    messages: list[ConversationMessage]


class AskRequest(BaseModel):
    question: str = Field(..., description="User research question.")

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Question must be a string.")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or whitespace only.")
        if len(cleaned) > settings.max_question_length:
            raise ValueError(
                f"Question exceeds maximum length of {settings.max_question_length} characters."
            )
        return cleaned


class SourceMetadata(BaseModel):
    document_id: str
    title: str
    section: str
    source: str
    chunk_id: str


class ResearchMetadata(BaseModel):
    sub_question_count: int
    research_iteration_count: int
    query_type: str


class AskResponse(BaseModel):
    answer: str
    supported: bool
    sources: list[SourceMetadata]
    conversation_id: str
    metadata: ResearchMetadata


@router.post("", response_model=CreateConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation() -> CreateConversationResponse:
    """Create a new conversation session."""
    conv = conversation_store.create_conversation()
    return CreateConversationResponse(
        conversation_id=conv["conversation_id"],
        created_at=conv["created_at"],
    )


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(conversation_id: str) -> ConversationDetailResponse:
    """Retrieve conversation details and message history."""
    conv = conversation_store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )
    return ConversationDetailResponse(
        conversation_id=conv["conversation_id"],
        created_at=conv["created_at"],
        messages=[ConversationMessage(**m) for m in conv["messages"]],
    )


@router.post("/{conversation_id}/ask", response_model=AskResponse)
def ask_question(conversation_id: str, request: AskRequest) -> AskResponse:
    """Ask a question within an existing conversation using the multi-step research engine."""
    conv = conversation_store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    # 1. Load bounded conversation history
    history = conversation_store.get_history(
        conversation_id=conversation_id,
        max_turns=settings.max_history_turns,
    )

    # 2. Build initial state for research graph
    initial: ResearchState = {
        "original_query": request.question,
        "conversation_history": history,
        "query_type": "standalone",  # type: ignore[assignment]
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

    # 3. Invoke research workflow
    try:
        final: ResearchState = research_graph.invoke(initial)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("Research graph failed for conversation_id=%s query=%r", conversation_id, request.question[:80])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Research engine error. Please try again later.",
        )

    answer = final.get("answer") or _UNSUPPORTED_ANSWER
    supported = bool(final.get("answer_supported", False))

    # 4. Extract detailed source metadata
    sources: list[SourceMetadata] = []
    seen_chunk_ids: set[str] = set()

    sub_results = final.get("sub_question_results") or []
    for res in sub_results:
        for ev in res.get("evidence", []):
            chunk_id = ev.get("chunk_id", "")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                sources.append(
                    SourceMetadata(
                        document_id=str(ev.get("document_id", "")),
                        title=str(ev.get("title", "")),
                        section=str(ev.get("section", "")),
                        source=str(ev.get("source", "")),
                        chunk_id=chunk_id,
                    )
                )

    # 5. Persist turn to conversation store
    conversation_store.add_message(conversation_id, "user", request.question)
    conversation_store.add_message(conversation_id, "assistant", answer)

    # 6. Build response metadata
    sub_questions = final.get("sub_questions") or []
    sub_count = len(sub_questions) if sub_questions else 1
    iteration_count = final.get("iteration_count", 0)
    query_type_val = final.get("query_type", "standalone")
    # FIX 6: QueryType is a str Enum, so its str() is "QueryType.standalone".
    # Use .value to get the clean string "standalone".
    if hasattr(query_type_val, "value"):
        query_type_str = query_type_val.value
    else:
        query_type_str = str(query_type_val)


    return AskResponse(
        answer=answer,
        supported=supported,
        sources=sources,
        conversation_id=conversation_id,
        metadata=ResearchMetadata(
            sub_question_count=sub_count,
            research_iteration_count=iteration_count,
            query_type=query_type_str,
        ),
    )
