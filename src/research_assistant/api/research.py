"""POST /research — run the multi-step LangGraph research engine."""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from research_assistant.agent.graph import research_graph
from research_assistant.agent.state import ResearchState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])

_UNSUPPORTED_ANSWER = (
    "The available knowledge base does not contain sufficient information "
    "to answer this question."
)


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="Research question.")
    conversation_history: list[dict] = Field(
        default_factory=list,
        max_length=20,
        description="Prior turns: [{\"role\": \"user|assistant\", \"content\": \"...\"}]",
    )


class ResearchResponse(BaseModel):
    answer: str
    answer_supported: bool
    citations: list[str]
    query_type: str
    sub_questions: list[str]
    iterations: int


@router.post("", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    """Run the multi-step research workflow and return a verified answer."""
    initial: ResearchState = {
        "original_query": request.query,
        "conversation_history": request.conversation_history,
        "query_type": "standalone",   # type: ignore[assignment]
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

    try:
        final: ResearchState = research_graph.invoke(initial)
    except RuntimeError as exc:
        # Missing API key
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception:
        logger.exception("research graph failed for query=%r", request.query[:80])
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Research engine error. Please try again later.",
        )

    return ResearchResponse(
        answer=final.get("answer") or _UNSUPPORTED_ANSWER,
        answer_supported=final.get("answer_supported", False),
        citations=final.get("citations") or [],
        query_type=str(final.get("query_type", "standalone")),
        sub_questions=final.get("sub_questions") or [],
        iterations=final.get("iteration_count", 0),
    )
