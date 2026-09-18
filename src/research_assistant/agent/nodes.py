"""LangGraph node functions for the research engine.

Each node receives a ResearchState snapshot and returns a partial dict.
Nodes must not mutate the received state object.

Security:
- Retrieved document text is sanitised before it touches any LLM call.
- Conversation history is treated as untrusted user input.
- System prompts are never overridable by data in the user/evidence role.
- Chain-of-thought is never surfaced to callers.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_openai import ChatOpenAI

from research_assistant.agent.prompts import (
    DECOMPOSE_SYSTEM,
    DECOMPOSE_USER,
    EVALUATE_SYSTEM,
    EVALUATE_USER,
    REFINE_SYSTEM,
    REFINE_USER,
    SYNTHESIZE_SYSTEM,
    SYNTHESIZE_USER,
    UNDERSTAND_SYSTEM,
    UNDERSTAND_USER,
    VERIFY_SYSTEM,
    VERIFY_USER,
    REWRITE_SYSTEM,
    REWRITE_USER,
)
from research_assistant.agent.state import (
    EvidenceItem,
    QueryType,
    ResearchState,
    SubQuestionResult,
)
from research_assistant.config import settings
from research_assistant.vectorstore.client import similarity_search
from research_assistant.vectorstore.embedder import get_embeddings

logger = logging.getLogger(__name__)

_UNSUPPORTED_ANSWER = (
    "The available knowledge base does not contain sufficient information "
    "to answer this question."
)


def _llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance. Lazy so import never fails without a key."""
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.research_llm_temperature,
        openai_api_key=settings.openai_api_key.get_secret_value(),
    )


_MAX_EVIDENCE_CHARS = 1500   # per retrieved chunk; prevents runaway LLM context
_MAX_HISTORY_CHARS = 4000    # total history block injected into any LLM call

# Characters that break XML-style prompt delimiters or JSON parsing
_STRIP_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\ufff0-\uffff]")


def _sanitise(text: str) -> str:
    """Remove control characters and Unicode direction/invisible overrides.

    Does NOT strip < > or {{ }} because they are legitimate in document text;
    the prompt templates wrap all untrusted data in explicit XML delimiters
    so the LLM sees them as data, not structure.
    """
    return _STRIP_PATTERN.sub("", text)


def _cap(text: str, limit: int) -> str:
    """Truncate text to *limit* characters, appending a notice when cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + f" [truncated at {limit} chars]"


def _history_text(history: list[dict]) -> str:
    """Render conversation history as plain text, respecting config limits.

    Uses settings.max_history_turns so the behaviour matches what the API
    sliced before passing history into the graph.
    """
    if not history:
        return "(none)"
    # honour the configured turn limit — take the most-recent turns
    recent = history[-(settings.max_history_turns * 2):]
    lines = []
    for turn in recent:
        role = _sanitise(str(turn.get("role", "user")))
        content = _cap(_sanitise(str(turn.get("content", ""))), 800)
        lines.append(f"{role}: {content}")
    return _cap("\n".join(lines), _MAX_HISTORY_CHARS)


def _call(system: str, user: str) -> str:
    """Call the LLM with explicit system/user separation."""
    from langchain_core.messages import HumanMessage, SystemMessage
    response = _llm().invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return str(response.content).strip()


# ── Nodes ─────────────────────────────────────────────────────────────────────


def understand_query(state: ResearchState) -> dict:
    """Classify the query type."""
    raw = _sanitise(state["original_query"])
    history = _history_text(state.get("conversation_history", []))
    user_msg = UNDERSTAND_USER.format(history=history, query=raw)

    try:
        result = _call(UNDERSTAND_SYSTEM, user_msg)
        data = json.loads(result)
        query_type = QueryType(data.get("query_type", "standalone"))
    except Exception:
        logger.warning("understand_query parse failed; defaulting to standalone")
        query_type = QueryType.standalone

    logger.debug("understand_query: type=%s", query_type)
    return {"query_type": query_type}


def rewrite_query(state: ResearchState) -> dict:
    """Rewrite the query into a standalone, unambiguous form."""
    raw = _sanitise(state["original_query"])
    history = _history_text(state.get("conversation_history", []))
    user_msg = REWRITE_USER.format(history=history, query=raw)

    try:
        rewritten = _call(REWRITE_SYSTEM, user_msg)
    except Exception:
        logger.warning("rewrite_query failed; using original query")
        rewritten = raw

    logger.debug("rewrite_query: %r", rewritten)
    return {"rewritten_query": rewritten or raw}


def decompose_query(state: ResearchState) -> dict:
    """Split a compound query into focused sub-questions (capped at max_sub_questions)."""
    query = _sanitise(state.get("rewritten_query") or state["original_query"])
    user_msg = DECOMPOSE_USER.format(query=query)
    system = DECOMPOSE_SYSTEM.format(max_sub=settings.max_sub_questions)

    try:
        result = _call(system, user_msg)
        sub_questions: list[str] = json.loads(result)
        if not isinstance(sub_questions, list):
            raise ValueError("not a list")
        sub_questions = [str(q).strip() for q in sub_questions if str(q).strip()]
    except Exception:
        logger.warning("decompose_query parse failed; using single sub-question")
        sub_questions = [query]

    # Hard cap
    sub_questions = sub_questions[: settings.max_sub_questions]
    logger.debug("decompose_query: %d sub-questions", len(sub_questions))
    return {"sub_questions": sub_questions}


def retrieve_evidence(state: ResearchState) -> dict:
    """Retrieve evidence from Qdrant for each sub-question independently."""
    sub_questions: list[str] = state.get("sub_questions") or [
        state.get("rewritten_query") or state["original_query"]
    ]
    # Use refined query if set from a previous iteration
    refined = state.get("refined_query", "")
    if refined:
        sub_questions = [refined] + sub_questions[1:]

    results: list[SubQuestionResult] = []
    for sq in sub_questions:
        sq_clean = _sanitise(sq)
        try:
            vectors = get_embeddings([sq_clean])
            hits = similarity_search(
                query_vector=vectors[0],
                top_k=settings.evidence_top_k,
                score_threshold=settings.evidence_min_score,
            )
        except Exception:
            logger.exception("retrieve_evidence failed for sub_question=%r", sq)
            hits = []

        evidence: list[EvidenceItem] = []
        for hit in hits:
            if not hit.payload:
                continue
            raw_text = _sanitise(hit.payload.get("text", ""))
            evidence.append(
                EvidenceItem(
                    chunk_id=str(hit.id),
                    document_id=hit.payload.get("document_id", ""),
                    title=hit.payload.get("title", ""),
                    section=hit.payload.get("section", ""),
                    source=hit.payload.get("source", ""),
                    # Cap text so a single large chunk cannot fill the LLM context.
                    text=_cap(raw_text, _MAX_EVIDENCE_CHARS),
                    score=hit.score,
                    sub_question=sq,
                )
            )
        results.append(SubQuestionResult(sub_question=sq, evidence=evidence, supported=False))

    logger.debug("retrieve_evidence: %d sub-questions, total hits=%d",
                 len(results), sum(len(r["evidence"]) for r in results))
    return {"sub_question_results": results}


def evaluate_evidence(state: ResearchState) -> dict:
    """Ask the LLM whether each sub-question's evidence actually supports an answer."""
    results = state["sub_question_results"]
    updated: list[SubQuestionResult] = []
    any_supported = False

    for item in results:
        sq = item["sub_question"]
        evidence = item["evidence"]

        if not evidence:
            updated.append(SubQuestionResult(sub_question=sq, evidence=[], supported=False))
            continue

        excerpts = "\n\n".join(
            # Each evidence text was already sanitised and capped at retrieve time;
            # re-apply sanitise here as a defence-in-depth measure.
            f"[{_sanitise(e['source'])}] {_sanitise(e['text'])}" for e in evidence
        )
        user_msg = EVALUATE_USER.format(sub_question=_sanitise(sq), excerpts=excerpts)

        try:
            result = _call(EVALUATE_SYSTEM, user_msg)
            data = json.loads(result)
            supported = bool(data.get("supported", False))
        except Exception:
            logger.warning("evaluate_evidence parse failed for sub_question=%r", sq)
            supported = False

        if supported:
            any_supported = True
        updated.append(SubQuestionResult(sub_question=sq, evidence=evidence, supported=supported))

    logger.debug("evaluate_evidence: any_supported=%s", any_supported)
    return {
        "sub_question_results": updated,
        "evidence_sufficient": any_supported,
    }


def refine_query(state: ResearchState) -> dict:
    """Generate a refined search query targeting sub-questions with no support."""
    failed = [
        r["sub_question"]
        for r in state["sub_question_results"]
        if not r["supported"]
    ]
    original = _cap(_sanitise(state.get("rewritten_query") or state["original_query"]), 500)
    failed_text = "\n".join(f"- {_cap(_sanitise(q), 300)}" for q in failed)
    user_msg = REFINE_USER.format(query=original, failed=failed_text)

    try:
        refined = _call(REFINE_SYSTEM, user_msg)
    except Exception:
        logger.warning("refine_query failed; using original query")
        refined = original

    new_count = state.get("iteration_count", 0) + 1
    logger.debug("refine_query: iteration=%d refined=%r", new_count, refined)
    return {
        "refined_query": refined or original,
        "iteration_count": new_count,
    }


def synthesize_answer(state: ResearchState) -> dict:
    """Build an answer using only the retrieved evidence."""
    question = _sanitise(state.get("rewritten_query") or state["original_query"])
    all_evidence = [
        e for r in state["sub_question_results"] for e in r["evidence"]
    ]
    if not all_evidence:
        return {"answer": _UNSUPPORTED_ANSWER, "answer_supported": False, "citations": []}

    evidence_text = "\n\n".join(
        f"[{e['source']}] (section: {e['section'] or 'N/A'})\n{e['text']}"
        for e in all_evidence
    )
    user_msg = SYNTHESIZE_USER.format(question=question, evidence=evidence_text)

    try:
        answer = _call(SYNTHESIZE_SYSTEM, user_msg)
    except Exception:
        logger.exception("synthesize_answer failed")
        answer = _UNSUPPORTED_ANSWER

    citations = sorted({e["source"] for e in all_evidence if e["source"]})
    logger.debug("synthesize_answer: citations=%s", citations)
    return {"answer": answer, "answer_supported": True, "citations": citations}


def verify_answer(state: ResearchState) -> dict:
    """Verify the answer against the evidence; remove unsupported claims."""
    answer = state.get("answer", "")
    all_evidence = [
        e for r in state["sub_question_results"] for e in r["evidence"]
    ]

    if not all_evidence or not state.get("answer_supported", False):
        return {"answer": _UNSUPPORTED_ANSWER, "answer_supported": False}

    evidence_text = "\n\n".join(
        f"[{e['source']}]\n{e['text']}" for e in all_evidence
    )
    user_msg = VERIFY_USER.format(answer=answer, evidence=evidence_text)

    try:
        verified = _call(VERIFY_SYSTEM, user_msg)
    except Exception:
        logger.exception("verify_answer failed; returning previous answer")
        verified = answer

    # Detect when the LLM issued the unsupported signal
    unsupported_signal = "does not contain sufficient information"
    if unsupported_signal in verified.lower():
        return {"answer": _UNSUPPORTED_ANSWER, "answer_supported": False}

    logger.debug("verify_answer: complete")
    return {"answer": verified, "answer_supported": True}
