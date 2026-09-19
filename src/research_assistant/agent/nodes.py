"""LangGraph node functions for the research engine.

Each node receives a ResearchState snapshot and returns a partial dict.
Nodes must not mutate the received state object.

Security:
- Retrieved document text is sanitised before it touches any LLM call.
- Conversation history is treated as untrusted user input.
- System prompts are never overridable by data in the user/evidence role.
- Chain-of-thought is never surfaced to callers.
- Delimiter characters (< >) are escaped in retrieved evidence so a malicious
  document cannot break out of its <evidence> block.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_groq import ChatGroq

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


def _llm() -> ChatGroq:
    """Return a ChatGroq instance. Lazy so import never fails without a key."""
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    return ChatGroq(
        model=settings.groq_model,
        temperature=settings.research_llm_temperature,
        groq_api_key=settings.groq_api_key.get_secret_value(),
    )


_MAX_EVIDENCE_CHARS = 1500   # per retrieved chunk; prevents runaway LLM context
_MAX_HISTORY_CHARS = 4000    # total history block injected into any LLM call

# Characters that break XML-style prompt delimiters or JSON parsing
_STRIP_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\ufff0-\uffff]")


def _sanitise(text: str) -> str:
    """Remove control characters and Unicode direction/invisible overrides."""
    return _STRIP_PATTERN.sub("", text)


def _escape_delimiters(text: str) -> str:
    """Escape < and > so untrusted text cannot break XML-style prompt delimiters.

    FIX 5: A malicious document could contain ``</evidence>`` followed by
    instruction text.  Escaping makes the raw characters visible to the LLM
    as document data, not as markup structure.
    """
    return text.replace("<", "&lt;").replace(">", "&gt;")


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
    """Call the LLM with explicit system/user separation.

    FIX 1: This function does NOT catch exceptions — Groq/API/network failures
    propagate to the node caller which decides whether to treat the failure as
    a retryable parse error or a hard API failure.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    response = _llm().invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return str(response.content).strip()


# ── Nodes ─────────────────────────────────────────────────────────────────────


def understand_query(state: ResearchState) -> dict:
    """Classify the query type.

    FIX 1: Only JSON parse/validation errors are caught and defaulted; actual
    LLM/API failures propagate so the API can return 503.
    """
    raw = _sanitise(state["original_query"])
    history = _history_text(state.get("conversation_history", []))
    user_msg = UNDERSTAND_USER.format(history=history, query=raw)

    result = _call(UNDERSTAND_SYSTEM, user_msg)
    try:
        data = json.loads(result)
        query_type = QueryType(data.get("query_type", "standalone"))
    except (json.JSONDecodeError, ValueError, KeyError):
        logger.warning("understand_query parse failed; defaulting to standalone")
        query_type = QueryType.standalone

    logger.debug("understand_query: type=%s", query_type)
    return {"query_type": query_type}


def rewrite_query(state: ResearchState) -> dict:
    """Rewrite the query into a standalone, unambiguous form.

    FIX 1: LLM failures propagate; only the no-output edge case is defaulted.
    """
    raw = _sanitise(state["original_query"])
    history = _history_text(state.get("conversation_history", []))
    user_msg = REWRITE_USER.format(history=history, query=raw)

    rewritten = _call(REWRITE_SYSTEM, user_msg)
    # Edge case: LLM returned an empty string — fall back to the original
    if not rewritten:
        logger.warning("rewrite_query returned empty string; using original query")
        rewritten = raw

    logger.debug("rewrite_query: %r", rewritten)
    return {"rewritten_query": rewritten}


def decompose_query(state: ResearchState) -> dict:
    """Split a compound query into focused sub-questions (capped at max_sub_questions).

    FIX 1: Only JSON parse/validation errors are caught; LLM failures propagate.
    """
    query = _sanitise(state.get("rewritten_query") or state["original_query"])
    user_msg = DECOMPOSE_USER.format(query=query)
    system = DECOMPOSE_SYSTEM.format(max_sub=settings.max_sub_questions)

    result = _call(system, user_msg)
    try:
        sub_questions: list[str] = json.loads(result)
        if not isinstance(sub_questions, list):
            raise ValueError("not a list")
        sub_questions = [str(q).strip() for q in sub_questions if str(q).strip()]
        if not sub_questions:
            raise ValueError("empty list")
    except (json.JSONDecodeError, ValueError):
        logger.warning("decompose_query parse failed; using single sub-question")
        sub_questions = [query]

    # Hard cap
    sub_questions = sub_questions[: settings.max_sub_questions]
    logger.debug("decompose_query: %d sub-questions", len(sub_questions))
    return {"sub_questions": sub_questions}


def retrieve_evidence(state: ResearchState) -> dict:
    """Retrieve evidence from Qdrant for each sub-question independently.

    FIX 3: When a refined_query is present, only replace evidence for the
    sub-questions that previously had no support; already-supported results are
    preserved unchanged.
    """
    sub_questions: list[str] = state.get("sub_questions") or [
        state.get("rewritten_query") or state["original_query"]
    ]

    # FIX 3: Carry forward already-supported results from a prior iteration.
    prior_results: list[SubQuestionResult] = state.get("sub_question_results") or []
    prior_by_sq = {r["sub_question"]: r for r in prior_results}

    refined = state.get("refined_query", "")

    results: list[SubQuestionResult] = []
    for sq in sub_questions:
        # If this sub-question was already supported, keep its evidence as-is.
        prior = prior_by_sq.get(sq)
        if prior and prior.get("supported"):
            results.append(prior)
            continue

        # Use the refined query only for sub-questions that previously failed.
        search_query = refined if refined else sq
        sq_clean = _sanitise(search_query)
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
    """Ask the LLM whether each sub-question's evidence actually supports an answer.

    FIX 1: Only JSON parse/validation errors are caught; LLM failures propagate.
    FIX 2: Evidence sufficiency is tracked per sub-question; all sub-questions
           must be supported for overall evidence_sufficient to be True.
    FIX 5: Evidence text is delimiter-escaped before insertion into the prompt.
    """
    results = state["sub_question_results"]
    updated: list[SubQuestionResult] = []
    all_supported = True

    for item in results:
        sq = item["sub_question"]
        evidence = item["evidence"]

        # FIX 3: Already-supported items from a prior iteration keep their status.
        if item.get("supported") and evidence:
            updated.append(item)
            continue

        if not evidence:
            all_supported = False
            updated.append(SubQuestionResult(sub_question=sq, evidence=[], supported=False))
            continue

        # FIX 5: Escape delimiters before building the excerpts block.
        excerpts = "\n\n".join(
            f"[{_sanitise(_escape_delimiters(e['source']))}] "
            f"{_sanitise(_escape_delimiters(e['text']))}"
            for e in evidence
        )
        user_msg = EVALUATE_USER.format(sub_question=_sanitise(sq), excerpts=excerpts)

        result = _call(EVALUATE_SYSTEM, user_msg)
        try:
            data = json.loads(result)
            supported = bool(data.get("supported", False))
        except (json.JSONDecodeError, ValueError, KeyError):
            logger.warning("evaluate_evidence parse failed for sub_question=%r", sq)
            supported = False

        if not supported:
            all_supported = False
        updated.append(SubQuestionResult(sub_question=sq, evidence=evidence, supported=supported))

    # FIX 2: evidence_sufficient is only True when every sub-question is supported.
    evidence_sufficient = all_supported and len(updated) > 0

    logger.debug("evaluate_evidence: evidence_sufficient=%s", evidence_sufficient)
    return {
        "sub_question_results": updated,
        "evidence_sufficient": evidence_sufficient,
    }


def refine_query(state: ResearchState) -> dict:
    """Generate a refined search query targeting sub-questions with no support.

    FIX 1: LLM failures propagate; only the empty-output edge case is defaulted.
    FIX 3: The refined query is built from the *specific* failed sub-questions,
           not from the overall query, so it targets the right gap.
    """
    failed = [
        r["sub_question"]
        for r in state["sub_question_results"]
        if not r["supported"]
    ]
    original = _cap(_sanitise(state.get("rewritten_query") or state["original_query"]), 500)
    failed_text = "\n".join(f"- {_cap(_sanitise(q), 300)}" for q in failed)
    user_msg = REFINE_USER.format(query=original, failed=failed_text)

    refined = _call(REFINE_SYSTEM, user_msg)
    if not refined:
        logger.warning("refine_query returned empty string; using original query")
        refined = original

    new_count = state.get("iteration_count", 0) + 1
    logger.debug("refine_query: iteration=%d refined=%r", new_count, refined)
    return {
        "refined_query": refined,
        "iteration_count": new_count,
    }


def synthesize_answer(state: ResearchState) -> dict:
    """Build an answer using only the retrieved evidence.

    FIX 1: LLM failures propagate; the empty-evidence short-circuit is preserved.
    FIX 5: Evidence text is delimiter-escaped before insertion into the prompt.
    """
    question = _sanitise(state.get("rewritten_query") or state["original_query"])
    all_evidence = [
        e for r in state["sub_question_results"] for e in r["evidence"]
    ]
    if not all_evidence:
        return {"answer": _UNSUPPORTED_ANSWER, "answer_supported": False, "citations": []}

    # FIX 5: Escape delimiter characters so document content cannot break out.
    evidence_text = "\n\n".join(
        f"[{_escape_delimiters(e['source'])}] "
        f"(section: {_escape_delimiters(e['section'] or 'N/A')})\n"
        f"{_escape_delimiters(e['text'])}"
        for e in all_evidence
    )
    user_msg = SYNTHESIZE_USER.format(question=question, evidence=evidence_text)

    answer = _call(SYNTHESIZE_SYSTEM, user_msg)

    citations = sorted({e["source"] for e in all_evidence if e["source"]})
    logger.debug("synthesize_answer: citations=%s", citations)
    return {"answer": answer, "answer_supported": True, "citations": citations}


def verify_answer(state: ResearchState) -> dict:
    """Verify the answer against the evidence; remove unsupported claims.

    FIX 1: LLM failures propagate; the empty-evidence short-circuit is preserved.
    FIX 5: Evidence text is delimiter-escaped before insertion into the prompt.
    """
    answer = state.get("answer", "")
    all_evidence = [
        e for r in state["sub_question_results"] for e in r["evidence"]
    ]

    if not all_evidence or not state.get("answer_supported", False):
        return {"answer": _UNSUPPORTED_ANSWER, "answer_supported": False}

    # FIX 5: Escape delimiter characters in evidence.
    evidence_text = "\n\n".join(
        f"[{_escape_delimiters(e['source'])}]\n{_escape_delimiters(e['text'])}"
        for e in all_evidence
    )
    user_msg = VERIFY_USER.format(answer=answer, evidence=evidence_text)

    verified = _call(VERIFY_SYSTEM, user_msg)

    # Detect when the LLM issued the unsupported signal
    unsupported_signal = "does not contain sufficient information"
    if unsupported_signal in verified.lower():
        return {"answer": _UNSUPPORTED_ANSWER, "answer_supported": False}

    logger.debug("verify_answer: complete")
    return {"answer": verified, "answer_supported": True}
