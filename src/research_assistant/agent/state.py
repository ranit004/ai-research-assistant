"""Strongly typed state for the research graph.

All fields are immutable between nodes — nodes return partial dicts and
LangGraph merges them.  No node may mutate the state object it receives.
"""

from __future__ import annotations

from enum import Enum
from typing import TypedDict


class QueryType(str, Enum):
    standalone = "standalone"    # self-contained, no context needed
    ambiguous = "ambiguous"      # under-specified or multi-meaning
    follow_up = "follow_up"      # refers to prior conversation turn
    compound = "compound"        # multiple distinct questions in one


class EvidenceItem(TypedDict):
    """One retrieved chunk with its retrieval score."""
    chunk_id: str
    document_id: str
    title: str
    section: str
    source: str
    text: str          # sanitised before it reaches the LLM
    score: float
    sub_question: str  # which sub-question retrieved this


class SubQuestionResult(TypedDict):
    sub_question: str
    evidence: list[EvidenceItem]
    supported: bool    # True only when evidence passes evaluation, not just score


class ResearchState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    original_query: str
    conversation_history: list[dict]   # [{"role": "user|assistant", "content": "..."}]

    # ── Query understanding ────────────────────────────────────────────────────
    query_type: QueryType
    rewritten_query: str

    # ── Decomposition ─────────────────────────────────────────────────────────
    sub_questions: list[str]

    # ── Retrieval + evaluation ─────────────────────────────────────────────────
    sub_question_results: list[SubQuestionResult]
    evidence_sufficient: bool

    # ── Refinement loop guard ──────────────────────────────────────────────────
    iteration_count: int
    refined_query: str

    # ── Output ─────────────────────────────────────────────────────────────────
    answer: str
    answer_supported: bool     # False when KB cannot support the answer
    citations: list[str]       # source filenames referenced
