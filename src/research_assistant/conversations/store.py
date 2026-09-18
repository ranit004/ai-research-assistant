"""In-process conversation store with thread safety."""

from datetime import datetime, timezone
import threading
import uuid
from typing import TypedDict

from research_assistant.config import settings


class MessageDict(TypedDict):
    role: str
    content: str


class ConversationDict(TypedDict):
    conversation_id: str
    created_at: str
    messages: list[MessageDict]


class ConversationStore:
    """Thread-safe in-memory store for conversation sessions and messages."""

    def __init__(self) -> None:
        self._conversations: dict[str, ConversationDict] = {}
        self._lock = threading.Lock()

    def create_conversation(self) -> ConversationDict:
        cid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        conv: ConversationDict = {
            "conversation_id": cid,
            "created_at": now,
            "messages": [],
        }
        with self._lock:
            self._conversations[cid] = conv
        return self._get_snapshot(conv)

    def get_conversation(self, conversation_id: str) -> ConversationDict | None:
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return None
            return self._get_snapshot(conv)

    def add_message(self, conversation_id: str, role: str, content: str) -> bool:
        """Append a message, evicting oldest entries when the hard cap is reached.

        The hard cap is ``4 * max_history_turns`` messages (i.e. twice what
        ``get_history`` returns), giving a reasonable retention window without
        unlimited memory growth.
        """
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return False
            conv["messages"].append({"role": role, "content": content})
            cap = settings.max_history_turns * 4
            if len(conv["messages"]) > cap:
                # Drop the oldest messages to stay within the cap.
                conv["messages"] = conv["messages"][-cap:]
            return True

    def get_history(self, conversation_id: str, max_turns: int = 10) -> list[dict]:
        """Return the recent conversation turns (up to 2 * max_turns messages)."""
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return []
            messages = conv["messages"]
            limit = max_turns * 2
            return [dict(m) for m in messages[-limit:]]

    def _get_snapshot(self, conv: ConversationDict) -> ConversationDict:
        return {
            "conversation_id": conv["conversation_id"],
            "created_at": conv["created_at"],
            "messages": [dict(m) for m in conv["messages"]],
        }

    def clear(self) -> None:
        """Reset the store (used for tests)."""
        with self._lock:
            self._conversations.clear()


conversation_store = ConversationStore()
