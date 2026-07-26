from __future__ import annotations

from collections import deque

from memory.models import ConversationTurn, Role


class ShortTermMemory:
    """Rolling window of the current conversation. Not persisted across
    process restarts by design — that's what long-term/semantic memory are
    for. Keeping this in-process keeps latency low for the common case of
    "what did I just say"."""

    def __init__(self, max_turns: int = 40) -> None:
        self._max_turns = max_turns
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)

    def add(self, role: Role, content: str, agent: str | None = None) -> None:
        self._turns.append(ConversationTurn(role=role, content=content, agent=agent))

    def add_user(self, content: str) -> None:
        self.add(Role.USER, content)

    def add_assistant(self, content: str, agent: str | None = None) -> None:
        self.add(Role.ASSISTANT, content, agent=agent)

    def as_list(self) -> list[ConversationTurn]:
        return list(self._turns)

    def as_llm_messages(self, limit: int | None = None) -> list[dict]:
        """Format for a standard chat-completions-style messages array.
        `limit`, if given, returns only the most recent `limit` turns."""
        turns = list(self._turns)[-limit:] if limit else list(self._turns)
        return [{"role": t.role.value, "content": t.content} for t in turns]

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)