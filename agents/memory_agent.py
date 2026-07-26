from __future__ import annotations

from agents.base_agent import BaseAgent
from memory.long_term import LongTermMemory


def _parse_parts(instruction: str, n: int) -> list[str] | None:
    """Split an instruction into exactly n parts. Small local models don't
    always follow the '<category> | <key>' format exactly, so accept a
    couple of reasonable variants before giving up:
      'user | name'   (preferred)
      'user.name'     (dot-separated)
      'user name'     (whitespace, only when n == 2 and there's no other separator)
    """
    text = instruction.strip()
    if "|" in text:
        parts = [p.strip() for p in text.split("|")]
    elif "." in text and text.count(".") == n - 1:
        parts = [p.strip() for p in text.split(".")]
    elif n == 2 and len(text.split()) == 2:
        parts = text.split()
    else:
        return None
    return parts if len(parts) == n else None


class MemoryAgent(BaseAgent):
    """Lets the assistant explicitly save, recall, and list durable facts
    about the user's world (devices, servers, preferences, projects,
    schedules) — the structured long-term memory store, as opposed to
    short-term chat history or fuzzy semantic search."""

    name = "memory"
    description = "Remembers and recalls durable facts (devices, servers, preferences, projects)."

    def __init__(self, long_term: LongTermMemory) -> None:
        self._long_term = long_term
        super().__init__()

    def register_tools(self) -> None:
        self.add_tool(
            "remember_fact",
            "Save a fact. Instruction: '<category> | <key> | <value>', "
            "e.g. 'device | wifi_adapter | TP-Link Archer T3U'.",
            self.remember_fact,
        )
        self.add_tool(
            "recall_fact",
            "Look up a fact. Instruction: '<category> | <key>', e.g. 'device | wifi_adapter'.",
            self.recall_fact,
        )
        self.add_tool(
            "list_facts",
            "List all facts in a category. Instruction: category name, e.g. 'device'.",
            self.list_facts,
        )
        self.add_tool(
            "forget_fact",
            "Delete a fact. Instruction: '<category> | <key>'.",
            self.forget_fact,
        )

    async def remember_fact(self, instruction: str) -> str:
        parts = _parse_parts(instruction, 3)
        if parts is None:
            return "remember_fact expects '<category> | <key> | <value>'"
        category, key, value = parts
        fact = self._long_term.upsert(category, key, value)
        return f"Remembered: {fact.category}.{fact.key} = {fact.value}"

    async def recall_fact(self, instruction: str) -> str:
        parts = _parse_parts(instruction, 2)
        if parts is None:
            return "recall_fact expects '<category> | <key>'"
        category, key = parts
        fact = self._long_term.get(category, key)
        return f"{fact.category}.{fact.key} = {fact.value}" if fact else "No such fact stored."

    async def list_facts(self, instruction: str) -> str:
        category = instruction.strip()
        facts = self._long_term.list_by_category(category)
        if not facts:
            return f"No facts stored under '{category}'."
        return "\n".join(f"- {f.key} = {f.value}" for f in facts)

    async def forget_fact(self, instruction: str) -> str:
        parts = _parse_parts(instruction, 2)
        if parts is None:
            return "forget_fact expects '<category> | <key>'"
        category, key = parts
        deleted = self._long_term.delete(category, key)
        return "Forgotten." if deleted else "No such fact was stored."