from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    # A tool handler takes a single freeform instruction string (the
    # planner writes plain-English instructions per step, not structured
    # args, to keep the planner prompt simple) and returns a result string.
    handler: Callable[[str], Awaitable[str]]
    # Destructive tools require explicit confirmation before running,
    # regardless of what the planner or config say.
    destructive: bool = False


class BaseAgent(ABC):
    name: str
    description: str

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.register_tools()

    @abstractmethod
    def register_tools(self) -> None:
        """Populate self._tools via self.add_tool(...)."""

    def add_tool(
        self,
        name: str,
        description: str,
        handler: Callable[[str], Awaitable[str]],
        destructive: bool = False,
    ) -> None:
        self._tools[name] = ToolSpec(name, description, handler, destructive)

    @property
    def tools(self) -> dict[str, ToolSpec]:
        return self._tools

    def describe_tools(self) -> str:
        lines = [f"Agent: {self.name} — {self.description}"]
        for tool in self._tools.values():
            flag = " [DESTRUCTIVE — requires confirmation]" if tool.destructive else ""
            lines.append(f"  - {tool.name}: {tool.description}{flag}")
        return "\n".join(lines)

    async def run_tool(self, tool_name: str, instruction: str) -> str:
        if tool_name not in self._tools:
            raise KeyError(
                f"Agent '{self.name}' has no tool '{tool_name}'. "
                f"Available: {list(self._tools)}"
            )
        return await self._tools[tool_name].handler(instruction)