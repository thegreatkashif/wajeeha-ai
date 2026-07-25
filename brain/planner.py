from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from brain.llm_router import LLMRouter

PLANNER_SYSTEM_PROMPT = """You are the planning module of an autonomous assistant
called Wajeeha. Given a user goal, the list of available agents (with their
tools), and relevant memory, produce a short step-by-step plan.

Rules:
- Break the goal into the smallest number of concrete steps needed.
- Each step must name exactly one agent and one tool that agent exposes.
- Never invent a tool or agent that wasn't listed.
- If the goal is already answerable with no tool (pure conversation),
  return an empty steps list and put the answer in "direct_response".
- Respond with ONLY a JSON object, no prose, no markdown fences, matching
  this shape:

{
  "goal": "<restated goal>",
  "direct_response": "<string or null>",
  "steps": [
    {"agent": "<agent name>", "tool": "<tool name>", "instruction": "<what to do>"}
  ]
}
"""


class PlanStep(BaseModel):
    agent: str
    tool: str
    instruction: str
    result: str | None = None
    error: str | None = None
    attempts: int = 0


class Plan(BaseModel):
    goal: str
    direct_response: str | None = None
    steps: list[PlanStep] = Field(default_factory=list)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


class Planner:
    """Turns a user goal into a Plan by asking the configured LLM to
    decompose it against the tools we tell it are available. Deliberately
    does not execute anything — that's the Orchestrator's job — so the
    plan itself can be logged, reviewed, or (for destructive steps)
    confirmed by the user before a single tool runs.
    """

    def __init__(self, llm_router: LLMRouter, max_plan_steps: int = 12) -> None:
        self._llm = llm_router
        self._max_plan_steps = max_plan_steps

    async def create_plan(
        self,
        goal: str,
        available_tools_description: str,
        memory_context: str = "",
        provider: str | None = None,
    ) -> Plan:
        user_content = (
            f"Available agents and tools:\n{available_tools_description}\n\n"
            f"Relevant memory:\n{memory_context or '(none)'}\n\n"
            f"User goal: {goal}"
        )
        resp = await self._llm.complete(
            messages=[{"role": "user", "content": user_content}],
            system=PLANNER_SYSTEM_PROMPT,
            provider=provider,
        )
        cleaned = _strip_code_fences(resp.text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Planner LLM did not return valid JSON. Raw output:\n{resp.text}"
            ) from exc

        plan = Plan(**data)
        if len(plan.steps) > self._max_plan_steps:
            plan.steps = plan.steps[: self._max_plan_steps]
        return plan