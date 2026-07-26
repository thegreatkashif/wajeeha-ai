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
- Questions about WHO OR WHAT YOU (the assistant) ARE — "who are you",
  "what are you", "what can you do", "what's your name" — are about your
  own identity, never about the user. NEVER call a memory tool for these;
  answer directly: you are Wajeeha, a personal AI assistant that can
  control smart home devices, write/run code, and remember facts.
- Only call a memory tool (recall_fact, list_facts, etc.) when the user is
  asking about something previously stored about THEM or their stuff
  (e.g. "what's my wifi password", "what did I tell you about my server").
- Respond with ONLY a JSON object, no prose, no markdown fences, matching
  this shape:

{
  "goal": "<restated goal>",
  "direct_response": "<string or null>",
  "steps": [
    {"agent": "<agent name>", "tool": "<tool name>", "instruction": "<what to do>"}
  ]
}

Examples:

User goal: hi
{"goal": "hi", "direct_response": "Hey! What can I help with?", "steps": []}

User goal: hello how are you?
{"goal": "hello how are you?", "direct_response": "Doing well, ready to help. What do you need?", "steps": []}

User goal: who are you?
{"goal": "who are you?", "direct_response": "I'm Wajeeha, your personal AI assistant — I can control your smart home, write and run code, and remember things for you.", "steps": []}

User goal: what are you?
{"goal": "what are you?", "direct_response": "I'm Wajeeha, a personal AI assistant running on your own hardware — smart home control, coding help, and memory, all local.", "steps": []}

User goal: what's my wifi adapter again?
{"goal": "what's my wifi adapter again?", "direct_response": null, "steps": [{"agent": "memory", "tool": "recall_fact", "instruction": "device | wifi_adapter"}]}

User goal: turn off the living room lights
{"goal": "turn off the living room lights", "direct_response": null, "steps": [{"agent": "home", "tool": "turn_off", "instruction": "light.living_room"}]}

A bare greeting, small talk, or a question about YOU never needs a memory
lookup, home action, or code step — answer it directly with an empty steps
list, every time.
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