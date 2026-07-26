from __future__ import annotations

import logging
from typing import Awaitable, Callable

from agents.base_agent import BaseAgent
from brain.llm_router import LLMRouter
from brain.planner import Plan, Planner
from memory.long_term import LongTermMemory
from memory.semantic import SemanticMemory
from memory.short_term import ShortTermMemory

logger = logging.getLogger("wajeeha.orchestrator")

ConfirmFn = Callable[[str], Awaitable[bool]]


class Orchestrator:
    """The brain's execution loop: take a user goal, ask the Planner to
    decompose it against the currently registered agents, run each step
    against the right agent/tool, reflect on failures and retry, and write
    the outcome back into memory.

    This is intentionally the only place that both (a) knows about every
    agent and (b) is allowed to execute plan steps — agents don't call each
    other directly, which keeps the "who did what" trail auditable in the
    logs and in long-term memory.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        semantic: SemanticMemory,
        agents: list[BaseAgent],
        max_retries_per_step: int = 2,
    ) -> None:
        self._llm = llm_router
        self._planner = Planner(llm_router)
        self._short_term = short_term
        self._long_term = long_term
        self._semantic = semantic
        self._agents: dict[str, BaseAgent] = {a.name: a for a in agents}
        self._max_retries = max_retries_per_step

    def _describe_all_tools(self) -> str:
        return "\n".join(agent.describe_tools() for agent in self._agents.values())

    def _gather_memory_context(self, goal: str) -> str:
        hits = self._semantic.search(goal, n_results=3)
        if not hits:
            return ""
        return "\n".join(f"- {h.text} (score={h.score:.2f})" for h in hits)

    async def handle_goal(self, goal: str) -> str:
        self._short_term.add_user(goal)
        memory_context = self._gather_memory_context(goal)

        plan = await self._planner.create_plan(
            goal=goal,
            available_tools_description=self._describe_all_tools(),
            memory_context=memory_context,
        )
        logger.info("Plan for goal %r: %s", goal, plan.model_dump())

        if not plan.steps:
            answer = plan.direct_response or (
                "Hi! I'm here — ask me to do something (control a device, "
                "look something up, work on code) or just chat."
            )
            self._short_term.add_assistant(answer)
            return answer

        step_summary = await self._execute_plan(plan)
        # If the planner already gave a conversational answer, that's what
        # the user sees — the steps ran for their side effects (saving a
        # fact, flipping a light, etc.) but a tool error on a supporting
        # step shouldn't override a perfectly good direct answer.
        final_summary = plan.direct_response or step_summary
        self._short_term.add_assistant(final_summary)
        self._semantic.add(f"Goal: {goal}\nOutcome: {final_summary}", metadata={"kind": "goal_log"})
        return final_summary

    async def _execute_plan(self, plan: Plan) -> str:
        results: list[str] = []
        for step in plan.steps:
            agent = self._agents.get(step.agent)
            if agent is None:
                step.error = f"Unknown agent '{step.agent}'"
                results.append(f"[{step.agent}.{step.tool}] FAILED: {step.error}")
                continue

            for attempt in range(1, self._max_retries + 2):
                step.attempts = attempt
                try:
                    step.result = await agent.run_tool(step.tool, step.instruction)
                    results.append(f"[{step.agent}.{step.tool}] {step.result}")
                    break
                except Exception as exc:  # noqa: BLE001 — deliberately broad: any
                    # tool failure should be caught, logged, and retried/reported
                    # rather than crashing the whole plan.
                    step.error = str(exc)
                    logger.warning(
                        "Step %s.%s failed (attempt %d): %s",
                        step.agent, step.tool, attempt, exc,
                    )
                    if attempt > self._max_retries:
                        results.append(
                            f"[{step.agent}.{step.tool}] FAILED after {attempt} attempts: {exc}"
                        )

        return "\n".join(results)