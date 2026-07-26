from __future__ import annotations

import asyncio
import logging

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Confirm

from agents.coding_agent import CodingAgent
from agents.home_agent import HomeAgent
from brain.llm_router import LLMRouter
from brain.orchestrator import Orchestrator
from config.settings import ensure_runtime_dirs, get_config, get_secrets
from memory.long_term import LongTermMemory
from memory.semantic import SemanticMemory
from memory.short_term import ShortTermMemory
from agents.memory_agent import MemoryAgent

app = typer.Typer(add_completion=False)
console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


async def _confirm(prompt: str) -> bool:
    # Runs the blocking Rich prompt in a thread so it doesn't stall the
    # event loop for other async work.
    return await asyncio.to_thread(Confirm.ask, f"[yellow]{prompt}[/yellow]")


async def _build_orchestrator() -> Orchestrator:
    config = get_config()
    secrets = get_secrets()
    ensure_runtime_dirs(config)
    _setup_logging(config.logging.level)

    llm_router = LLMRouter(config, secrets)
    short_term = ShortTermMemory(max_turns=config.memory.short_term.max_turns)
    long_term = LongTermMemory(db_path=config.memory.long_term.db_path)
    semantic = SemanticMemory(
        chroma_path=config.memory.semantic.chroma_path,
        collection_name=config.memory.semantic.collection_name,
        embedding_model=config.memory.semantic.embedding_model,
    )

    agents = []
    
    agents.append(MemoryAgent(long_term))

    if config.agents.coding.enabled:
        agents.append(
            CodingAgent(
                workspace_root=config.agents.coding.workspace_root,
                safe_shell_commands=config.agents.coding.safe_shell_commands,
                require_confirmation_for_writes=config.agents.coding.require_confirmation_for_writes,
                confirm_fn=_confirm,
            )
        )

    if config.agents.home.enabled:
        if not secrets.home_assistant_token:
            console.print(
                "[yellow]HOME_ASSISTANT_TOKEN not set — home agent disabled for this session.[/yellow]"
            )
        else:
            agents.append(
                HomeAgent(
                    base_url=config.agents.home.home_assistant.base_url,
                    token=secrets.home_assistant_token,
                    default_room=config.agents.home.default_room,
                )
            )

    return Orchestrator(
        llm_router=llm_router,
        short_term=short_term,
        long_term=long_term,
        semantic=semantic,
        agents=agents,
        max_retries_per_step=config.planner.max_retries_per_step,
    )


@app.command()
def chat() -> None:
    """Start an interactive session with Wajeeha."""

    async def _run() -> None:
        orchestrator = await _build_orchestrator()
        console.print("[bold cyan]Wajeeha AI[/bold cyan] — type 'exit' to quit.\n")
        while True:
            goal = console.input("[bold green]you>[/bold green] ").strip()
            if goal.lower() in {"exit", "quit"}:
                break
            if not goal:
                continue
            console.print("[cyan]thinking...[/cyan]")
            answer = await orchestrator.handle_goal(goal)
            console.print(f"[bold magenta]wajeeha>[/bold magenta] {answer}\n")

    asyncio.run(_run())


@app.command()
def run(goal: str) -> None:
    """Run a single goal non-interactively, e.g.:
    python cli.py run "turn off the living room lights"
    """

    async def _run() -> None:
        orchestrator = await _build_orchestrator()
        answer = await orchestrator.handle_goal(goal)
        console.print(answer)

    asyncio.run(_run())


@app.command()
def speak(text: str, out: str = "output.wav") -> None:
    """Synthesize `text` in the cloned voice (needs voice.enabled: true and
    reference clips in voice.reference_samples_dir), e.g.:
    python cli.py speak "Good morning, lights are on in the living room."
    """
    config = get_config()
    if not config.voice.enabled:
        console.print(
            "[yellow]Voice is disabled — set voice.enabled: true in "
            "config.yaml and add reference clips first.[/yellow]"
        )
        raise typer.Exit(1)

    from voice.clone import VoiceCloner

    cloner = VoiceCloner(
        reference_samples_dir=config.voice.reference_samples_dir,
        output_dir=config.voice.output_dir,
        language=config.voice.language,
    )
    console.print("[cyan]synthesizing...[/cyan]")
    path = cloner.synthesize(text, out_filename=out)
    console.print(f"[bold green]saved:[/bold green] {path}")


if __name__ == "__main__":
    app()