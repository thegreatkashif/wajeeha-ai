import tempfile

import pytest

from agents.coding_agent import CodingAgent, PathEscapesWorkspaceError


async def _always_confirm(_prompt: str) -> bool:
    return True


async def _always_deny(_prompt: str) -> bool:
    return False


@pytest.mark.asyncio
async def test_read_write_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        agent = CodingAgent(
            workspace_root=tmp,
            safe_shell_commands=["git status"],
            require_confirmation_for_writes=False,
            confirm_fn=_always_confirm,
        )
        result = await agent.write_file("notes.txt ||| hello world")
        assert "Wrote notes.txt" in result
        content = await agent.read_file("notes.txt")
        assert content.strip() == "hello world"


@pytest.mark.asyncio
async def test_write_blocked_without_confirmation():
    with tempfile.TemporaryDirectory() as tmp:
        agent = CodingAgent(
            workspace_root=tmp,
            safe_shell_commands=[],
            require_confirmation_for_writes=True,
            confirm_fn=_always_deny,
        )
        result = await agent.write_file("notes.txt ||| hello world")
        assert "cancelled" in result.lower()


@pytest.mark.asyncio
async def test_path_traversal_is_blocked():
    with tempfile.TemporaryDirectory() as tmp:
        agent = CodingAgent(
            workspace_root=tmp,
            safe_shell_commands=[],
            require_confirmation_for_writes=False,
            confirm_fn=_always_confirm,
        )
        with pytest.raises(PathEscapesWorkspaceError):
            agent._resolve("../../etc/passwd")


@pytest.mark.asyncio
async def test_safe_shell_command_runs_without_confirmation():
    with tempfile.TemporaryDirectory() as tmp:
        agent = CodingAgent(
            workspace_root=tmp,
            safe_shell_commands=["echo"],
            require_confirmation_for_writes=False,
            confirm_fn=_always_deny,  # would fail the test if confirmation were requested
        )
        result = await agent.run_shell("echo hi")
        assert "hi" in result