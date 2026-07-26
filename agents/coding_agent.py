from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Awaitable, Callable

from agents.base_agent import BaseAgent

ConfirmFn = Callable[[str], Awaitable[bool]]


class PathEscapesWorkspaceError(Exception):
    pass


class CodingAgent(BaseAgent):
    """Reads, writes, and runs code inside a single workspace root. Every
    path is resolved and checked against that root before touching disk —
    this is the one guard that matters most here, since a coding agent
    with unrestricted filesystem access is the whole ballgame if the
    planner ever hallucinates a path.

    Shell commands are split into a config-defined "safe" allowlist
    (read-only inspection: git status, pytest, ls, ...) that run without
    prompting, and everything else, which goes through `confirm_fn` before
    executing. Writes go through the same gate when
    require_confirmation_for_writes is set.
    """

    name = "coding"
    description = (
        "Reads and writes files, runs shell commands, and works with git — "
        "all confined to the configured workspace."
    )

    def __init__(
        self,
        workspace_root: str,
        safe_shell_commands: list[str],
        require_confirmation_for_writes: bool,
        confirm_fn: ConfirmFn,
    ) -> None:
        self._root = Path(workspace_root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._safe_prefixes = tuple(safe_shell_commands)
        self._require_confirmation_for_writes = require_confirmation_for_writes
        self._confirm_fn = confirm_fn
        super().__init__()

    def register_tools(self) -> None:
        self.add_tool("read_file", "Read a text file. Instruction: relative path.", self.read_file)
        self.add_tool(
            "write_file",
            "Write/overwrite a text file. Instruction: '<relative path> ||| <content>'.",
            self.write_file,
            destructive=True,
        )
        self.add_tool("list_files", "List files under a relative directory (default: root).", self.list_files)
        self.add_tool(
            "run_shell",
            "Run a shell command inside the workspace. Instruction: the command itself.",
            self.run_shell,
            destructive=True,
        )
        self.add_tool(
            "git_status",
            "Show 'git status' for the workspace repo.",
            lambda _instruction: self._run_raw("git status"),
        )

    def _resolve(self, relative_path: str) -> Path:
        candidate = (self._root / relative_path.strip()).resolve()
        if self._root not in candidate.parents and candidate != self._root:
            raise PathEscapesWorkspaceError(
                f"'{relative_path}' resolves outside the workspace root ({self._root})"
            )
        return candidate

    async def read_file(self, instruction: str) -> str:
        path = self._resolve(instruction)
        if not path.exists():
            return f"File not found: {path.relative_to(self._root)}"
        return path.read_text(encoding="utf-8", errors="replace")

    async def write_file(self, instruction: str) -> str:
        if "|||" not in instruction:
            return "write_file expects '<relative path> ||| <content>'"
        rel_path, content = instruction.split("|||", 1)
        path = self._resolve(rel_path)

        if self._require_confirmation_for_writes:
            approved = await self._confirm_fn(
                f"Write {len(content)} chars to {path.relative_to(self._root)}?"
            )
            if not approved:
                return "Write cancelled by user."

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.strip() + "\n", encoding="utf-8")
        return f"Wrote {path.relative_to(self._root)}"

    async def list_files(self, instruction: str) -> str:
        rel_dir = instruction.strip() or "."
        path = self._resolve(rel_dir)
        if not path.exists():
            return f"Directory not found: {rel_dir}"
        entries = sorted(p.relative_to(self._root).as_posix() for p in path.rglob("*"))
        return "\n".join(entries) if entries else "(empty)"

    async def run_shell(self, instruction: str) -> str:
        command = instruction.strip()
        is_safe = any(command.startswith(prefix) for prefix in self._safe_prefixes)
        if not is_safe:
            approved = await self._confirm_fn(f"Run shell command: `{command}` ?")
            if not approved:
                return "Command cancelled by user."
        return await self._run_raw(command)

    async def _run_raw(self, command: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=self._root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode(errors="replace")
        return output if output.strip() else f"(command exited {proc.returncode}, no output)"