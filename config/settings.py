"""
Typed configuration for Wajeeha AI.

Non-secret behavior lives in config.yaml. Secrets (API keys, tokens) live
in a local .env file (see .env.example) and are never written to disk by
the app itself.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Secrets(BaseSettings):
    """Loaded from environment / .env. Never logged, never persisted elsewhere."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    home_assistant_token: str | None = Field(default=None, alias="HOME_ASSISTANT_TOKEN")


class LLMProviderConfig(BaseModel):
    model: str
    max_tokens: int = 4096
    base_url: str | None = None


class ShortTermConfig(BaseModel):
    max_turns: int = 40


class LongTermConfig(BaseModel):
    db_path: str = "./data/long_term.sqlite3"


class SemanticConfig(BaseModel):
    chroma_path: str = "./data/chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_name: str = "wajeeha_semantic"


class MemoryConfig(BaseModel):
    short_term: ShortTermConfig
    long_term: LongTermConfig
    semantic: SemanticConfig


class HomeAssistantConfig(BaseModel):
    base_url: str


class HomeAgentConfig(BaseModel):
    enabled: bool = True
    home_assistant: HomeAssistantConfig
    default_room: str = "living_room"


class CodingAgentConfig(BaseModel):
    enabled: bool = True
    workspace_root: str = "./workspace"
    safe_shell_commands: list[str] = Field(default_factory=list)
    require_confirmation_for_writes: bool = True


class AgentsConfig(BaseModel):
    coding: CodingAgentConfig
    home: HomeAgentConfig


class PlannerConfig(BaseModel):
    max_plan_steps: int = 12
    max_retries_per_step: int = 2
    reflect_after_each_step: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "./logs"


class LLMConfig(BaseModel):
    default_provider: str
    providers: dict[str, LLMProviderConfig]


class AppConfig(BaseModel):
    llm: LLMConfig
    memory: MemoryConfig
    agents: AgentsConfig
    planner: PlannerConfig
    logging: LoggingConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Copy config/config.yaml.example "
            "if it's missing, or restore the default."
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    raw = _load_yaml(CONFIG_PATH)
    return AppConfig(**raw)


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    return Secrets()


def ensure_runtime_dirs(config: AppConfig) -> None:
    """Create data/log/workspace directories referenced by config, if missing."""
    paths = [
        Path(config.memory.long_term.db_path).parent,
        Path(config.memory.semantic.chroma_path),
        Path(config.logging.log_dir),
        Path(config.agents.coding.workspace_root),
    ]
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")