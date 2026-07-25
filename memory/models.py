from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ConversationTurn(BaseModel):
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str | None = None  # which agent produced/handled this turn, if any


class LongTermFact(BaseModel):
    """A durable fact about the user's world: a project, a preference, a
    device, a server, a schedule item, etc. Freeform key/value plus category
    so the planner can query by kind ("device", "preference", "project")."""

    id: int | None = None
    category: str
    key: str
    value: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SemanticMemoryHit(BaseModel):
    text: str
    metadata: dict
    score: float