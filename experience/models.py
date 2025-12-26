from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


Status = Literal["success", "failure", "partial", "timeout", "aborted"]


class ToolCallEvent(BaseModel):
    tool: str
    command: str | None = None
    args: dict[str, Any] | None = None
    exit_code: int | None = None
    duration_ms: int | None = None
    stdout: str | None = None
    stderr: str | None = None


class TestRunEvent(BaseModel):
    framework: str | None = None
    command: str
    passed: bool | None = None
    duration_ms: int | None = None
    summary: str | None = None


class ErrorEvent(BaseModel):
    error_type: str
    message: str | None = None
    stack: str | None = None
    file: str | None = None
    line: int | None = None


class ExperienceIngestRequest(BaseModel):
    # Task/run identity
    run_id: str | None = Field(default=None, description="uuid (если есть). Если нет — создадим новый.")
    task_type: str = Field(default="generic")
    goal: str | None = None

    # Context
    project: str | None = None
    repo: str | None = None
    branch: str | None = None
    commit: str | None = None
    stack: dict[str, Any] | None = None  # python/node версии, фреймворк и т.п.
    affected_files: list[str] = Field(default_factory=list)

    # Timeline + result
    started_at: datetime | None = None
    ended_at: datetime | None = None
    status: Status = "success"
    error_type: str | None = None
    quality_score: float | None = None
    duration_ms: int | None = None

    # Details
    tool_calls: list[ToolCallEvent] = Field(default_factory=list)
    test_runs: list[TestRunEvent] = Field(default_factory=list)
    errors: list[ErrorEvent] = Field(default_factory=list)


class ExperienceQuery(BaseModel):
    task_type: str | None = None
    context_hash: str | None = None
    limit: int = 5


class ExperienceResult(BaseModel):
    items: list[dict[str, Any]]


