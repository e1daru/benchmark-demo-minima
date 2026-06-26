"""The minimal task description the router consumes (shared by both tracks)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    id: str
    prompt: str
    task_type: str         # one of minima.schemas.common.TaskType values
    difficulty: str = "medium"
