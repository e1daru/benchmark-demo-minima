"""Shared metric schema for both tracks — one definition of a measured result and a routing event.

``grade_outcome`` and its thresholds are imported from the public ``minima_harness`` package so the
demo's success/partial/failure labels match Minima's own convention exactly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from minima_harness.tasks.task_set import (  # reuse Minima's own grading convention
    PARTIAL_THRESHOLD,
    SUCCESS_THRESHOLD,
    grade_outcome,
)

__all__ = [
    "Cell", "RoutedRecord", "grade_outcome", "SUCCESS_THRESHOLD", "PARTIAL_THRESHOLD",
    "write_jsonl", "read_jsonl",
]


@dataclass
class Cell:
    """One realized ``(task, model)`` result — the atom both baselines and routing read from."""

    accuracy: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    latency_ms: float | None = None
    text: str = ""        # captured model output (live build); dropped on fixture replay
    error: str = ""       # non-empty if the model call failed

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class RoutedRecord:
    """One Minima routing decision plus the realized outcome it was scored on.

    A stream of these (ordered by ``step``) is the learning curve; one per slider is a Pareto point.
    """

    track: str            # "catalog" | "dataset"
    step: int             # position in the streamed task order (for the curve)
    task_id: str
    task_type: str
    slider: float
    model_id: str
    provider: str
    decision_basis: str   # memory | prior | llm
    confidence: float
    predicted_success: float
    est_cost_usd: float   # what Minima predicted before the run
    accuracy: float       # realized
    cost_usd: float       # realized
    input_tokens: int
    output_tokens: int
    latency_ms: float | None
    outcome: str          # success | partial | failure
    learned: bool         # True if feedback was sent (curve); False for measurement-only sweeps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_jsonl(path: str | Path, records: Iterable[Any]) -> Path:
    """Write dataclass records (or dicts) one-per-line; create parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict() if hasattr(r, "to_dict") else r) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]
