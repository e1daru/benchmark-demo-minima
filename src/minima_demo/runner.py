"""The prequential routing loop shared by both tracks.

Stream tasks through Minima in a fixed order: ``recommend`` → look up the chosen model's realized
result → optionally ``feedback`` (which is what makes the next recommendation smarter). With
``learn=True`` over a fresh namespace this *is* the learning curve; with ``learn=False`` it measures
the current (warm) policy at a slider without changing it — one Pareto operating point.

The realized result for a (task, model) pair is supplied by a ``resolve_cell`` callback so the same
loop serves the live track (results from real model calls) and the dataset track (results looked up
in LLMRouterBench), test-then-train style.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

import httpx
from minima_client import MinimaClient, MinimaError
from minima.schemas.common import Constraints

from .metrics import Cell, RoutedRecord, grade_outcome
from .spec import TaskSpec

# (task_id, model_id) -> realized Cell
ResolveCell = Callable[[str, str], Cell]

T = TypeVar("T")
_TRANSIENT = (httpx.TransportError, httpx.RemoteProtocolError, ConnectionError)


def _retry(fn: Callable[[], T], attempts: int = 4, base_delay: float = 0.6) -> T:
    """Retry a hosted call through transient network blips (connection resets, read errors)."""
    for i in range(attempts):
        try:
            return fn()
        except _TRANSIENT:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i))
        except MinimaError as exc:
            status = getattr(exc, "status", None)
            if i == attempts - 1 or (status is not None and status < 500):
                raise
            time.sleep(base_delay * (2 ** i))
    raise RuntimeError("unreachable")


def run_stream(
    client: MinimaClient,
    *,
    track: str,
    namespace: str,
    tasks: list[TaskSpec],
    candidate_models: list[str],
    resolve_cell: ResolveCell,
    slider: float,
    baseline_model_id: str | None = None,
    learn: bool = True,
    on_step: Callable[[RoutedRecord], None] | None = None,
    workers: int = 1,
) -> list[RoutedRecord]:
    """Route every task once, returning one RoutedRecord per task.

    The learning curve (``learn=True``) is always sequential — feedback from task *i* must land
    before task *i+1* is recommended. Measurement-only passes (``learn=False``, the slider sweep)
    have no such dependency, so ``workers>1`` fans the recommends out concurrently.
    """
    constraints = Constraints(candidate_models=candidate_models)

    def route(step: int, task: TaskSpec) -> RoutedRecord:
        rec = _retry(lambda: client.recommend(
            {"task": task.prompt, "task_type": task.task_type, "difficulty": task.difficulty},
            cost_quality_tradeoff=slider,
            namespace=namespace,
            constraints=constraints,
            baseline_model_id=baseline_model_id,
        ))
        rm = rec.recommended_model
        cell = resolve_cell(task.id, rm.model_id)
        outcome = grade_outcome(cell.accuracy)
        if learn:
            _retry(lambda: client.feedback(
                rec.recommendation_id, rm.model_id, outcome,
                quality_score=round(cell.accuracy, 6),
                input_tokens=cell.input_tokens, output_tokens=cell.output_tokens,
                actual_cost_usd=round(cell.cost_usd, 8),
                latency_ms=int(cell.latency_ms) if cell.latency_ms is not None else None,
                verified_in_production=True,
            ))
        return RoutedRecord(
            track=track, step=step, task_id=task.id, task_type=task.task_type, slider=slider,
            model_id=rm.model_id, provider=rm.provider, decision_basis=str(rec.decision_basis),
            confidence=rec.confidence, predicted_success=rm.predicted_success,
            est_cost_usd=rm.est_cost_usd, accuracy=cell.accuracy, cost_usd=cell.cost_usd,
            input_tokens=cell.input_tokens, output_tokens=cell.output_tokens,
            latency_ms=cell.latency_ms, outcome=outcome, learned=learn,
        )

    if learn or workers <= 1:
        out: list[RoutedRecord] = []
        for step, task in enumerate(tasks):
            record = route(step, task)
            out.append(record)
            if on_step:
                on_step(record)
        return out

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def safe(step: int, task: TaskSpec):
        try:
            return step, route(step, task)
        except Exception as exc:  # noqa: BLE001 — one bad call must not abort the measurement pass
            print(f"  [warn] step {step} ({task.id}) dropped: {type(exc).__name__}", flush=True)
            return step, None

    ordered: list[RoutedRecord | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(safe, i, t) for i, t in enumerate(tasks)]
        for fut in as_completed(futs):
            step, record = fut.result()
            ordered[step] = record
    return [r for r in ordered if r is not None]
