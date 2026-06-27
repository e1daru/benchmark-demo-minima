"""Baselines and the margin metric, computed from a per-(task, model) result matrix.

Definitions follow ``minima/tests/eval/harness.py`` (the eval's ``_baselines``):

- **all-premium**  — always use the single best model (highest mean accuracy on the suite).
- **cheapest**     — always use the model with the lowest mean realized cost.
- **oracle**       — per task, the highest-accuracy model, ties broken by lowest cost. The
                     "most-effective / perfect router" we measure the *margin* against.
- **random**       — analytic expectation of picking a model uniformly at random.

The *margin* of any router on a task is ``oracle_accuracy - router_accuracy`` (0 = perfect).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .metrics import Cell


@dataclass
class Point:
    """An (accuracy, cost) operating point with a label, for the Pareto chart and cards."""

    label: str
    accuracy: float
    cost_usd: float
    model_id: str = ""


@dataclass
class Matrix:
    """Realized results for every resolved model on every task in the suite."""

    cells: dict[str, dict[str, Cell]]      # task_id -> model_id -> Cell
    models: list[str]                      # resolved model ids (columns)
    prices: dict[str, tuple[float, float]] # model_id -> (input_$/Mtok, output_$/Mtok)
    task_types: dict[str, str]             # task_id -> task type
    task_order: list[str]                  # canonical task order (for stable iteration)
    difficulties: dict[str, str] = field(default_factory=dict)  # task_id -> easy/medium/hard

    # --- (de)serialization — the committed fixture that makes replay key-free ------------------
    def to_dict(self) -> dict:
        return {
            "models": self.models,
            "prices": {m: list(p) for m, p in self.prices.items()},
            "task_types": self.task_types,
            "difficulties": self.difficulties,
            "task_order": self.task_order,
            "cells": {t: {m: asdict(c) for m, c in row.items()} for t, row in self.cells.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Matrix":
        cells = {t: {m: Cell(**c) for m, c in row.items()} for t, row in d["cells"].items()}
        prices = {m: tuple(p) for m, p in d["prices"].items()}
        return cls(cells=cells, models=list(d["models"]), prices=prices,
                   task_types=dict(d["task_types"]), task_order=list(d["task_order"]),
                   difficulties=dict(d.get("difficulties", {})))

    # --- per-model aggregates -----------------------------------------------------------------
    def _mean(self, getter) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in self.models:
            vals = [getter(self.cells[t][m]) for t in self.task_order if m in self.cells.get(t, {})]
            out[m] = sum(vals) / len(vals) if vals else 0.0
        return out

    def model_accuracy(self) -> dict[str, float]:
        return self._mean(lambda c: c.accuracy)

    def model_cost(self) -> dict[str, float]:
        return self._mean(lambda c: c.cost_usd)

    def model_points(self) -> list[Point]:
        acc, cost = self.model_accuracy(), self.model_cost()
        return [Point(m, acc[m], cost[m], m) for m in self.models]

    # --- named single-model baselines ---------------------------------------------------------
    # "all-premium" is the standard *strong-model* baseline (cf. RouteLLM): the highest-accuracy
    # model you'd default to if cost were no object, tie-broken toward the pricier one so it lands
    # on the genuine flagship. "cheapest" is the lowest-priced model. We never let a strictly
    # dominated model (pricier *and* weaker) become "premium".
    def _blended_price(self, model_id: str) -> float:
        lo, hi = self.prices.get(model_id, (0.0, 0.0))
        return lo + hi

    PREMIUM_ACC_TOL = 0.03  # models within this of the best count as a quality-tie

    def premium_model(self) -> str:
        # The flagship a cost-unaware user defaults to: among models statistically tied for best
        # accuracy, the most expensive one. When one model is a clear accuracy winner (e.g. on hard
        # benchmarks) it stands alone; when many models tie (easy suites) the priciest is "premium".
        acc = self.model_accuracy()
        best = max(acc.values()) if acc else 0.0
        tied = [m for m in self.models if acc[m] >= best - self.PREMIUM_ACC_TOL]
        return max(tied, key=self._blended_price)

    def cheapest_model(self) -> str:
        return min(self.models, key=self._blended_price)

    def _single_model_point(self, label: str, model_id: str) -> Point:
        acc = self.model_accuracy()[model_id]
        cost = self.model_cost()[model_id]
        return Point(label, acc, cost, model_id)

    # --- per-task oracle ----------------------------------------------------------------------
    def oracle_for(self, task_id: str) -> tuple[str, float, float]:
        row = self.cells[task_id]
        best = max(row, key=lambda m: (row[m].accuracy, -row[m].cost_usd))
        return best, row[best].accuracy, row[best].cost_usd

    def oracle_point(self) -> Point:
        accs, costs = [], []
        for t in self.task_order:
            _, a, c = self.oracle_for(t)
            accs.append(a); costs.append(c)
        n = len(self.task_order) or 1
        return Point("oracle (most-effective)", sum(accs) / n, sum(costs) / n)

    def random_point(self) -> Point:
        acc, cost = self.model_accuracy(), self.model_cost()
        n = len(self.models) or 1
        return Point("random", sum(acc.values()) / n, sum(cost.values()) / n)

    # --- full baseline summary ----------------------------------------------------------------
    def baselines(self) -> dict[str, Point]:
        prem, cheap = self.premium_model(), self.cheapest_model()
        return {
            "premium": self._single_model_point(f"all-premium ({prem})", prem),
            "cheapest": self._single_model_point(f"cheapest ({cheap})", cheap),
            "oracle": self.oracle_point(),
            "random": self.random_point(),
        }
