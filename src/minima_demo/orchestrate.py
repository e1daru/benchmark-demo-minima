"""Shared tail for both tracks: stream the learning curve, sweep sliders, write artifacts + report.

Given a finished result Matrix and the task specs, this is identical for the live and dataset
tracks — the only thing that differs upstream is how the Matrix was built (real calls vs dataset
lookup). Keeping it here means one definition of the run protocol and output layout.
"""

from __future__ import annotations

import json
from pathlib import Path

from minima_client import MinimaClient

from . import report
from .baselines import Matrix
from .config import DEFAULT_CURVE_SLIDER, DEFAULT_SLIDERS, fresh_namespace
from .metrics import Cell, RoutedRecord, write_jsonl
from .runner import run_stream
from .spec import TaskSpec


def route_and_report(client: MinimaClient, *, track: str, matrix: Matrix, specs: list[TaskSpec],
                     sliders: tuple[float, ...] = DEFAULT_SLIDERS,
                     curve_slider: float = DEFAULT_CURVE_SLIDER, epochs: int = 1,
                     outdir: Path | None = None) -> Path:
    premium_id = matrix.premium_model()
    pool_ids = matrix.models

    def resolve(task_id: str, model_id: str) -> Cell:
        return matrix.cells.get(task_id, {}).get(model_id) or Cell(0.0, 0.0, 0, 0)

    ns = fresh_namespace(track)
    # Stream the task set `epochs` times with feedback on: a fresh namespace starts cold and Minima
    # converges as repeated outcomes accumulate — that climb is the learning curve.
    curve_specs = specs * epochs
    print(f"\nrouting learning curve (namespace={ns}, slider={curve_slider}) over "
          f"{len(curve_specs)} steps ({len(specs)} tasks x {epochs} epochs) …")
    curve = run_stream(
        client, track=track, namespace=ns, tasks=curve_specs, candidate_models=pool_ids,
        resolve_cell=resolve, slider=curve_slider, baseline_model_id=premium_id, learn=True,
        on_step=lambda r: print(f"  step {r.step+1:3d} {r.task_id[:22]:22s} -> {r.model_id:22s} "
                                f"basis={r.decision_basis} acc={r.accuracy:.2f}", flush=True),
    )

    print(f"\nrouting slider sweep (warm, no feedback, concurrent) over {len(sliders)} sliders …")
    sweep: list[RoutedRecord] = []
    for s in sliders:
        sweep.extend(run_stream(client, track=track, namespace=ns, tasks=specs,
                                candidate_models=pool_ids, resolve_cell=resolve, slider=s,
                                baseline_model_id=premium_id, learn=False, workers=8))

    try:
        savings = client.savings(namespace=ns).model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001
        savings = {"error": str(exc)}

    out = outdir or Path("results") / f"{track}-{ns.split('-')[-1]}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix.json").write_text(json.dumps(matrix.to_dict(), indent=1))
    write_jsonl(out / "routed_curve.jsonl", curve)
    write_jsonl(out / "routed_sweep.jsonl", sweep)
    (out / "savings.json").write_text(json.dumps(savings, indent=2))

    results = report.assemble(track, matrix, curve, sweep, savings,
                              namespace=ns, curve_slider=curve_slider)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    report.render(results, out / "report.html")
    print(f"\n✓ {track} track complete → {out}/report.html")
    return out
