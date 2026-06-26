"""Dataset (replay) track — reproducible, large-N, free, from the public LLMRouterBench benchmark.

LLMRouterBench ships realized ``score`` and ``cost`` for every (prompt, model), so we get the
premium/cheapest/oracle baselines and the learning curve with zero live model calls. The routing
universe is the subset of LLMRouterBench models that resolve to the hosted catalog (see
:data:`constants.DATASET_TO_CATALOG_ALIAS`) — gpt-5 and the open models aren't in the catalog, so
they're dropped (documented in docs/methodology.md). The matrix is keyed by the *catalog* ids the
hosted service routes over; cells are looked up under the corresponding dataset ids.

The matrix build reads the cached tarball with no network (HF offline); only the routing pass to
api.minima.sh needs connectivity.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from pathlib import Path

from .baselines import Matrix
from .catalog import fetch_catalog, price_map, resolve_dataset_pool
from .config import DEFAULT_CURVE_SLIDER, DEFAULT_SLIDERS, Settings, make_client
from .constants import task_type_for
from .metrics import Cell
from .orchestrate import route_and_report
from .spec import TaskSpec

# A spread of dataset families covering qa / reasoning / math / code by default.
DEFAULT_DATASETS: tuple[str, ...] = (
    "simpleqa", "mmlupro", "gpqa", "arenahard_math", "arenahard_coding", "aime",
)
HARD_DATASETS = {"aime", "gpqa", "hle", "swe-bench", "livecodebench", "arc-agi",
                 "livemathbench", "arenahard_math"}

# Committed fixture: matrix + prompts. Lets the dataset track replay with no tarball and no scan.
FIXTURE = Path("fixtures/dataset_matrix.json")


def _save_fixture(matrix: Matrix, specs: list[TaskSpec], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"matrix": matrix.to_dict(),
                                "specs": [asdict(s) for s in specs]}, indent=1))


def _load_fixture(path: Path) -> tuple[Matrix, list[TaskSpec]]:
    d = json.loads(path.read_text())
    return Matrix.from_dict(d["matrix"]), [TaskSpec(**s) for s in d["specs"]]


def _cached_tarball() -> str:
    """Path to the cached LLMRouterBench tarball; force HF offline so this never hits the network."""
    import minima.seeding.llmrouterbench as lr
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return lr.download_tarball()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "LLMRouterBench tarball is not cached. Fetch it once (≈1.28 GB) with network on:\n"
            "  make fetch-dataset\n"
            f"(underlying error: {exc})"
        )


def build_dataset_matrix(dataset_pool: list[tuple[str, str]], prices: dict[str, tuple[float, float]],
                         datasets: tuple[str, ...], per_dataset: int, seed: int) -> tuple[Matrix, list[TaskSpec]]:
    import minima.seeding.llmrouterbench as lr

    tarball = _cached_tarball()
    ds_ids = [d for d, _ in dataset_pool]
    cat_of = dict(dataset_pool)
    rng = random.Random(seed)

    cells: dict[str, dict[str, Cell]] = {}
    task_types: dict[str, str] = {}
    order: list[str] = []
    specs: list[TaskSpec] = []

    # One pass over the tarball for ALL datasets (a per-dataset scan would re-read ~1.28GB each time).
    from collections import defaultdict
    scan: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in lr.iter_raw_records(tarball_path=tarball, datasets=set(datasets), models=set(ds_ids)):
        scan[r["dataset_id"]][str(r["index"])][r["model_name"]] = r

    for ds in datasets:
        bucket = scan.get(ds, {})
        complete = [(idx, recs) for idx, recs in bucket.items() if all(m in recs for m in ds_ids)]
        complete.sort(key=lambda kv: (len(kv[0]), kv[0]))  # stable order before sampling
        if len(complete) > per_dataset:
            picks = sorted(rng.sample(range(len(complete)), per_dataset))
            complete = [complete[i] for i in picks]

        tt = task_type_for(ds)
        diff = "hard" if ds in HARD_DATASETS else "medium"
        for idx, recs in complete:
            tid = f"{ds}#{idx}"
            row: dict[str, Cell] = {}
            for ds_id in ds_ids:
                rec = recs[ds_id]
                row[cat_of[ds_id]] = Cell(
                    accuracy=float(rec["score"]),
                    cost_usd=float(rec["cost"]),
                    input_tokens=int(float(rec["prompt_tokens"])),
                    output_tokens=int(float(rec["completion_tokens"])),
                )
            cells[tid] = row
            task_types[tid] = tt
            order.append(tid)
            prompt = (recs[ds_ids[0]]["prompt"] or "")[:4000]
            specs.append(TaskSpec(id=tid, prompt=prompt, task_type=tt, difficulty=diff))

    # Interleave task types into a mixed stream (a fixed shuffle) for a representative curve.
    rng.shuffle(order)
    spec_by_id = {s.id: s for s in specs}
    specs = [spec_by_id[t] for t in order]

    models = [cat_of[d] for d in ds_ids]
    matrix = Matrix(cells=cells, models=models, prices={m: prices[m] for m in models},
                    task_types=task_types, task_order=order)
    return matrix, specs


def run_dataset(settings: Settings, *, max_tasks: int = 96, per_dataset: int = 16,
                datasets: set[str] | None = None, rebuild: bool = False, epochs: int = 1,
                sliders: tuple[float, ...] = DEFAULT_SLIDERS,
                curve_slider: float = DEFAULT_CURVE_SLIDER, outdir: Path | None = None) -> Path:
    client = make_client(settings)

    if FIXTURE.exists() and not rebuild:
        matrix, specs = _load_fixture(FIXTURE)
        print(f"replaying cached dataset matrix from {FIXTURE} "
              f"({len(matrix.task_order)} tasks x {len(matrix.models)} models; "
              f"no tarball/scan needed — pass --rebuild to refresh).")
    else:
        catalog = fetch_catalog(client)
        pool = resolve_dataset_pool(catalog)
        if len(pool) < 2:
            raise SystemExit(f"dataset pool too small ({len(pool)} models resolve to the catalog).")
        ds = tuple(datasets) if datasets else DEFAULT_DATASETS
        print(f"dataset track: building matrix (one ~1.28GB scan) — pool {[c for _, c in pool]} "
              f"over {list(ds)} ({per_dataset}/dataset, cap {max_tasks}) …")
        matrix, specs = build_dataset_matrix(pool, price_map(catalog), ds, per_dataset, settings.seed)
        if len(specs) > max_tasks:
            specs = specs[:max_tasks]
            keep = {s.id for s in specs}
            matrix.task_order = [t for t in matrix.task_order if t in keep]
            matrix.cells = {t: matrix.cells[t] for t in keep}
            matrix.task_types = {t: matrix.task_types[t] for t in keep}
        _save_fixture(matrix, specs, FIXTURE)
        print(f"built + cached {len(matrix.task_order)} tasks x {len(matrix.models)} models "
              f"-> {FIXTURE} (no model spend).")

    return route_and_report(client, track="dataset", matrix=matrix, specs=specs,
                            sliders=sliders, curve_slider=curve_slider, epochs=epochs, outdir=outdir)
