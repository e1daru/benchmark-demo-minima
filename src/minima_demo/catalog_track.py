"""Catalog (live) track — route over the real hosted catalog using your provider keys.

Pipeline:
  1. Build the (task, model) matrix: call every resolved catalog model on every task once, score
     the output with the task's deterministic ``quality_fn``, and record cost/tokens/latency. This
     yields the premium / cheapest / oracle baselines from *real* numbers, and is cached to a JSON
     fixture so the dashboard can be regenerated later with no keys and no spend (``--use-fixture``).
  2. Stream the tasks through Minima (fresh namespace) with feedback on → the learning curve.
  3. Re-route at each slider (warm namespace, no feedback) → Minima's Pareto operating points.

A ``--dry-run`` simulator fabricates a plausible matrix so the whole pipeline can be exercised
without spending a cent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .baselines import Matrix
from .spec import TaskSpec
from .catalog import fetch_catalog, resolve_live_pool
from .config import DEFAULT_CURVE_SLIDER, DEFAULT_SLIDERS, Settings, make_client
from .metrics import Cell
from .orchestrate import route_and_report
from .providers import call_model
from .tasks import to_spec
from .tasks.livecode_suite import load_livecode_suite
from .tasks.suite import LiveTask, select

FIXTURE = Path("fixtures/catalog_matrix.json")
HARD_FIXTURE = Path("fixtures/hard_matrix.json")
CODE_FIXTURE = Path("fixtures/code_matrix.json")


# --- matrix construction ----------------------------------------------------------------------

def _cost(in_tok: int, out_tok: int, price: tuple[float, float]) -> float:
    return in_tok / 1e6 * price[0] + out_tok / 1e6 * price[1]


def estimate_cost(tasks: list[LiveTask], pool, max_tokens: int) -> float:
    """Rough USD upper-ish estimate for a full live matrix build."""
    total = 0.0
    for t in tasks:
        in_tok = len(t.prompt) // 4 + 24
        out_tok = int(max_tokens * 0.6)
        for m in pool:
            total += _cost(in_tok, out_tok, m.price)
    return total


def _simulate_cell(task: LiveTask, model_id: str, tier_rank: float, price) -> Cell:
    """Deterministic fake result for --dry-run: stronger (pricier) models do better on hard tasks."""
    h = int(hashlib.sha256(f"{task.id}:{model_id}".encode()).hexdigest(), 16) % 1000 / 1000.0
    diff = {"easy": 0.1, "medium": 0.45, "hard": 0.8}.get(task.difficulty, 0.4)
    base = 0.55 + 0.4 * tier_rank - diff + 0.15 * (h - 0.5)
    acc = 1.0 if base > 0.6 else (0.5 if base > 0.3 else 0.0)
    in_tok, out_tok = len(task.prompt) // 4 + 24, 120 + int(300 * h)
    return Cell(acc, _cost(in_tok, out_tok, price), in_tok, out_tok, latency_ms=400 + 1200 * h,
                text="[simulated]")


def _live_cell(settings: Settings, task: LiveTask, model, max_tokens: int) -> Cell:
    rn = call_model(model.provider, model.model_id, task.prompt,
                    api_key=settings.provider_keys[model.provider], max_tokens=max_tokens)
    acc = 0.0 if not rn.ok else task.quality_fn(rn.text)
    return Cell(accuracy=acc, cost_usd=_cost(rn.input_tokens, rn.output_tokens, model.price),
                input_tokens=rn.input_tokens, output_tokens=rn.output_tokens,
                latency_ms=rn.latency_ms, text=(rn.text or "")[:200], error=rn.error)


def build_matrix(settings: Settings, tasks: list[LiveTask], pool, *, max_tokens: int,
                 dry_run: bool, workers: int = 16) -> Matrix:
    prices = {m.model_id: m.price for m in pool}
    model_ids = [m.model_id for m in pool]
    # tier rank in [0,1] by output price — only used by the simulator.
    outs = sorted({m.output_cost_per_mtok for m in pool})
    rank = {m.model_id: outs.index(m.output_cost_per_mtok) / max(1, len(outs) - 1) for m in pool}

    cells: dict[str, dict[str, Cell]] = {t.id: {} for t in tasks}
    if dry_run:
        for t in tasks:
            for m in pool:
                cells[t.id][m.model_id] = _simulate_cell(t, m.model_id, rank[m.model_id], m.price)
    else:
        # Calls are independent — fan them out so a 12×N matrix builds in seconds, not minutes
        # (and a mid-run network drop only fails the in-flight cells, captured as errors).
        from concurrent.futures import ThreadPoolExecutor, as_completed
        jobs = [(t, m) for t in tasks for m in pool]
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_live_cell, settings, t, m, max_tokens): (t, m) for t, m in jobs}
            for fut in as_completed(futs):
                t, m = futs[fut]
                cells[t.id][m.model_id] = fut.result()
                done += 1
                if done % len(pool) == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} model calls complete", flush=True)

    return Matrix(cells=cells, models=model_ids, prices=prices,
                  task_types={t.id: t.task_type for t in tasks},
                  difficulties={t.id: t.difficulty for t in tasks},
                  task_order=[t.id for t in tasks])


# --- orchestration ----------------------------------------------------------------------------

def run_catalog(settings: Settings, *, max_tasks: int | None = None,
                providers: set[str] | None = None, max_tokens: int = 768,
                dry_run: bool = False, use_fixture: bool = False, assume_yes: bool = False,
                hard: bool = False, hard_per_dataset: int = 3, code: bool = False,
                workers: int = 16, sliders: tuple[float, ...] = DEFAULT_SLIDERS,
                curve_slider: float = DEFAULT_CURVE_SLIDER, epochs: int = 3,
                outdir: Path | None = None) -> Path:
    client = make_client(settings)
    catalog = fetch_catalog(client)
    pool = resolve_live_pool(catalog, settings)
    if providers:
        pool = [m for m in pool if m.provider in providers]
    if len(pool) < 2:
        raise SystemExit(f"live pool too small ({len(pool)}); need keys for >=2 providers' models.")

    # Three task universes share this live pipeline:
    #  - code: LiveCodeBench problems, scored by REALLY RUNNING the model's code against tests.
    #  - hard: a difficulty-graded, multi-type frontier mix (MATH-500 levels 1–5 + LLMRouterBench
    #          aime/gpqa/... + IFEval) — spans easy→hard so routing actually has something to do.
    #  - catalog: the curated everyday suite (cheap models suffice — the "save cost" regime).
    # The harder universes get a bigger output budget so step-by-step solutions + the answer fit.
    track = "code" if code else ("hard" if hard else "catalog")
    fixture = CODE_FIXTURE if code else (HARD_FIXTURE if hard else FIXTURE)
    if code:
        tasks = load_livecode_suite(n=max_tasks, seed=settings.seed)
        max_tokens = max(max_tokens, 4096)
    elif hard:
        from .tasks.frontier_suite import load_frontier_suite
        tasks = load_frontier_suite(seed=settings.seed, llmrouterbench_per_dataset=hard_per_dataset)
        if max_tasks:
            tasks = tasks[:max_tasks]
        max_tokens = max(max_tokens, 2048)
    else:
        tasks = select(max_tasks=max_tasks)
    print(f"{track} track: {len(tasks)} tasks x {len(pool)} models "
          f"({', '.join(sorted({m.provider for m in pool}))})")

    # --- matrix (fixture replay, live build, or simulation) ---
    # The fixture stores the matrix AND the task prompts ("specs"), so a replay routes the exact
    # same tasks without re-sampling (hard prompts are sampled from LLMRouterBench and would
    # otherwise differ across processes).
    if use_fixture:
        if not fixture.exists():
            raise SystemExit(f"no fixture at {fixture}; run a live build first (drop --use-fixture).")
        raw = json.loads(fixture.read_text())
        if isinstance(raw, dict) and "matrix" in raw and "specs" in raw:
            matrix = Matrix.from_dict(raw["matrix"])
            specs = [TaskSpec(**s) for s in raw["specs"]]
        else:  # legacy fixture: raw matrix only — re-derive specs (fine for the fixed `suite`)
            matrix = Matrix.from_dict(raw)
            specs = [to_spec(t) for t in tasks if t.id in matrix.cells]
        print(f"replaying cached matrix from {fixture} ({len(matrix.task_order)} tasks).")
    else:
        if not dry_run:
            est = estimate_cost(tasks, pool, max_tokens)
            print(f"\n  estimated live spend for the matrix: ~${est:.2f} "
                  f"({len(tasks)*len(pool)} model calls, max_tokens={max_tokens})")
            if not assume_yes:
                if input("  proceed with live calls? [y/N] ").strip().lower() != "y":
                    raise SystemExit("aborted (use --dry-run to simulate, or --yes to skip prompt).")
        print("\nbuilding matrix" + (" (simulated)" if dry_run else " (live)")
              + f" with {workers} parallel workers…")
        matrix = build_matrix(settings, tasks, pool, max_tokens=max_tokens, dry_run=dry_run,
                              workers=workers)
        specs = [to_spec(t) for t in tasks if t.id in matrix.cells]
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(json.dumps(
            {"matrix": matrix.to_dict(), "specs": [asdict(s) for s in specs]}, indent=1))
        print(f"cached matrix -> {fixture}")
    return route_and_report(client, track=track, matrix=matrix, specs=specs, sliders=sliders,
                            curve_slider=curve_slider, epochs=epochs, workers=workers, outdir=outdir)
