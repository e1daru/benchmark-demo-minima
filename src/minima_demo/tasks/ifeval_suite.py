"""IFEval instruction-following track — deterministic constraint checking (a distinct task type).

Pulls `google/IFEval` (541 prompts, each with verifiable formatting/length/keyword constraints) and
scores responses with the stdlib checkers in :mod:`ifeval_checks` — no LLM judge. Only prompts whose
every constraint is supported are kept; difficulty is proxied by the number of constraints
(1 → easy, 2 → medium, ≥3 → hard), giving the suite an instruction-following dimension across the
easy→hard range. Sampled prompts are cached to ``fixtures/ifeval_problems.json`` for offline replay.
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

from . import ifeval_checks
from .suite import LiveTask

PROBLEMS_FIXTURE = Path("fixtures/ifeval_problems.json")
_URL = "https://huggingface.co/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl"
# constraint count -> difficulty (a reasonable proxy: more simultaneous constraints = harder)
_STRATA = {"easy": 5, "medium": 5, "hard": 5}


def _difficulty(n_constraints: int) -> str:
    return "easy" if n_constraints <= 1 else ("medium" if n_constraints == 2 else "hard")


def _scorer(ids: list[str], kwargs_list: list[dict]):
    def score(text: str) -> float:
        return ifeval_checks.score(text or "", ids, kwargs_list)
    return score


def _fetch_problems(strata: dict[str, int], seed: int) -> list[dict]:
    req = urllib.request.Request(_URL, headers={"User-Agent": "minima-demo/1.0"})
    data = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    rows = [json.loads(l) for l in data.splitlines() if l.strip()]
    # keep only fully-supported prompts, bucket by difficulty proxy
    buckets: dict[str, list[dict]] = {k: [] for k in strata}
    for r in rows:
        ids = r.get("instruction_id_list", [])
        if not ids or not ifeval_checks.supported(ids):
            continue
        diff = _difficulty(len(ids))
        buckets[diff].append({
            "id": r.get("key"), "prompt": r["prompt"], "difficulty": diff,
            "instruction_id_list": ids, "kwargs": r.get("kwargs", [{} for _ in ids]),
        })
    rng = random.Random(seed)
    out: list[dict] = []
    for diff, n in strata.items():
        b = buckets[diff]
        rng.shuffle(b)
        out.extend(b[:n])
    return out


def _load_problems(strata: dict[str, int], seed: int) -> list[dict]:
    if PROBLEMS_FIXTURE.exists():
        return json.loads(PROBLEMS_FIXTURE.read_text())
    problems = _fetch_problems(strata, seed)
    if not problems:
        raise SystemExit("could not fetch IFEval (network?) and no cached fixture.")
    PROBLEMS_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    PROBLEMS_FIXTURE.write_text(json.dumps(problems, indent=1))
    return problems


def load_ifeval_suite(n: int | None = None, seed: int = 7) -> list[LiveTask]:
    problems = _load_problems(_STRATA, seed)
    if n:
        problems = problems[:n]
    slider = {"easy": 2.0, "medium": 4.0, "hard": 6.0}
    # Minima's TaskType enum has no "instruction-following" member, so we route these as "other"
    # (the designated catch-all). The id prefix `ifeval#` still identifies them in the artifacts.
    return [LiveTask(id=f"ifeval#{p['id']}", prompt=p["prompt"], task_type="other",
                     quality_fn=_scorer(p["instruction_id_list"], p["kwargs"]),
                     difficulty=p["difficulty"], slider=slider.get(p["difficulty"], 4.0))
            for p in problems]
