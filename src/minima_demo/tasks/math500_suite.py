"""MATH-500 track — a clean easy→hard difficulty gradient (the dataset's own `level` 1–5).

`HuggingFaceH4/MATH-500` ships 500 competition-math problems each tagged with a difficulty `level`
1–5 and a ground-truth `answer`. We sample across all five levels (mapping 1–2 → easy, 3 → medium,
4–5 → hard), ask the model to box its final answer, and grade with the existing numeric-aware
:func:`hard_suite.math_boxed`. This is the cleanest single source of a *wide difficulty range* in one
task type, and the dashboard's by-difficulty chart reads straight off it. Cached to
``fixtures/math500_problems.json`` for offline replay.
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

from .hard_suite import math_boxed
from .suite import LiveTask

PROBLEMS_FIXTURE = Path("fixtures/math500_problems.json")
_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
_PER_LEVEL = {1: 6, 2: 6, 3: 6, 4: 6, 5: 6}  # 30 problems spanning the full difficulty range
_LEVEL_DIFFICULTY = {1: "easy", 2: "easy", 3: "medium", 4: "hard", 5: "hard"}

_PROMPT = ("Solve the following math problem. Reason step by step, then give ONLY the final answer "
           "inside \\boxed{{}}.\n\n{problem}")


def _scorer(answer: str):
    fn = math_boxed(answer)
    return lambda text: fn(text or "")


def _fetch_problems(per_level: dict[int, int], seed: int) -> list[dict]:
    req = urllib.request.Request(_URL, headers={"User-Agent": "minima-demo/1.0"})
    data = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    rows = [json.loads(l) for l in data.splitlines() if l.strip()]
    by_level: dict[int, list[dict]] = {lvl: [] for lvl in per_level}
    for r in rows:
        try:
            lvl = int(r.get("level"))
        except (TypeError, ValueError):
            continue
        if lvl in by_level and r.get("answer"):
            by_level[lvl].append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    for lvl, n in per_level.items():
        b = by_level[lvl]
        rng.shuffle(b)
        for r in b[:n]:
            out.append({
                "id": (r.get("unique_id") or f"L{lvl}").replace("test/", "").replace(".json", ""),
                "level": lvl, "difficulty": _LEVEL_DIFFICULTY[lvl],
                "problem": r["problem"], "answer": str(r["answer"]), "subject": r.get("subject"),
            })
    return out


def _load_problems(per_level: dict[int, int], seed: int) -> list[dict]:
    if PROBLEMS_FIXTURE.exists():
        return json.loads(PROBLEMS_FIXTURE.read_text())
    problems = _fetch_problems(per_level, seed)
    if not problems:
        raise SystemExit("could not fetch MATH-500 (network?) and no cached fixture.")
    PROBLEMS_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    PROBLEMS_FIXTURE.write_text(json.dumps(problems, indent=1))
    return problems


def load_math500_suite(n: int | None = None, seed: int = 7) -> list[LiveTask]:
    problems = _load_problems(_PER_LEVEL, seed)
    if n:
        problems = problems[:n]
    slider = {"easy": 3.0, "medium": 5.0, "hard": 7.0}
    return [LiveTask(id=f"math500#{p['id']}", prompt=_PROMPT.format(problem=p["problem"]),
                     task_type="reasoning", quality_fn=_scorer(p["answer"]),
                     difficulty=p["difficulty"], slider=slider.get(p["difficulty"], 5.0))
            for p in problems]
