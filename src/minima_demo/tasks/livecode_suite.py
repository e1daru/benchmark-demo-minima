"""LiveCodeBench `code` track — real execution against held-out tests (the hardest coding signal).

The curated `suite.py` code tasks are scored by *substring/structure* checks: they reward a
plausible-looking solution, not a correct one, so 2026 models all pass and there's no routing gap.
This track instead pulls real problems from **LiveCodeBench** (``livecodebench/code_generation_lite``,
release_v6 — atcoder/leetcode/codeforces) and **runs the model's code against the problem's own test
cases** (see :mod:`code_exec`). A problem is solved only if the program actually passes every test —
so weak models genuinely fail and strong ones genuinely pass, and the per-model accuracy gap is real.

Problems (prompt + a capped set of test cases) are sampled once, stratified across the dataset's
easy/medium/hard labels, and cached to ``fixtures/livecode_problems.json``. That fixture is committed,
so the track is reproducible and runnable **without** the (large) upstream download — and a fresh
sample is only drawn when the fixture is absent.
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

from . import code_exec
from .suite import LiveTask

PROBLEMS_FIXTURE = Path("fixtures/livecode_problems.json")
_RAW_URL = ("https://huggingface.co/datasets/livecodebench/code_generation_lite/"
            "resolve/main/test6.jsonl")
# How many of each difficulty to sample when building the fixture (gap needs the easy↔hard spread).
_STRATA = {"easy": 4, "medium": 5, "hard": 5}
_MAX_TESTS = 12   # cap test cases per problem so live grading stays fast and the fixture stays small


# --- prompt construction (mirrors LiveCodeBench's own generation prompt) ----------------------

def _build_prompt(p: dict) -> str:
    head = ("You are an expert competitive programmer. Solve the problem below in Python 3. "
            "Think step by step, then give your final program as a single ```python code block.\n\n")
    if p["testtype"] == "functional":
        return (head + "Complete the solution class — return the full class.\n\n"
                f"### Question:\n{p['question_content']}\n\n"
                f"### Starter code:\n```python\n{p['starter_code']}\n```\n")
    return (head + "Read the input from standard input and write the answer to standard output.\n\n"
            f"### Question:\n{p['question_content']}\n")


def _scorer(p: dict):
    tests, ttype, func, starter = p["tests"], p["testtype"], p.get("func_name", ""), \
        p.get("starter_code", "")

    def score(text: str) -> float:
        return code_exec.grade(code_exec.extract_code(text or ""), tests, testtype=ttype,
                               func_name=func, starter_code=starter, max_cases=_MAX_TESTS)
    return score


# --- problem set (fixture replay, or one streamed sample from the upstream dataset) -----------

def _stream_problems(n_per: dict[str, int], seed: int) -> list[dict]:
    """Stream test6.jsonl line-by-line and reservoir-sample per difficulty (no full download)."""
    rng = random.Random(seed)
    seen: dict[str, int] = {k: 0 for k in n_per}
    keep: dict[str, list[dict]] = {k: [] for k in n_per}
    req = urllib.request.Request(_RAW_URL, headers={"User-Agent": "minima-demo/1.0"})
    buf = b""
    with urllib.request.urlopen(req, timeout=120) as r:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                rec = json.loads(line)
                diff = rec.get("difficulty")
                if diff not in n_per:
                    continue
                # Reservoir sampling: every record has an equal chance, deterministic for a seed.
                seen[diff] += 1
                bucket = keep[diff]
                if len(bucket) < n_per[diff]:
                    bucket.append(rec)
                else:
                    j = rng.randint(0, seen[diff] - 1)
                    if j < n_per[diff]:
                        bucket[j] = rec
            if all(len(keep[k]) >= n_per[k] for k in n_per) and all(seen[k] > 8 * n_per[k]
                                                                    for k in n_per):
                break  # enough draws for a representative sample; stop pulling the 134MB file

    out: list[dict] = []
    for diff, recs in keep.items():
        for rec in recs:
            pub = code_exec.decode_tests(rec.get("public_test_cases") or "")
            priv = code_exec.decode_tests(rec.get("private_test_cases") or "")
            tests = (pub + priv)[:_MAX_TESTS]
            if not tests:
                continue
            md = json.loads(rec.get("metadata") or "{}")
            out.append({
                "id": rec.get("question_id") or rec.get("question_title"),
                "title": rec.get("question_title"), "platform": rec.get("platform"),
                "difficulty": diff, "testtype": tests[0].get("testtype", "stdin"),
                "func_name": md.get("func_name", ""), "starter_code": rec.get("starter_code", ""),
                "question_content": rec.get("question_content", ""), "tests": tests,
            })
    return out


def _load_problems(strata: dict[str, int], seed: int) -> list[dict]:
    if PROBLEMS_FIXTURE.exists():
        return json.loads(PROBLEMS_FIXTURE.read_text())
    problems = _stream_problems(strata, seed)
    if not problems:
        raise SystemExit("could not sample LiveCodeBench problems (network?) and no cached fixture.")
    PROBLEMS_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    PROBLEMS_FIXTURE.write_text(json.dumps(problems, indent=1))
    return problems


def load_livecode_suite(n: int | None = None, seed: int = 7) -> list[LiveTask]:
    """Build executable LiveCodeBench tasks (from the committed fixture, or a fresh sample)."""
    problems = _load_problems(_STRATA, seed)
    if n:
        problems = problems[:n]
    # harder problems lean the slider toward quality; easy ones toward cost.
    slider = {"easy": 3.0, "medium": 6.0, "hard": 8.0}
    return [LiveTask(id=f"lcb#{p['id']}", prompt=_build_prompt(p), task_type="code",
                     quality_fn=_scorer(p), difficulty=p["difficulty"],
                     slider=slider.get(p["difficulty"], 6.0))
            for p in problems]
