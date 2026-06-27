"""Sandboxed execution + grading of model-generated code (the `code` track's scorer).

LiveCodeBench problems come in two shapes, each with `(input, output)` test cases:

- **stdin** (atcoder/codeforces style) — the program reads ``stdin`` and writes ``stdout``; we run
  the model's code as a script, feed the case input, and compare stdout (whitespace-normalised).
- **functional** (leetcode style) — a ``class Solution`` method named ``func_name``; we exec the
  model's code, JSON-parse each input line into an argument, call the method, and compare the
  return value to the expected (type-tolerant equality).

A problem scores **1.0 iff the code passes every evaluated case, else 0.0** — binary pass@1, the
standard LiveCodeBench metric. This is *real execution*, so it's the honest coding signal the
substring scorers in ``suite.py`` can't give.

**Sandboxing.** Each case runs in a fresh ``python -I`` (isolated-mode) subprocess in a scratch CWD
with a hard wall-clock ``timeout`` (the process is killed on expiry) and, on POSIX, a soft CPU-time
rlimit. Model code from frontier models solving competitive problems is low-risk, but it is still
*untrusted* — never point this at a network-connected or privileged environment. There is no
filesystem/network jail beyond process isolation; treat it as a benchmark harness, not a secure
sandbox.
"""

from __future__ import annotations

import base64
import json
import pickle
import re
import subprocess
import sys
import tempfile
import zlib

_FENCE = re.compile(r"```(?:python|py|python3)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


# --- decoding LiveCodeBench test payloads -----------------------------------------------------

def decode_tests(raw: str) -> list[dict]:
    """Parse a public/private test payload into ``[{input, output, testtype}, …]``.

    Public tests are plain JSON; private tests are usually ``base64(zlib(pickle(json_str)))``.
    Returns ``[]`` for an empty/unparseable payload rather than raising.
    """
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        pass
    try:
        blob = base64.b64decode(raw.encode("utf-8"))
        try:
            blob = zlib.decompress(blob)
        except zlib.error:
            pass
        try:
            obj = pickle.loads(blob)
        except (pickle.UnpicklingError, EOFError, AttributeError, TypeError, ValueError):
            obj = blob.decode("utf-8", "replace")
        return json.loads(obj) if isinstance(obj, str) else obj
    except Exception:  # noqa: BLE001 — a bad payload must not crash the suite
        return []


# --- extracting code from the model's answer --------------------------------------------------

def extract_code(text: str) -> str:
    """The last fenced ```python block (models explain, then give the final program); else the
    raw text if it already looks like code."""
    blocks = _FENCE.findall(text or "")
    if blocks:
        return blocks[-1].strip()
    t = (text or "").strip()
    # No fence — accept it only if it plausibly contains code, else treat as a non-answer.
    return t if ("def " in t or "class " in t or "import " in t or "print(" in t) else ""


# --- output comparison ------------------------------------------------------------------------

def _norm_stdout(s: str) -> str:
    lines = [ln.rstrip() for ln in (s or "").replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _tolerant_eq(a, b) -> bool:
    """Equality that tolerates list/tuple and int/float differences (for functional returns)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_tolerant_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_tolerant_eq(a[k], b[k]) for k in a)
    return a == b


# --- runners ----------------------------------------------------------------------------------

_FUNCTIONAL_HARNESS = """\
import json, sys
from typing import *          # List/Optional/... appear in leetcode signatures
import collections, math, heapq, bisect, itertools, functools, re, string
from collections import *
{user_code}

def _eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    return a == b

_inp = {input!r}
_exp = json.loads({output!r})
_args = [json.loads(l) for l in _inp.split(chr(10)) if l.strip() != ""]
try:
    if "class Solution" in {has_solution!r}:
        _res = getattr(Solution(), {func!r})(*_args)
    else:
        _res = {func}(*_args)
except Exception as _e:
    print("__FAIL__"); sys.exit(0)
print("__PASS__" if _eq(_res, _exp) else "__FAIL__")
"""


def _run(argv: list[str], stdin: str, timeout: float) -> tuple[bool, str]:
    """Run a child process; return (exited_cleanly, stdout). Never raises."""
    try:
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(argv, input=stdin, capture_output=True, text=True,
                                  timeout=timeout, cwd=cwd, preexec_fn=_limit)
        return proc.returncode == 0, proc.stdout
    except subprocess.TimeoutExpired:
        return False, ""
    except Exception:  # noqa: BLE001
        return False, ""


def _limit():  # pragma: no cover — POSIX only, best-effort CPU cap on the child
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (8, 9))
    except Exception:
        pass


def grade(code: str, tests: list[dict], *, testtype: str, func_name: str = "",
          starter_code: str = "", timeout: float = 6.0, max_cases: int = 12) -> float:
    """1.0 iff `code` passes every (capped) test case, else 0.0."""
    if not code or not tests:
        return 0.0
    cases = tests[:max_cases]
    if testtype == "functional":
        for c in cases:
            src = _FUNCTIONAL_HARNESS.format(
                user_code=code, input=c.get("input", ""), output=c.get("output", "null"),
                func=func_name, has_solution=("class Solution" if "class Solution" in code
                                              or "class Solution" in starter_code else ""))
            ok, out = _run([sys.executable, "-I", "-c", src], "", timeout)
            if not ok or "__PASS__" not in out:
                return 0.0
        return 1.0
    # stdin/stdout
    for c in cases:
        ok, out = _run([sys.executable, "-I", "-c", code], c.get("input", ""), timeout)
        if not ok or _norm_stdout(out) != _norm_stdout(c.get("output", "")):
            return 0.0
    return 1.0
