"""Curated live task suite with deterministic [0,1] scorers — no LLM judge, fully reproducible.

The first three tasks are reused verbatim from the public ``minima_harness.tasks.task_set`` corpus;
the rest extend it across code / qa / reasoning / extraction / tool_use with a deliberate spread of
difficulty. The spread is the point: easy tasks any cheap model nails, hard ones only strong models
get right — so there is a real per-task best model for the oracle, and routing actually matters.

Scorers are tolerant of model verbosity (case-insensitive substring / last-number extraction /
structural checks) so a correct-but-chatty answer still scores 1.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from minima_harness.tasks.task_set import TASKS as HARNESS_TASKS

from ..spec import TaskSpec

QualityFn = Callable[[str], float]


@dataclass(frozen=True)
class LiveTask:
    id: str
    prompt: str
    task_type: str
    quality_fn: QualityFn
    difficulty: str = "medium"
    slider: float = 5.0


def to_spec(t: LiveTask) -> TaskSpec:
    return TaskSpec(id=t.id, prompt=t.prompt, task_type=t.task_type, difficulty=t.difficulty)


# --- scorer helpers ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def contains_all(*subs: str) -> QualityFn:
    needles = [s.lower() for s in subs]
    return lambda t: 1.0 if all(n in _norm(t) for n in needles) else 0.0


def contains_any(*subs: str) -> QualityFn:
    needles = [s.lower() for s in subs]
    return lambda t: 1.0 if any(n in _norm(t) for n in needles) else 0.0


def final_number(expected: float, tol: float = 1e-6) -> QualityFn:
    """1.0 iff the last number in the output equals ``expected`` (commas/$ tolerated)."""
    def score(t: str) -> float:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", (t or "").replace("$", ""))
        if not nums:
            return 0.0
        try:
            got = float(nums[-1].replace(",", ""))
        except ValueError:
            return 0.0
        return 1.0 if abs(got - expected) <= tol else 0.0
    return score


def code_has(*tokens: str) -> QualityFn:
    """1.0 if all tokens present, 0.5 if some — mirrors the harness's partial-credit code scorers."""
    toks = [s.lower() for s in tokens]
    def score(t: str) -> float:
        n = _norm(t)
        hit = sum(1 for x in toks if x in n)
        if hit == len(toks):
            return 1.0
        return 0.5 if hit else 0.0
    return score


def code_solution(defname: str, *mechanisms: str) -> QualityFn:
    """Structural code check tolerant of idioms: needs the function def, any one accepted
    mechanism, and some self-test (assert/unittest). Partial credit (0.5) if only the def is there.
    (Deterministic, no execution — alternative correct implementations all pass.)"""
    mechs = [m.lower() for m in mechanisms]
    def score(t: str) -> float:
        n = _norm(t)
        if defname.lower() not in n:
            return 0.0
        mech_ok = (not mechs) or any(m in n for m in mechs)
        test_ok = any(x in n for x in ("assert", "unittest", "doctest"))
        return 1.0 if (mech_ok and test_ok) else 0.5
    return score


def _from_harness() -> list[LiveTask]:
    out = []
    for t in HARNESS_TASKS:
        qf = t.quality_fn or (lambda _t: 0.5)
        out.append(LiveTask(id=t.label, prompt=t.prompt, task_type=t.task_type,
                            quality_fn=qf, slider=t.slider,
                            difficulty="hard" if t.slider >= 7 else "medium"))
    return out


# --- the suite --------------------------------------------------------------------------------

SUITE: list[LiveTask] = _from_harness() + [
    # extraction --------------------------------------------------------------------------------
    LiveTask("extract-email", "Reply with only the email address in: 'ping bob_42@mail.co please'.",
             "extraction", contains_all("bob_42@mail.co"), "easy", 2.0),
    LiveTask("extract-iso-date", "What is the ISO-8601 date for 'March 3, 2024'? Answer YYYY-MM-DD.",
             "extraction", contains_all("2024-03-03"), "easy", 2.0),
    LiveTask("extract-json-field", "Given {\"sku\":\"X9\",\"qty\":7}, reply with only the qty value.",
             "extraction", final_number(7), "easy", 2.0),
    # qa (factual) ------------------------------------------------------------------------------
    LiveTask("qa-capital-au", "What is the capital of Australia? One word.",
             "qa", contains_all("canberra"), "easy", 1.0),
    LiveTask("qa-element-symbol", "What is the chemical symbol for gold?",
             "qa", contains_any("au"), "easy", 1.0),
    LiveTask("qa-speed-light", "Speed of light in vacuum in m/s, to 3 significant figures?",
             "qa", contains_any("3.00", "2.998", "299792458", "3 x 10", "3×10"), "medium", 3.0),
    LiveTask("qa-shakespeare", "Who wrote the play 'Hamlet'? Surname only.",
             "qa", contains_all("shakespeare"), "easy", 1.0),
    # reasoning / math --------------------------------------------------------------------------
    LiveTask("math-trains", "A train goes 60 km in 45 min. Its speed in km/h? Give the number only.",
             "reasoning", final_number(80), "medium", 5.0),
    LiveTask("math-discount", "A $250 coat is 30% off, then 8% tax on the discounted price. "
             "Final price in dollars? Number only.", "reasoning", final_number(189.0, tol=0.5),
             "hard", 7.0),
    LiveTask("math-gsm", "Liam has 3 boxes of 12 pencils and gives away 17. How many remain? "
             "Number only.", "reasoning", final_number(19), "medium", 5.0),
    LiveTask("math-series", "Sum of integers from 1 to 100? Number only.",
             "reasoning", final_number(5050), "medium", 4.0),
    LiveTask("logic-ages", "Anna is twice Ben's age. In 5 years their ages sum to 40. "
             "Anna's age now? Number only.", "reasoning", final_number(20), "hard", 7.0),
    # code --------------------------------------------------------------------------------------
    LiveTask("code-fizzbuzz", "Write a Python fizzbuzz(n) function. Idiomatic.",
             "code", code_has("def fizzbuzz", "fizz", "buzz"), "medium", 5.0),
    LiveTask("code-reverse", "Write Python reverse_string(s) and an assert test.",
             "code", code_solution("def reverse_string", "[::-1]", "reversed"), "easy", 3.0),
    LiveTask("code-isprime", "Write Python is_prime(n) handling edge cases, with an assert.",
             "code", code_solution("def is_prime", "return"), "medium", 5.0),
    LiveTask("code-anagram", "Write Python is_anagram(a, b) ignoring case/space, with a test.",
             "code", code_solution("def is_anagram", "sorted", "counter", "collections"), "hard", 6.0),
    # summarization / classification / tool_use -------------------------------------------------
    LiveTask("classify-sentiment", "Classify sentiment as POSITIVE or NEGATIVE: "
             "'This product broke on day one and support ignored me.'",
             "classification", contains_all("negative"), "easy", 1.0),
    LiveTask("classify-lang", "What language is 'Bonjour, comment ça va?' One word in English.",
             "classification", contains_all("french"), "easy", 1.0),
    LiveTask("tool-args", "To call get_weather(city, units), produce ONLY a JSON object of arguments "
             "for 'weather in Paris in celsius'.", "tool_use",
             contains_all("paris", "celsius"), "medium", 4.0),
    LiveTask("summarize-keyword", "In one word, the main topic of: 'The committee debated tax "
             "policy for three hours.'", "summarization", contains_any("tax", "taxation"), "easy", 2.0),
]


def select(max_tasks: int | None = None, task_types: set[str] | None = None) -> list[LiveTask]:
    tasks = [t for t in SUITE if not task_types or t.task_type in task_types]
    return tasks[:max_tasks] if max_tasks else tasks
