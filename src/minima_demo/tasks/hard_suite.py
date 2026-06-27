"""Hard live task suite — verified LLMRouterBench prompts, scored against ground truth.

The curated `suite.py` is too easy for 2026 frontier models (every model ~aces it, so there's no
routing signal). This suite instead pulls genuinely hard, auto-gradable problems from
LLMRouterBench and runs them against the *live* 12-model catalog:

- **aime, livemathbench** — competition math; the prompt asks for the answer in ``\\boxed{...}``,
  ground truth is the value. Scored by extracting the boxed answer and comparing (numeric-aware).
- **gpqa, mmlupro** — hard multiple choice; the prompt asks for ``Answer: $LETTER``, ground truth
  is the letter. Scored by extracting the chosen letter.

Weak models fail these and strong models solve them, so the per-model accuracy gap is large and
routing actually matters. No LLM judge — the prompts carry their own answer format, matching the
benchmark's own scoring.
"""

from __future__ import annotations

import os
import random
import re

from .suite import LiveTask

# dataset -> (Minima task_type, answer kind)
HARD_DATASETS: dict[str, tuple[str, str]] = {
    "aime": ("reasoning", "math"),
    "livemathbench": ("reasoning", "math"),
    "gpqa": ("qa", "mcq"),
    "mmlupro": ("qa", "mcq"),
    # Humanity's Last Exam — the hardest set here. Most items need an LLM judge (free-form symbolic
    # answers), so we keep only the cleanly auto-gradable ones: a single MCQ letter or a numeric
    # value, extracted from HLE's own "Answer: {...}" response format.
    "hle": ("qa", "hle"),
}


# --- deterministic scorers --------------------------------------------------------------------

def _extract_boxed(text: str) -> str:
    """Last \\boxed{...} content (brace-balanced); fall back to the last number in the text."""
    key = "\\boxed{"
    idx = text.rfind(key)
    if idx == -1:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        return nums[-1].replace(",", "") if nums else ""
    i, depth, out = idx + len(key), 1, []
    while i < len(text) and depth > 0:
        c = text[i]
        depth += (c == "{") - (c == "}")
        if depth > 0:
            out.append(c)
        i += 1
    return "".join(out)


def _norm_math(s: str) -> str:
    s = s.strip().strip("$").replace(" ", "").replace(",", "").replace("\\!", "")
    return s.replace("\\dfrac", "\\frac").replace("\\left", "").replace("\\right", "")


def math_boxed(ground_truth: str):
    g = _norm_math(ground_truth)
    def score(text: str) -> float:
        a = _norm_math(_extract_boxed(text or ""))
        if a and a == g:
            return 1.0
        try:
            return 1.0 if abs(float(a) - float(g)) < 1e-6 else 0.0
        except ValueError:
            return 0.0
    return score


def mcq_letter(ground_truth: str):
    g = (ground_truth or "").strip().upper()[:1]
    def score(text: str) -> float:
        t = text or ""
        hits = re.findall(r"[Aa]nswer\s*:?\s*\**\(?\s*([A-Ja-j])\b", t)
        if hits:
            return 1.0 if hits[-1].upper() == g else 0.0
        tail = re.findall(r"\b([A-J])\b", t[-80:])  # fall back to a trailing standalone letter
        return 1.0 if (tail and tail[-1].upper() == g) else 0.0
    return score


def _after_answer(text: str) -> str:
    """The text following the last 'Answer:' marker (HLE's response format), up to the next line
    or the 'Confidence:' field; the whole text if there is no marker."""
    marks = list(re.finditer(r"answer\s*:?\s*", text or "", re.I))
    seg = text[marks[-1].end():] if marks else (text or "")
    return re.split(r"\n|confidence", seg, maxsplit=1, flags=re.I)[0].strip()


def hle_answer(ground_truth: str):
    """Score an HLE item: MCQ-letter if the ground truth is one letter, else numeric match."""
    g = (ground_truth or "").strip()
    is_letter = len(g) == 1 and g.isalpha()
    try:
        target = float(g.replace(",", ""))
        numeric = True
    except ValueError:
        numeric = False

    def score(text: str) -> float:
        seg = _after_answer(text or "")
        if is_letter:
            hits = re.findall(r"[A-Ja-j]", seg)
            return 1.0 if (hits and hits[0].upper() == g.upper()) else 0.0
        if numeric:
            nums = re.findall(r"-?\d[\d,]*\.?\d*", seg.replace(",", ""))
            try:
                return 1.0 if nums and abs(float(nums[-1]) - target) < 1e-6 else 0.0
            except ValueError:
                return 0.0
        return 1.0 if g and g.lower() in seg.lower() else 0.0
    return score


def hle_scorable(ground_truth: str) -> bool:
    """Keep only HLE items we can grade deterministically: a single letter or a numeric value."""
    g = (ground_truth or "").strip()
    if len(g) == 1 and g.isalpha():
        return True
    try:
        float(g.replace(",", ""))
        return True
    except ValueError:
        return False


def _scorer(kind: str, ground_truth: str):
    if kind == "math":
        return math_boxed(ground_truth)
    if kind == "hle":
        return hle_answer(ground_truth)
    return mcq_letter(ground_truth)


# --- loader -----------------------------------------------------------------------------------

def load_hard_suite(datasets: tuple[str, ...] | None = None, per_dataset: int = 8,
                    seed: int = 7) -> list[LiveTask]:
    """Sample hard LLMRouterBench prompts (one tarball scan) into live tasks with bound scorers."""
    import minima.seeding.llmrouterbench as lr

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        tarball = lr.download_tarball()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"LLMRouterBench tarball not cached; run `make fetch-dataset` ({exc}).")

    ds = tuple(datasets) if datasets else tuple(HARD_DATASETS)
    rng = random.Random(seed)
    # One model's records are enough — we only need the prompt + ground truth per question.
    buckets: dict[str, dict[str, dict]] = {d: {} for d in ds}
    for r in lr.iter_raw_records(tarball_path=tarball, datasets=set(ds), models={"gemini-2.5-flash"}):
        d = r["dataset_id"]
        gt = r.get("ground_truth")
        if gt is None:
            continue
        if d == "hle" and not hle_scorable(str(gt)):  # skip free-form items needing an LLM judge
            continue
        buckets[d][str(r["index"])] = r

    tasks: list[LiveTask] = []
    for d in ds:
        ttype, kind = HARD_DATASETS[d]
        items = sorted(buckets[d].items(), key=lambda kv: (len(kv[0]), kv[0]))
        if len(items) > per_dataset:
            items = [items[i] for i in sorted(rng.sample(range(len(items)), per_dataset))]
        for idx, rec in items:
            tasks.append(LiveTask(
                id=f"{d}#{idx}", prompt=rec["prompt"], task_type=ttype,
                quality_fn=_scorer(kind, str(rec["ground_truth"])), difficulty="hard", slider=7.0))
    return tasks
