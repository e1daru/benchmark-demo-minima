"""Deterministic checkers for IFEval verifiable-instruction types (no LLM judge, no heavy deps).

IFEval (`google/IFEval`) pairs each prompt with an ``instruction_id_list`` and aligned ``kwargs``.
Each instruction is *machine-verifiable* (word counts, keyword presence, formatting, casing, …). We
reimplement a faithful subset of the official `instruction_following_eval` checkers using only the
stdlib — every type except ``language:response_language`` (which needs language detection). A prompt
is usable only if **all** its instruction ids are in :data:`SUPPORTED`.

Scoring follows IFEval's *loose* spirit: a constraint counts as satisfied if it holds for the raw
response **or** any of a few benign normalisations (strip a leading "Sure, here is…" line, drop a
trailing line, remove surrounding markdown ``*``). The per-prompt score is the **fraction of
constraints satisfied** (instruction-level accuracy) — partial credit that gives a smooth routing
signal rather than all-or-nothing. Each checker is unit-tested before any live run.
"""

from __future__ import annotations

import json
import re


# --- relation + normalisation helpers ---------------------------------------------------------

def _rel_ok(count: int, relation: str, n: int) -> bool:
    relation = (relation or "").strip().lower()
    if relation == "at least":
        return count >= n
    if relation == "less than":
        return count < n
    if relation == "at most":
        return count <= n
    if relation == "more than":
        return count > n
    return count == n  # "exactly" / unknown


def _words(text: str) -> list[str]:
    return text.split()


def _sentences(text: str) -> list[str]:
    # Split on sentence-final punctuation followed by whitespace; good enough for at-least/less-than.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _paragraphs(text: str) -> list[str]:
    # IFEval separates paragraphs with the markdown divider ``***`` (with optional surrounding ws).
    parts = re.split(r"\s*\*\s*\*\s*\*\s*", text.strip())
    return [p for p in parts if p.strip()]


# --- individual checkers: (response, kwargs) -> bool -------------------------------------------

def _kw_existence(r, kw):
    return all(re.search(re.escape(k), r, re.I) for k in kw.get("keywords", []))


def _kw_frequency(r, kw):
    kwd = kw.get("keyword", "")
    n = re.findall(re.escape(kwd), r, re.I)
    return _rel_ok(len(n), kw.get("relation", "at least"), kw.get("frequency", 1))


def _kw_forbidden(r, kw):
    return not any(re.search(r"\b" + re.escape(w) + r"\b", r, re.I)
                   for w in kw.get("forbidden_words", []))


def _kw_letter_frequency(r, kw):
    letter = (kw.get("letter") or "").lower()
    count = r.lower().count(letter) if letter else 0
    return _rel_ok(count, kw.get("let_relation", kw.get("relation", "at least")),
                   kw.get("let_frequency", kw.get("frequency", 1)))


def _len_words(r, kw):
    return _rel_ok(len(_words(r)), kw.get("relation", "at least"), kw.get("num_words", 0))


def _len_sentences(r, kw):
    return _rel_ok(len(_sentences(r)), kw.get("relation", "at least"), kw.get("num_sentences", 0))


def _len_paragraphs(r, kw):
    return len(_paragraphs(r)) == kw.get("num_paragraphs", 0)


def _len_nth_paragraph_first_word(r, kw):
    paras = [p for p in re.split(r"\n\s*\n", r.strip()) if p.strip()]
    n = kw.get("nth_paragraph", 1)
    if len(paras) < kw.get("num_paragraphs", n) or n < 1 or n > len(paras):
        return False
    first = paras[n - 1].strip().split()
    return bool(first) and first[0].strip(".,!?;:'\"").lower() == \
        (kw.get("first_word") or "").lower()


def _content_placeholders(r, kw):
    return _rel_ok(len(re.findall(r"\[.+?\]", r)), "at least", kw.get("num_placeholders", 0))


def _content_postscript(r, kw):
    marker = (kw.get("postscript_marker") or "P.S.").lower()
    return marker.lower() in r.lower()


def _fmt_bullets(r, kw):
    bullets = re.findall(r"^\s*[\*\-]\s+\S", r, re.M)
    return len(bullets) == kw.get("num_bullets", 0)


def _fmt_constrained_response(r, kw):
    options = ("my answer is yes.", "my answer is no.", "my answer is maybe.")
    return r.strip().lower() in options


def _fmt_highlighted(r, kw):
    hl = re.findall(r"\*[^\*\n]+\*", r) + re.findall(r"\*\*[^\*\n]+\*\*", r)
    return _rel_ok(len(hl), "at least", kw.get("num_highlights", 0))


def _fmt_multiple_sections(r, kw):
    spliter = kw.get("section_spliter", "Section")
    count = len(re.findall(re.escape(spliter) + r"\s*\d+", r, re.I))
    return count >= kw.get("num_sections", 0)


def _fmt_json(r, kw):
    s = r.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s).strip()
    try:
        json.loads(s)
        return True
    except (ValueError, TypeError):
        return False


def _fmt_title(r, kw):
    return bool(re.search(r"<<[^\n]+>>", r))


def _comb_repeat_prompt(r, kw):
    rep = (kw.get("prompt_to_repeat") or "").strip()
    return bool(rep) and r.strip().lower().startswith(rep.lower())


def _comb_two_responses(r, kw):
    parts = [p for p in r.split("******") if p.strip()]
    return len(parts) == 2


def _startend_end(r, kw):
    phrase = (kw.get("end_phrase") or "").strip().lower()
    return bool(phrase) and r.strip().lower().endswith(phrase)


def _startend_quotation(r, kw):
    s = r.strip()
    return len(s) >= 2 and s.startswith('"') and s.endswith('"')


def _case_lower(r, kw):
    return r == r.lower()


def _case_upper(r, kw):
    return r == r.upper()


def _case_capital_word_frequency(r, kw):
    caps = [w for w in _words(r) if len(w) > 1 and w.isupper()]
    return _rel_ok(len(caps), kw.get("capital_relation", "at least"),
                   kw.get("capital_frequency", 1))


def _punct_no_comma(r, kw):
    return "," not in r


REGISTRY = {
    "keywords:existence": _kw_existence,
    "keywords:frequency": _kw_frequency,
    "keywords:forbidden_words": _kw_forbidden,
    "keywords:letter_frequency": _kw_letter_frequency,
    "length_constraints:number_words": _len_words,
    "length_constraints:number_sentences": _len_sentences,
    "length_constraints:number_paragraphs": _len_paragraphs,
    "length_constraints:nth_paragraph_first_word": _len_nth_paragraph_first_word,
    "detectable_content:number_placeholders": _content_placeholders,
    "detectable_content:postscript": _content_postscript,
    "detectable_format:number_bullet_lists": _fmt_bullets,
    "detectable_format:constrained_response": _fmt_constrained_response,
    "detectable_format:number_highlighted_sections": _fmt_highlighted,
    "detectable_format:multiple_sections": _fmt_multiple_sections,
    "detectable_format:json_format": _fmt_json,
    "detectable_format:title": _fmt_title,
    "combination:repeat_prompt": _comb_repeat_prompt,
    "combination:two_responses": _comb_two_responses,
    "startend:end_checker": _startend_end,
    "startend:quotation": _startend_quotation,
    "change_case:english_lowercase": _case_lower,
    "change_case:english_capital": _case_upper,
    "change_case:capital_word_frequency": _case_capital_word_frequency,
    "punctuation:no_comma": _punct_no_comma,
}
SUPPORTED = frozenset(REGISTRY)  # `language:response_language` deliberately excluded (needs langdetect)


# --- loose scoring -----------------------------------------------------------------------------

def _variants(response: str) -> list[str]:
    """A few benign normalisations (IFEval's loose spirit) to avoid false negatives."""
    r = response or ""
    out = {r, r.strip()}
    lines = [ln for ln in r.split("\n")]
    if len(lines) > 1:
        out.add("\n".join(lines[1:]).strip())   # drop a leading "Sure, here is…" line
        out.add("\n".join(lines[:-1]).strip())  # drop a trailing line
    out.add(r.replace("*", "").strip())          # remove markdown emphasis
    return [v for v in out if v]


def supported(instruction_ids: list[str]) -> bool:
    return all(i in SUPPORTED for i in instruction_ids)


def score(response: str, instruction_ids: list[str], kwargs_list: list[dict]) -> float:
    """Fraction of constraints satisfied (loose). 0.0 if there are no checkable constraints."""
    if not instruction_ids:
        return 0.0
    variants = _variants(response)
    ok = 0
    for iid, kw in zip(instruction_ids, kwargs_list):
        fn = REGISTRY.get(iid)
        if fn is None:
            continue
        kw = kw or {}
        if any(_safe(fn, v, kw) for v in variants):
            ok += 1
    return ok / len(instruction_ids)


def _safe(fn, response, kw) -> bool:
    try:
        return bool(fn(response, kw))
    except Exception:  # noqa: BLE001 — a checker must never crash the suite
        return False
