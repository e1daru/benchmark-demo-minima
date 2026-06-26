"""Google Gemini adapter (google-genai SDK).

Gemini 2.5/3 are *thinking* models: thinking tokens count against ``max_output_tokens`` and are
billed as output. With a tight cap the model can spend the whole budget thinking and return an
empty answer — so we (a) ask for a generous output budget and (b) first try with thinking disabled
(fine for flash tiers), falling back to thinking-allowed + a large budget if the answer is empty
(needed for the pro tiers, which can't fully disable thinking). Output tokens are billed as
``total - prompt`` (visible + thinking).
"""

from __future__ import annotations

import time

from . import ModelRun


def run(model_id: str, prompt: str, api_key: str, max_tokens: int) -> ModelRun:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    out_budget = max(max_tokens, 1500)

    def _call(cfg):
        return client.models.generate_content(model=model_id, contents=prompt, config=cfg)

    started = time.monotonic()
    try:
        resp = _call(types.GenerateContentConfig(
            max_output_tokens=out_budget,
            thinking_config=types.ThinkingConfig(thinking_budget=0)))
        if not (resp.text or "").strip():
            raise ValueError("empty answer (thinking likely required)")
    except Exception:
        # Pro tiers reject thinking_budget=0 or need to think first — give them ample room.
        resp = _call(types.GenerateContentConfig(max_output_tokens=max(out_budget, 8192)))

    latency = int((time.monotonic() - started) * 1000)
    um = resp.usage_metadata
    prompt_tok = getattr(um, "prompt_token_count", 0) or 0
    total_tok = getattr(um, "total_token_count", 0) or 0
    visible_tok = getattr(um, "candidates_token_count", 0) or 0
    output_tok = max(total_tok - prompt_tok, visible_tok)
    return ModelRun(text=resp.text or "", input_tokens=prompt_tok,
                    output_tokens=output_tok, latency_ms=latency)
