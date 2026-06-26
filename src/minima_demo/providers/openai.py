"""OpenAI adapter ("Codex" = OpenAI API models). Catalog ships gpt-4o / gpt-4o-mini."""

from __future__ import annotations

import time

from . import ModelRun


def run(model_id: str, prompt: str, api_key: str, max_tokens: int) -> ModelRun:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    started = time.monotonic()
    messages = [{"role": "user", "content": prompt}]
    try:
        resp = client.chat.completions.create(
            model=model_id, messages=messages, max_tokens=max_tokens,
        )
    except Exception:
        # Newer reasoning models reject max_tokens; retry with max_completion_tokens.
        resp = client.chat.completions.create(
            model=model_id, messages=messages, max_completion_tokens=max_tokens,
        )
    latency = int((time.monotonic() - started) * 1000)
    u = resp.usage
    return ModelRun(
        text=resp.choices[0].message.content or "",
        input_tokens=u.prompt_tokens,
        output_tokens=u.completion_tokens,
        latency_ms=latency,
    )
