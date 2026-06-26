"""Anthropic adapter — generalizes minima/examples/06_routed_llm_call.py."""

from __future__ import annotations

import time

from . import ModelRun


def run(model_id: str, prompt: str, api_key: str, max_tokens: int) -> ModelRun:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    started = time.monotonic()
    # Stream so long outputs never hit a request timeout, then collect the final message.
    with client.messages.stream(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    return ModelRun(
        text=text,
        input_tokens=msg.usage.input_tokens,
        output_tokens=msg.usage.output_tokens,
        latency_ms=int((time.monotonic() - started) * 1000),
    )
