"""Thin provider adapters: run a model and report (text, tokens, latency).

Each adapter returns a uniform :class:`ModelRun`. Cost is computed by the caller from catalog
prices (a model's tokens × its $/Mtok), so token accounting lives here and pricing stays in one
place. Failures are captured as ``error`` rather than raised, so one flaky model never aborts a run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class ModelRun:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def call_model(
    provider: str,
    model_id: str,
    prompt: str,
    *,
    api_key: str,
    max_tokens: int = 768,
) -> ModelRun:
    """Dispatch to the right provider adapter. Never raises — errors land in ModelRun.error."""
    started = time.monotonic()
    try:
        if provider == "anthropic":
            from .anthropic import run as run_anthropic
            return run_anthropic(model_id, prompt, api_key, max_tokens)
        if provider == "google":
            from .google import run as run_google
            return run_google(model_id, prompt, api_key, max_tokens)
        if provider == "openai":
            from .openai import run as run_openai
            return run_openai(model_id, prompt, api_key, max_tokens)
        return ModelRun("", 0, 0, 0, error=f"unknown provider {provider!r}")
    except Exception as exc:  # noqa: BLE001 — adapters must degrade, not crash the suite
        return ModelRun("", 0, 0, int((time.monotonic() - started) * 1000),
                        error=f"{type(exc).__name__}: {exc}")
