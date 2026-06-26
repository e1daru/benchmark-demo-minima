"""Runtime configuration: env loading, the Minima SDK client factory, and run parameters.

Everything the demo needs from the environment is funnelled through :func:`load_settings`, so the
rest of the code never touches ``os.environ`` directly. The only hosted dependency is the public
``minima-cli`` client pointed at ``MINIMA_BASE_URL`` with ``MUBIT_API_KEY`` as the bearer token.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass

from dotenv import load_dotenv
from minima_client import MinimaClient

# Catalog provider id -> the env var holding that provider's API key.
PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# The cost/quality slider points we sweep for the Pareto frontier (0 = cheapest, 10 = best).
DEFAULT_SLIDERS: tuple[float, ...] = (1.0, 3.0, 5.0, 7.0, 9.0)
# The slider used for the streamed learning curve. Cost-leaning on purpose: at a low slider Minima
# is rewarded for finding the *cheapest model that still succeeds*, so as it learns, cost falls and
# accuracy holds — the "improves over time" story. (The full slider range is the Pareto sweep.)
DEFAULT_CURVE_SLIDER: float = 2.0


@dataclass(frozen=True)
class Settings:
    minima_base_url: str
    mubit_api_key: str
    provider_keys: dict[str, str]  # provider id -> key, only providers with a non-empty key
    seed: int = 7
    timeout_s: float = 60.0

    def has_provider(self, provider: str) -> bool:
        return bool(self.provider_keys.get(provider))

    @property
    def live_providers(self) -> list[str]:
        return sorted(self.provider_keys)


def load_settings(seed: int = 7) -> Settings:
    """Read ``.env`` (if present) + environment into a frozen Settings object."""
    load_dotenv()
    base = os.environ.get("MINIMA_BASE_URL", "https://api.minima.sh").rstrip("/")
    key = os.environ.get("MUBIT_API_KEY")
    if not key:
        raise SystemExit("MUBIT_API_KEY is not set — copy .env.example to .env and fill it in.")
    provider_keys = {
        prov: os.environ[env]
        for prov, env in PROVIDER_KEY_ENV.items()
        if os.environ.get(env)
    }
    return Settings(minima_base_url=base, mubit_api_key=key, provider_keys=provider_keys, seed=seed)


def make_client(settings: Settings) -> MinimaClient:
    """A configured public-SDK client. Auth is the Mubit key passed through as a bearer token."""
    return MinimaClient(settings.minima_base_url, api_key=settings.mubit_api_key,
                        timeout=settings.timeout_s)


def fresh_namespace(track: str) -> str:
    """A unique, empty memory lane per run → a true cold start for the learning curve.

    The curve's *shape* is reproducible for a fixed seed + task order even though the namespace
    string differs each run (memory is keyed by namespace, so a new one starts with no evidence).
    """
    return f"demo-{track}-{secrets.token_hex(3)}"
