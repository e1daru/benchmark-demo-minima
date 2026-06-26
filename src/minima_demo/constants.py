"""Benchmark constants — vendored from the Minima source tree (not shipped in the wheel).

The LLMRouterBench candidate set, premium pick, independent market prices, and dataset→task-type
map are lifted verbatim from ``minima/tests/eval/llmrouterbench_config.py`` (the design output of
the eval's Phase 2). They are *config*, not code: copying them keeps the demo dependent only on the
public ``minima-cli`` package while reusing the exact benchmark the Minima team validated against.

The DATASET_TO_CATALOG_ALIAS map is demo-specific: the hosted ``api.minima.sh`` catalog (12 current
models) only overlaps the dataset's frontier set on a few ids, so the dataset (replay) track routes
over the resolvable subset. See docs/methodology.md for the resolution log.
"""

from __future__ import annotations

# --- LLMRouterBench frontier suite (verbatim from llmrouterbench_config.py) -------------------

EVAL_DATASETS: tuple[str, ...] = (
    "aime", "arc-agi", "arenahard", "arenahard_coding", "arenahard_creative_writing",
    "arenahard_math", "gpqa", "hle", "livecodebench", "livemathbench",
    "mmlupro", "simpleqa", "swe-bench", "tau2",
)

CANDIDATES: tuple[str, ...] = (
    "gemini-2.5-pro", "gpt-5", "qwen3-235b-a22b-2507", "qwen3-235b-a22b-thinking-2507",
    "deepseek-r1-0528", "kimi-k2-0905", "deepseek-v3.1-terminus", "glm-4.6",
    "gemini-2.5-flash", "deepseek-v3-0324", "claude-sonnet-4", "intern-s1",
)

# "Always use the best model" baseline in the dataset (highest avg score on the common set).
PREMIUM: str = "gemini-2.5-pro"

# Independent market list prices (input_$/Mtok, output_$/Mtok) — the router DECIDES on these,
# never on the dataset's scored cost column (guard V2: no circularity).
MARKET_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 10.00), "gpt-5": (1.25, 10.00),
    "qwen3-235b-a22b-2507": (0.09, 0.60), "qwen3-235b-a22b-thinking-2507": (0.22, 0.88),
    "deepseek-r1-0528": (0.50, 2.15), "kimi-k2-0905": (0.60, 2.50),
    "deepseek-v3.1-terminus": (0.27, 1.00), "glm-4.6": (0.40, 1.75),
    "gemini-2.5-flash": (0.30, 2.50), "deepseek-v3-0324": (0.25, 0.88),
    "claude-sonnet-4": (3.00, 15.00), "intern-s1": (0.30, 1.00),
}

EVAL_DATASET_TASK_TYPE: dict[str, str] = {
    "aime": "reasoning", "arc-agi": "reasoning", "arenahard": "other",
    "arenahard_coding": "code", "arenahard_creative_writing": "creative",
    "arenahard_math": "reasoning", "gpqa": "qa", "hle": "qa", "livecodebench": "code",
    "livemathbench": "reasoning", "mmlupro": "qa", "simpleqa": "qa",
    "swe-bench": "code", "tau2": "tool_use",
}


def task_type_for(eval_name: str) -> str:
    """Dataset id -> Minima TaskType string."""
    return EVAL_DATASET_TASK_TYPE.get(eval_name, "other")


# --- Demo-specific: map dataset model ids onto hosted-catalog ids -----------------------------
# Only models that resolve to a real catalog id can be routed by the hosted service. We alias on
# exact-or-near identity only; we deliberately DO NOT alias gpt-5 -> gpt-4o (different model), so
# the dataset's gpt-5 scores are never misattributed.
DATASET_TO_CATALOG_ALIAS: dict[str, str] = {
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "claude-sonnet-4": "claude-sonnet-4-6",
}
