"""The broadened `hard` track — one difficulty-graded, multi-type suite.

Combines three deterministic sources so the suite spans a wide easy→hard range *and* several task
types (the routing story Minima is built for: cheap on easy, escalate on hard):

- **MATH-500** (`math500_suite`) — competition math across the dataset's own levels 1–5
  (easy→hard gradient), boxed-answer scoring.
- **LLMRouterBench frontier sets** (`hard_suite.load_hard_suite`) — aime / gpqa / livemathbench /
  mmlupro / HLE, all genuinely hard, scored against ground truth.
- **IFEval** (`ifeval_suite`) — instruction-following with verifiable constraints (a distinct task
  type), difficulty proxied by constraint count.

Every task carries a `difficulty` label, so the dashboard's by-difficulty chart reads straight off
the mix. Sizes are tuned to ~60 tasks (≈ within the live-run budget across all 12 models).
"""

from __future__ import annotations

from .hard_suite import load_hard_suite
from .ifeval_suite import load_ifeval_suite
from .math500_suite import load_math500_suite
from .suite import LiveTask


def load_frontier_suite(seed: int = 7, *, llmrouterbench_per_dataset: int = 3) -> list[LiveTask]:
    tasks: list[LiveTask] = []
    tasks += load_math500_suite(seed=seed)                                    # ~30, easy→hard math
    tasks += load_hard_suite(per_dataset=llmrouterbench_per_dataset, seed=seed)  # ~15, frontier hard
    tasks += load_ifeval_suite(seed=seed)                                     # ~15, instruction
    return tasks
