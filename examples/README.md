# Example dashboards

Real captured runs (open the `report.html` files in a browser — they're self-contained, no server).
Numbers are from live `recommend → run → feedback` loops against `api.minima.sh`; regenerate your own
with `make bench-catalog` / `make bench-dataset`.

## `catalog/` — live track, 12 real models (Anthropic · Google · OpenAI)

23 deterministically-scored tasks, every model called for real. Headline at the knee operating point
(cheapest slider retaining ≥90% premium accuracy):

- **85% cost saved vs all-premium** (`claude-opus-4-8`)
- **100% accuracy retained** vs premium
- **margin to oracle 0.022** (≈ perfect per-task routing)
- Minima learned online: **18 of 23** routing decisions were driven by recalled memory, not priors.

The takeaway: on practical tasks a cheap model usually suffices, and Minima *learns which one* — keeping
flagship-level quality at a fraction of the cost.

## `dataset/` — replay track, LLMRouterBench (genuinely hard tasks)

60 prompts (aime / gpqa / mmlu-pro / arena-hard math+coding / simpleqa) over the 3 frontier models that
resolve to the hosted catalog (`gemini-2.5-pro`, `gemini-2.5-flash`, `claude-sonnet-4`), scored from the
benchmark's own ground truth:

- **matches premium accuracy** (retention 100% at the quality-matching operating point)
- **+16.7 pp accuracy vs naive-cheapest** — routing up on the hard prompts the cheap model fails
- **margin to oracle 0.033** — close to the perfect per-task router
- On hard tasks there is *no free lunch*: keeping quality means paying for strong models, so the value
  is the **cost/quality dial** (the Pareto chart) + beating a naive cheapest-everywhere policy.

> Note: the learning curve depends on Minima's hosted memory accepting feedback writes. During some later
> runs the backend returned `memory_write_failed` (a transient Mubit-side issue), which forces prior-only
> routing and a flat curve. These captured runs are from when writes were healthy. See
> [../docs/methodology.md](../docs/methodology.md).
