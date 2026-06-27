# Example dashboards

Real captured runs (open the `report.html` files in a browser — they're self-contained, no server).
Numbers are from live `recommend → run → feedback` loops against `api.minima.sh`; regenerate your own
with `make bench-hard` / `make bench-catalog` / `make bench-dataset`.

## `hard/` — the headline benchmark, 12 real models on hard verified problems

20 problems from LLMRouterBench (aime / gpqa / livemathbench / mmlupro), scored against ground truth,
every one of the 12 catalog models called for real. This is where models actually differ:

- **Accuracy gap across models: 0.65** (best `claude-opus-4-8`/`claude-sonnet-4-6`/`gemini-3-flash-preview`
  ≈ 0.90, worst `gpt-4o-mini` ≈ 0.25). On these tasks price ≠ quality — e.g. cheap `gemini-3-flash-preview`
  (0.90) beats pricier `gemini-2.5-pro` (0.65).
- **Minima matches the best single model** (0.90, **100% retention**) and beats a naive
  cheapest-everywhere policy by **+45 points** (0.90 vs 0.45).
- **Margin to oracle 0.10**; the oracle reaches **1.00 at ~4× lower cost than premium** ($0.08 vs $0.34) —
  perfect per-task routing is both more accurate *and* cheaper, and Minima's accuracy climbs (0.80→0.87)
  as it accumulates feedback.

The honest read: on hard tasks you can't save money without losing quality, so the value is **routing
intelligence** — matching the best model and crushing naive-cheapest — plus visible headroom to the oracle.

## `catalog/` — live track, 12 real models (Anthropic · Google · OpenAI)

23 deterministically-scored tasks, every model called for real. Headline at the knee operating point
(cheapest slider retaining ≥90% premium accuracy):

- **46% cost saved vs all-premium** (`claude-opus-4-8`)
- **100% accuracy retained** vs premium
- **margin to oracle 0.022** (≈ perfect per-task routing)
- Minima learned online: **52 of 69** routing decisions were driven by recalled memory, not priors.

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
