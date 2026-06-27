# Example dashboards

Real captured runs (open the `report.html` files in a browser — they're self-contained, no server).
Numbers are from live `recommend → run → feedback` loops against `api.minima.sh`; regenerate your own
with `make bench-code` / `make bench-hard` / `make bench-catalog` / `make bench-dataset`.

## `code/` — the headline benchmark: 12 real models, **real code execution**

14 LiveCodeBench problems (atcoder + leetcode, stratified easy/medium/hard). Every model's generated
program is **actually run against the problem's test cases** — binary pass@1, no heuristics, no judge.
This is where models truly differ on code, and where the "pricier = better" assumption breaks:

- **Accuracy gap across models: 0.64** — best `gemini-3-flash-preview` **0.86**, worst `gpt-4o` /
  `gpt-4o-mini` **0.21**. Price ≠ quality: `gemini-3-flash-preview` tops the board at **~10× less
  cost than `gpt-4o`**, which lands near the bottom.
- **Minima routes to that cheap-yet-best model**: **100% accuracy retention**, **margin to oracle
  0.00** (it matches the perfect per-task choice exactly), and **+0.50 accuracy vs naive-cheapest**
  (0.86 vs 0.36).
- Because the best model is *also* one of the cheapest, "cost saved vs premium" is ~0 here — the
  premium baseline is already cheap. The story isn't squeezing cost; it's **routing intelligence**:
  Minima identifies, from a cold start, the model that is simultaneously top-accuracy and low-cost.

> Learning curve: in this capture the hosted router served every decision from its **prior**
> (`basis=prior`) — feedback was accepted/written, but recall did not change the pick, so the curve
> is flat (the prior already chose the optimal model). The visible online-learning *climb* is in the
> `catalog/` run below, captured when memory recall was engaging. See
> [../docs/methodology.md](../docs/methodology.md).

## `hard/` — 12 real models on hard verified problems (math / science / HLE)

40 problems from LLMRouterBench (aime / gpqa / livemathbench / mmlupro + **Humanity's Last Exam**),
scored against ground truth, every one of the 12 catalog models called for real. HLE is brutal enough
to pull even the strongest models down, so absolute accuracy is low — which is the point:

- **No single model dominates.** Best `gemini-3-flash-preview` **0.725**, worst `gpt-4o-mini` **0.35**
  (gap 0.375). Price ≠ quality again — the cheap `gemini-3-flash-preview` (0.725) **tops** pricier
  `claude-opus-4-8` (0.70) and `gemini-3.1-pro-preview` (0.50).
- **The per-task oracle reaches 0.90** — far above any single model's 0.725. That **0.20 gap is pure
  routing headroom**: picking the right model per task beats picking any one model everywhere.
- **Minima matches the best single model** (0.70, **100% retention** vs premium `claude-opus-4-8`) and
  beats naive-cheapest by **+0.25** (0.70 vs 0.45), at **margin 0.20** to the oracle.

The honest read: on truly hard problems you can't save money without losing quality (the best models
*are* the costly ones), so the value is **routing intelligence** — matching the best single model,
crushing naive-cheapest, and chasing an oracle ceiling no single model reaches.

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
