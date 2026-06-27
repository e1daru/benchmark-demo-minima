# Example dashboards

Real captured runs (open the `report.html` files in a browser — they're self-contained, no server).
Numbers are from live `recommend → run → feedback` loops against `api.minima.sh`; regenerate your own
with `make bench-code` / `make bench-hard` / `make bench-catalog` / `make bench-dataset`.

## `code/` — the headline benchmark: 12 real models, **real code execution**

36 LiveCodeBench problems (atcoder + leetcode, stratified **12 easy / 12 medium / 12 hard**). Every
model's generated program is **actually run against the problem's test cases** — binary pass@1, no
heuristics, no judge. The difficulty spread is the point: it gives routing something to do.

- **Accuracy gap across models: 0.50** — best `gemini-3.5-flash` **0.81**, worst `gpt-4o` **0.31**.
  Price ≠ quality: cheap `gemini-3-flash-preview` (0.78) beats `gpt-4o`/`gemini-2.5-pro` at a fraction
  of the cost.
- **Routing earns its keep across the difficulty range** (the *accuracy-by-difficulty* chart):
  on **easy**, the cheapest model already scores 0.92 (just go cheap); on **medium**, cheapest
  collapses to **0.25** while Minima holds **0.83**; on **hard**, cheapest hits **0.00** while Minima
  reaches **0.58** (route up).
- Net result at the knee operating point: **78% cost saved vs all-premium**, **97% accuracy retained**,
  **+0.39 accuracy vs naive-cheapest**, margin to oracle **0.08**. Because easy tasks dominate the
  savings and hard tasks dominate the quality, Minima captures both — the core routing value.

## `hard/` — a difficulty-graded, multi-type frontier mix (67 tasks)

A wide easy→hard suite across task types, every model called for real:
**MATH-500** (the dataset's own levels 1–5), LLMRouterBench's aime / gpqa / livemathbench / mmlupro +
**Humanity's Last Exam**, and **IFEval** instruction-following (24 stdlib constraint checkers). 18 easy /
14 medium / 35 hard.

- **Routing across difficulty** (the *accuracy-by-difficulty* chart): on **easy** every policy ties
  (~0.89 — go cheap); on **hard** the cheapest model drops to **0.51** while Minima holds **0.79**
  (premium 0.86, oracle 0.89).
- At the knee operating point: **35% cost saved vs all-premium**, **96% accuracy retained**,
  **+0.16 vs naive-cheapest**, margin to oracle **0.05**. Price ≠ quality — cheap
  `gemini-3-flash-preview` (0.79) ties pricier flagships on this mix.
- **Online learning is visible here:** **62 of 134** routing decisions were driven by recalled memory
  (`basis=memory`), with routing spread across `claude-opus-4-8`, `gemini-2.5-flash` and others — not a
  single fixed pick.

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
