# Methodology

## The routing protocol

Minima is a **recommender, not a proxy**. Per task the demo:

1. `recommend(task, cost_quality_tradeoff=slider, namespace, constraints=candidate_models)` → the
   chosen model + its predicted cost/latency/success and the full ranked list.
2. Looks up that model's **realized** result (accuracy, cost, tokens, latency) — from a live call
   (catalog track) or from LLMRouterBench (dataset track).
3. `feedback(recommendation_id, model, outcome, quality_score, tokens, cost, latency)` → writes the
   outcome to the namespace's memory, so the **next** recommendation is better-informed.

Streaming the tasks in a fixed order over a **fresh namespace** with feedback on is the **learning
curve** (cold → warm). Re-routing the warm namespace at each slider *without* feedback yields the
**Pareto operating points** without perturbing what was learned.

## Metrics & baselines

For task *t* and model *m* we hold `accuracy ∈ [0,1]`, `cost_usd`, `tokens`, `latency`. Over the task
set:

- **Minima** — the realized values of the model it routed to.
- **all-premium** — always the *strong model*: `argmax_m mean_t accuracy(t,m)`, ties broken toward the
  pricier model. (Standard strong-model baseline, cf. RouteLLM. We never let a strictly dominated
  model — pricier *and* weaker — be "premium".)
- **cheapest** — the lowest blended-list-price model.
- **oracle** — per task, `argmax_m accuracy(t,m)` with ties to lowest cost. The *perfect* router.
- **random** — uniform-over-models expectation.

Derived:

- **margin to oracle** = `mean_t (oracle_acc(t) − minima_acc(t))` — 0 means Minima matched the
  perfect per-task choice.
- **cost saved vs premium** = `(premium_total − minima_total) / premium_total`.
- **accuracy retention** = `minima_acc / premium_acc`.

Outcomes use Minima's own thresholds via `minima_harness.tasks.task_set.grade_outcome`
(`success ≥ 0.8`, `partial ≥ 0.4`, else `failure`).

## Hard track (the informative benchmark)

`bench-catalog --hard` sources **verified hard problems** from LLMRouterBench — `aime`,
`livemathbench` (competition math, answer in `\boxed{}`) and `gpqa`, `mmlupro` (hard MCQ,
`Answer: $LETTER`) — and runs them against the **live 12-model catalog**, scoring the model's output
against the dataset's ground truth (no LLM judge; the prompts carry their own answer format). Unlike
the easy `catalog` suite (where every 2026 model scores ~1.0), here the models span a large accuracy
gap (~0.65), and notably **price ≠ quality** on these tasks (a cheap `gemini-3-flash-preview` can beat a
pricier `gemini-2.5-pro`). That makes it the benchmark that actually exercises routing.

**Fair token budget.** Gemini "thinking" tokens are billed as output and count against
`max_output_tokens`, so an unbounded budget would both distort the cost axis and let one provider use
far more tokens than others. The Google adapter therefore caps total output at ~`max_tokens` with up to
half reserved for reasoning — comparable to the other providers.

**What it shows.** On hard tasks you cannot save money without losing accuracy (the best model *is* the
premium one), so "savings vs premium" is ~0 — the same honest result as the dataset track. The value is
**routing intelligence**: Minima matches the best single model (100% retention) and beats a naive
cheapest-everywhere policy by a wide margin, while the **oracle** (cheapest-correct model per task)
reveals large headroom — often higher accuracy at several-fold lower cost — which Minima narrows as it
accumulates feedback.

## Scoring (catalog track)

Deterministic `quality_fn` per task — exact/normalized substring, last-number extraction, or
structural code checks — so a correct-but-verbose answer still scores 1.0 and there is **no LLM
judge** in the loop. The suite mixes easy tasks (any cheap model wins) and hard ones (only strong
models win) so a real per-task best exists for the oracle and routing actually matters.

## Catalog ↔ dataset resolution (why the dataset pool is small)

Minima can only route over models in its **hosted catalog** (12 models as of the
`fallback-snapshot-2026-06` snapshot: Anthropic `claude-haiku-4-5 / sonnet-4-6 / opus-4-8`; Google
`gemini-2.5-flash-lite / 2.5-flash / 2.5-pro / 3-flash-preview / 3.1-flash-lite / 3.1-pro-preview /
3.5-flash`; OpenAI `gpt-4o-mini / gpt-4o`).

LLMRouterBench's 12 candidate models overlap this catalog on only three (exact or near-exact) ids:

| LLMRouterBench model | catalog id |
| --- | --- |
| `gemini-2.5-pro` | `gemini-2.5-pro` |
| `gemini-2.5-flash` | `gemini-2.5-flash` |
| `claude-sonnet-4` | `claude-sonnet-4-6` |

Dropped (not in the catalog, so unroutable): `gpt-5`, `qwen3-235b-a22b-2507`,
`qwen3-235b-a22b-thinking-2507`, `deepseek-r1-0528`, `kimi-k2-0905`, `deepseek-v3.1-terminus`,
`glm-4.6`, `deepseek-v3-0324`, `intern-s1`. We deliberately **do not** alias `gpt-5 → gpt-4o`
(different model — that would misattribute the dataset's scores).

Consequence: the **dataset track is the reproducible, large-N, zero-spend benchmark over 3 frontier
models**, and the **catalog track is the all-12-models, real-call benchmark**. They share the metric
schema and dashboard but are distinct routing universes.

## Known caveats

- **Easy-suite regime.** If the strongest model on a small/easy sample is also cheap, "savings vs
  premium" is honestly small — there is little to route around. The full task suite (with hard
  reasoning/code) restores the spread.
- **Tokens ≠ cost.** Routing saves *cost*, not necessarily *tokens*: a terse premium model can emit
  fewer tokens than a chatty cheaper model. "Tokens saved" can therefore be negative even when cost
  savings are large; cost is the metric that matters.
- **Gemini thinking tokens** are billed as output, so the Google adapter charges `total − prompt`
  tokens (visible + thinking), not just the visible candidate tokens.
- The **service-side `/v1/savings`** figure (saved as `savings.json`) is an independent cross-check
  on our locally computed savings; small differences are expected (different cost bases).
- **Memory-write dependency.** The learning curve only climbs when Minima's hosted memory accepts
  feedback writes. If `feedback()` returns `accepted=false` with `warnings=['memory_write_failed']`
  (a transient Mubit backend condition we hit during some runs), every recommendation falls back to
  `decision_basis='prior'` and the curve stays flat — the demo infrastructure is unaffected, but the
  learning story needs healthy writes. Check `accepted` in the feedback response; the committed
  example dashboards were captured when writes were healthy (memory drove most decisions).
- **Operating point.** Headline numbers are reported at the *knee*: the cheapest slider in Minima's
  sweep that still retains ≥90% of premium accuracy. Low sliders save more but trade quality; high
  sliders chase quality. The full dial is the Pareto chart.
