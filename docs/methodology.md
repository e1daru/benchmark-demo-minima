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

## Code track (`bench-code`) — real execution, the hardest coding signal

`bench-catalog --code` is the headline coding benchmark. It pulls real problems from
**LiveCodeBench** (`livecodebench/code_generation_lite`, release_v6 — atcoder / leetcode /
codeforces, stratified across the dataset's easy/medium/hard labels) and, for each model, **actually
runs the generated program against the problem's own test cases**. A problem scores **1.0 iff the
code passes every evaluated test, else 0.0** — binary pass@1, LiveCodeBench's own metric. There is no
substring or structure heuristic and no LLM judge: a solution counts only if it *runs correctly*.

Two problem shapes, each graded by execution (see `src/minima_demo/tasks/code_exec.py`):

- **stdin** (atcoder/codeforces) — run the code as a script, feed the case input on `stdin`, compare
  `stdout` (whitespace-normalised) to the expected output.
- **functional** (leetcode) — `exec` the code, JSON-parse each input line into an argument, call the
  `class Solution` method named by the problem's `func_name`, and compare the return value
  (type-tolerant equality: list/tuple and int/float differences are ignored).

Each case runs in a fresh `python -I` (isolated-mode) subprocess in a scratch directory under a hard
wall-clock timeout (the process is killed on expiry) plus a best-effort CPU rlimit on POSIX. Model
code from frontier models solving competitive problems is low-risk, but it is still *untrusted* — the
harness is process-isolated, **not** a security sandbox; don't run it network-connected or privileged.

The sampled problems (prompt + a capped set of test cases) are cached to
`fixtures/livecode_problems.json` — committed, so the track is reproducible and runnable **without**
the large upstream download; a fresh sample is drawn only when that fixture is absent. The results
matrix is cached to `fixtures/code_matrix.json` like the other tracks, so the dashboard regenerates
with no keys and no spend.

This is where the substring scorers can't follow: weak models emit plausible code that *fails the
tests*, strong models emit code that *passes*, so the per-model accuracy gap is real and earned.

## Hard track (`bench-hard`) — a difficulty-graded, multi-type frontier mix

`bench-catalog --hard` is a **wide easy→hard suite across several task types**, so routing actually
has something to do (cheap on easy, escalate on hard). It combines three deterministic, no-LLM-judge
sources (see `src/minima_demo/tasks/frontier_suite.py`):

- **MATH-500** (`HuggingFaceH4/MATH-500`) — competition math sampled across the dataset's own
  **difficulty levels 1–5** (mapped 1–2 → easy, 3 → medium, 4–5 → hard); the model boxes its final
  answer, graded by the numeric-aware `math_boxed`. This is the clean easy→hard gradient.
- **LLMRouterBench frontier sets** — `aime`, `livemathbench` (math, `\boxed{}`); `gpqa`, `mmlupro`
  (hard MCQ, `Answer: $LETTER`); and `hle` (**Humanity's Last Exam**, kept to its cleanly
  auto-gradable letter/numeric items). All genuinely hard, scored against ground truth.
- **IFEval** (`google/IFEval`) — instruction-following with **verifiable constraints** (word counts,
  keywords, formatting, casing), a distinct task type. We reimplement 24 of its 25 instruction
  checkers in the stdlib (`tasks/ifeval_checks.py`; only `language:response_language` is dropped) and
  sample only prompts whose every constraint is supported. The per-prompt score is the *fraction* of
  constraints satisfied (loose normalisation, à la the official metric). Difficulty is proxied by the
  number of simultaneous constraints (1 → easy, 2 → medium, ≥3 → hard).

Unlike the easy `catalog` suite (where every 2026 model scores ~1.0), the models span a large accuracy
gap here, and notably **price ≠ quality** (a cheap `gemini-3-flash-preview` can beat a pricier
`gemini-2.5-pro`). The dashboard's **accuracy-by-difficulty** chart reads straight off this mix. Every
checker (IFEval) and the boxed/MCQ scorers are unit-tested before any live run — a buggy checker would
understate every model.

**Fair token budget.** Gemini "thinking" tokens are billed as output and count against
`max_output_tokens`, so an unbounded budget would both distort the cost axis and let one provider use
far more tokens than others. The Google adapter therefore caps total output at ~`max_tokens` with up to
half reserved for reasoning — comparable to the other providers.

**What it shows.** Because the suite spans easy→hard, routing has real work to do and the savings story
returns: at the knee operating point Minima saved **~35% cost vs all-premium at ~96% retention**
(captured run, 67 tasks), going cheap where the cheapest model already suffices (easy: all policies
≈0.89) and escalating where it collapses (hard: cheapest 0.51 vs Minima 0.79). The value is **routing
intelligence**: match the best single model, beat naive-cheapest by a wide margin, and chase the
**oracle** (cheapest-correct model per task), which reveals further headroom. The *accuracy-by-difficulty*
chart makes this concrete.

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
- **Memory recall dependency.** Online learning shows only when Minima's hosted memory both *accepts
  writes* and *engages recall*. Writes are reliably `accepted=true`; recall is intermittent — hosted
  `agent_routed` recall runs close to its server-side timeout, so in some windows every decision falls
  back to `decision_basis='prior'` and the curve flattens (the demo infra is unaffected — only the
  learning story needs recall to land). In the committed runs recall engaged on the **`hard`** track
  (62 of 134 decisions were `basis=memory`, with routing spread across several models); the **`code`**
  track was prior-dominated (its cheap-but-strong pick was obvious from cold start). The gap, savings,
  and difficulty charts do not depend on recall.
- **Operating point.** Headline numbers are reported at the *knee*: the cheapest slider in Minima's
  sweep that still retains ≥90% of premium accuracy. Low sliders save more but trade quality; high
  sliders chase quality. The full dial is the Pareto chart.
