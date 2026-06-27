# Minima benchmark demo

A public, reproducible benchmark for **[Minima](https://docs.minima.sh)** — cost-aware LLM model
routing. It drives the public `minima-cli` SDK against the hosted `api.minima.sh` service and shows,
per task, how Minima trades **cost · latency · tokens · accuracy** against the **margin** to the
single most-effective model — and how it **improves within a run** as it learns from feedback.

Every run ends in a single self-contained `report.html` dashboard (opens offline).

## What it measures

For each task we record every candidate model's accuracy, cost, tokens, and latency, then compare
four policies:

| policy | meaning |
| --- | --- |
| **Minima** | the model Minima routed to (and the realized outcome it learned from) |
| **all-premium** | always use the strongest model (highest accuracy) — the "cost is no object" default |
| **cheapest** | always use the lowest-priced model |
| **oracle** | the per-task best model — the *perfect* router; Minima's **margin** is the gap to it |

Headline metrics: **cost saved vs all-premium**, **accuracy retained vs all-premium**, **margin to
oracle** (0 = perfect), **accuracy lift vs cheapest**, and **avg latency** — plus a **learning curve**
showing accuracy and savings climbing as Minima accumulates feedback over the task stream.

## Four tracks, one dashboard

- **`code` (live, headline)** — `bench-code`. Routes the **12 real hosted models** over
  **LiveCodeBench** problems (atcoder / leetcode / codeforces) and **actually runs each model's
  generated code against the problem's test cases** — binary pass@1, no heuristics, no LLM judge. This
  is the hardest, most honest coding signal: weak models write plausible code that *fails the tests*.
- **`hard` (live)** — `bench-catalog --hard`. Routes the 12 real models on **genuinely hard, verified
  problems** (LLMRouterBench's aime / gpqa / livemathbench / mmlupro + Humanity's Last Exam, scored
  against ground truth). Here the models really *differ*, so routing matters and the benchmark is
  informative.
- **`catalog` (live, easy)** — routes the 12 real models on a curated everyday-task suite with
  deterministic scorers. Useful to show the "a cheap model already suffices, so save cost" regime.
- **`dataset` (replay)** — routes over **LLMRouterBench** (public ACL-Findings benchmark) reusing its
  precomputed per-(prompt, model) scores/costs. Large-N, **zero model spend**, fully reproducible.
  Routes over the LLMRouterBench models that resolve to the hosted catalog (see
  [docs/methodology.md](docs/methodology.md) for the resolution log).

Every track caches its (task, model) matrix to a `fixtures/*.json` file, so any dashboard can be
regenerated later with **no keys and no spend**.

## Example results

Pre-rendered dashboards are in [`examples/`](examples/) — open the `report.html` files (self-contained,
no server). From real `recommend → run → feedback` loops against `api.minima.sh`:

- **code (live, 12 models — real execution, the headline):** on LiveCodeBench, with each model's code
  *actually run against the tests*, the models span a **0.64 accuracy gap** (best
  `gemini-3-flash-preview` **0.86**, worst `gpt-4o`/`gpt-4o-mini` **0.21**). Price ≠ quality —
  the top model costs **~10× less than `gpt-4o`**. Minima routes to it: **100% retention**, **margin
  to oracle 0.00**, **+0.50 accuracy vs naive-cheapest**.
- **hard (live, 12 models):** on aime/gpqa/livemathbench/mmlupro/**HLE** no single model dominates
  (best `gemini-3-flash-preview` **0.725**, worst **0.35**); the per-task **oracle hits 0.90** — a
  0.20 routing headroom above any one model. Minima **matches the best model (100% retention)** and
  beats naive-cheapest by **+0.25**.
- **catalog (live, easy):** **46% cost saved** vs all-premium at **100% accuracy retained** — the
  "cheap model suffices" regime; **52 of 69** decisions driven by learned memory.
- **dataset (LLMRouterBench replay, 3 models):** matches premium accuracy, **+16.7 pp vs cheapest**,
  **0.033 from oracle**.

## Quickstart

```bash
make setup                     # venv + install (pinned deps, incl. minima-cli[seed])
cp .env.example .env           # then fill in MUBIT_API_KEY + provider keys
make smoke                     # gate: health + recommend/feedback round-trip
make bench-code                # HEADLINE: LiveCodeBench, real code execution (asks before spending)
open results/*/report.html     # the dashboard
```

No keys handy / want a free run first:

```bash
make bench-catalog-dry         # full pipeline on a simulated matrix — no spend
make fetch-dataset             # one-time ~1.28GB LLMRouterBench download
make bench-dataset             # reproducible replay benchmark — no model spend
```

Re-render a dashboard offline from saved artifacts (no network, no keys):

```bash
make report RUN=results/<run-dir>
```

## How it uses Minima

It only depends on the published package — `pip install minima-cli[seed]` — and the hosted service:

- **SDK client** `minima_client.MinimaClient` → `recommend()` / `feedback()` / `savings()` / `models()`.
- **Dataset loaders** `minima.seeding.llmrouterbench` (the LLMRouterBench tarball) for the `hard`/`dataset`
  tracks, and the public `livecodebench/code_generation_lite` release for the `code` track.
- **Task corpus** `minima_harness.tasks.task_set` (seed tasks + the `grade_outcome` convention).

The LLMRouterBench *config constants* and the *baseline/oracle definitions* are vendored from the
Minima source (`tests/eval/`, not shipped in the wheel) with attribution — see `src/minima_demo/constants.py`
and `src/minima_demo/baselines.py`.

## Reproducibility

- Pinned `minima-cli==0.4.10` + provider SDK versions (`pyproject.toml`).
- Fixed seed → fixed task sampling/order; each run uses a **fresh memory namespace** (a true cold
  start, so the learning curve is real).
- Raw artifacts per run (`matrix.json`, `routed_*.jsonl`, `savings.json`, `results.json`) make the
  dashboard regenerable without re-running. Each track caches its (task, model) matrix to
  `fixtures/*.json`, and the `code` track also commits its sampled problems + test cases
  (`fixtures/livecode_problems.json`), so every live track is replayable with no keys and no spend.

See [docs/methodology.md](docs/methodology.md) for baseline math, the catalog↔dataset resolution, and
known caveats.
