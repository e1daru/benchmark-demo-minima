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

## Two tracks, one dashboard

- **`catalog` (live)** — routes over the **real hosted catalog** (12 current Anthropic / Google /
  OpenAI models) and **calls them with your keys**. Real cost/latency/tokens; deterministic scorers
  (no LLM judge). The (task, model) matrix is cached to `fixtures/catalog_matrix.json`, so the
  dashboard can later be regenerated with **no keys and no spend**.
- **`dataset` (replay)** — routes over **LLMRouterBench** (the public ACL-Findings benchmark), reusing
  its precomputed per-(prompt, model) scores and costs. Large-N, fully reproducible, **zero model
  spend**. Routes over the LLMRouterBench models that resolve to the hosted catalog (see
  [docs/methodology.md](docs/methodology.md) for the resolution log).

## Example results

Pre-rendered dashboards are in [`examples/`](examples/) — open the `report.html` files (self-contained,
no server). From real `recommend → run → feedback` loops against `api.minima.sh`:

- **catalog (live, 12 models):** **85% cost saved** vs all-premium (`claude-opus-4-8`), **100% accuracy
  retained**, **margin-to-oracle 0.022**; **18 of 23** routing decisions were driven by learned memory.
- **dataset (LLMRouterBench, hard):** **matches premium accuracy**, **+16.7 pp vs naive-cheapest**,
  **0.033 from oracle** — on hard tasks the value is the cost/quality dial, not free savings.

## Quickstart

```bash
make setup                     # venv + install (pinned deps, incl. minima-cli[seed])
cp .env.example .env           # then fill in MUBIT_API_KEY + provider keys
make smoke                     # gate: health + recommend/feedback round-trip
make bench-catalog             # LIVE: route over real models with your keys (asks before spending)
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
- **Dataset loader** `minima.seeding.llmrouterbench` (the LLMRouterBench tarball).
- **Task corpus** `minima_harness.tasks.task_set` (seed tasks + the `grade_outcome` convention).

The LLMRouterBench *config constants* and the *baseline/oracle definitions* are vendored from the
Minima source (`tests/eval/`, not shipped in the wheel) with attribution — see `src/minima_demo/constants.py`
and `src/minima_demo/baselines.py`.

## Reproducibility

- Pinned `minima-cli==0.4.10` + provider SDK versions (`pyproject.toml`).
- Fixed seed → fixed task sampling/order; each run uses a **fresh memory namespace** (a true cold
  start, so the learning curve is real).
- Raw artifacts per run (`matrix.json`, `routed_*.jsonl`, `savings.json`, `results.json`) make the
  dashboard regenerable without re-running. The committed `catalog` fixture makes the live track
  replayable with no keys.

See [docs/methodology.md](docs/methodology.md) for baseline math, the catalog↔dataset resolution, and
known caveats.
