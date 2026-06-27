"""Command-line entry point: smoke · resolve · bench-catalog · bench-dataset · report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import report as report_mod
from .catalog import cost_spread, fetch_catalog, resolve_dataset_pool, resolve_live_pool
from .config import load_settings, make_client


def _providers_arg(v: str | None) -> set[str] | None:
    return {p.strip() for p in v.split(",")} if v else None


def cmd_smoke(args) -> int:
    """Health + one recommend->feedback round-trip: the 'Minima is all set' gate."""
    s = load_settings()
    c = make_client(s)
    health = c.health()
    print(json.dumps(health, indent=2))
    if health.get("status") != "ok":
        print("health not ok", file=sys.stderr)
        return 1
    rec = c.recommend({"task": "Write a haiku about caching.", "task_type": "creative"},
                      cost_quality_tradeoff=3.0, namespace="demo-smoke")
    rm = rec.recommended_model
    print(f"\nrecommend OK -> {rm.model_id} ({rm.provider}) "
          f"basis={rec.decision_basis} est=${rm.est_cost_usd:.5f}")
    fb = c.feedback(rec.recommendation_id, rm.model_id, "success", quality_score=0.9,
                    input_tokens=20, output_tokens=40, actual_cost_usd=rm.est_cost_usd, latency_ms=800)
    print(f"feedback OK -> accepted={fb.accepted}")
    print("\n✓ Minima is all set.")
    return 0


def cmd_resolve(args) -> int:
    """Print the live catalog and the resolved candidate pools for both tracks."""
    s = load_settings()
    catalog = fetch_catalog(make_client(s))
    print(f"catalog: {len(catalog)} models")
    for m in sorted(catalog, key=lambda x: (x.provider, x.output_cost_per_mtok)):
        print(f"  {m.model_id:24s} {m.provider:10s} "
              f"in=${m.input_cost_per_mtok:6.3f} out=${m.output_cost_per_mtok:6.3f}/Mtok")

    live = resolve_live_pool(catalog, s)
    live_ids = [m.model_id for m in live]
    print(f"\nlive pool ({len(live)} models; providers with keys: {s.live_providers}):")
    print(f"  {live_ids}")
    print(f"  cost spread (priciest/cheapest output): {cost_spread(catalog, live_ids):.1f}x")

    ds = resolve_dataset_pool(catalog)
    print(f"\ndataset pool ({len(ds)} of {12} LLMRouterBench models resolve to the catalog):")
    for ds_id, cat_id in ds:
        print(f"  {ds_id:20s} -> {cat_id}")
    from .constants import CANDIDATES
    dropped = [m for m in CANDIDATES if m not in dict(ds)]
    print(f"  dropped (not in catalog): {dropped}")
    return 0


def cmd_bench_catalog(args) -> int:
    from .catalog_track import run_catalog
    s = load_settings(seed=args.seed)
    run_catalog(s, max_tasks=args.max_tasks, providers=_providers_arg(args.providers),
                max_tokens=args.max_tokens, dry_run=args.dry_run, use_fixture=args.use_fixture,
                assume_yes=args.yes, epochs=args.epochs, hard=args.hard,
                hard_per_dataset=args.hard_per_dataset)
    return 0


def cmd_bench_dataset(args) -> int:
    from .dataset_track import run_dataset
    s = load_settings(seed=args.seed)
    run_dataset(s, max_tasks=args.max_tasks, datasets=_providers_arg(args.datasets),
                per_dataset=args.per_dataset, rebuild=args.rebuild, epochs=args.epochs)
    return 0


def cmd_report(args) -> int:
    """Render a dashboard offline. A run directory is re-assembled (re-applies metric logic);
    a results.json is rendered as-is."""
    p = Path(args.results)
    if p.is_dir():
        results = report_mod.reassemble_dir(p)
        (p / "results.json").write_text(json.dumps(results, indent=2))
        out = p / "report.html"
    else:
        results = json.loads(p.read_text())
        out = p.with_name("report.html")
    report_mod.render(results, out)
    print(f"rendered {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="minima-demo", description="Minima benchmark demo")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke", help="health + recommend/feedback round-trip").set_defaults(fn=cmd_smoke)
    sub.add_parser("resolve", help="print catalog + resolved pools").set_defaults(fn=cmd_resolve)

    bc = sub.add_parser("bench-catalog", help="live track over the real 12-model catalog")
    bc.add_argument("--max-tasks", type=int, default=None)
    bc.add_argument("--providers", type=str, default=None, help="comma list: anthropic,google,openai")
    bc.add_argument("--max-tokens", type=int, default=768)
    bc.add_argument("--dry-run", action="store_true", help="simulate the matrix (no spend)")
    bc.add_argument("--use-fixture", action="store_true", help="replay fixtures/catalog_matrix.json")
    bc.add_argument("--yes", action="store_true", help="skip the live-spend confirmation prompt")
    bc.add_argument("--epochs", type=int, default=3, help="passes over the task set for the curve")
    bc.add_argument("--hard", action="store_true",
                    help="use verified hard LLMRouterBench prompts (aime/gpqa/...) — real model gap")
    bc.add_argument("--hard-per-dataset", type=int, default=8, help="hard prompts sampled per dataset")
    bc.add_argument("--seed", type=int, default=7)
    bc.set_defaults(fn=cmd_bench_catalog)

    bd = sub.add_parser("bench-dataset", help="reproducible LLMRouterBench replay track")
    bd.add_argument("--max-tasks", type=int, default=96)
    bd.add_argument("--per-dataset", type=int, default=16, help="prompts sampled per dataset")
    bd.add_argument("--datasets", type=str, default=None, help="comma list of dataset ids")
    bd.add_argument("--rebuild", action="store_true", help="rebuild the matrix from the tarball")
    bd.add_argument("--epochs", type=int, default=1, help="passes over the task set for the curve")
    bd.add_argument("--seed", type=int, default=7)
    bd.set_defaults(fn=cmd_bench_dataset)

    rp = sub.add_parser("report", help="re-render report.html from a results.json")
    rp.add_argument("results")
    rp.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
