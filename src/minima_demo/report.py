"""Turn a run's matrix + routed records into a results dict, then render the HTML dashboard.

``assemble`` does all the metric math (so it is unit-testable and the JSON is the source of truth);
``render`` reads *only* that dict, so ``report.html`` can be regenerated from ``results.json`` alone
— no keys, no re-run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .baselines import Matrix
from .metrics import RoutedRecord, read_jsonl


# --- metric assembly --------------------------------------------------------------------------

def _policy_totals(matrix: Matrix, tasks: list[str], pick) -> dict[str, float]:
    """Aggregate accuracy/cost/tokens for a policy that maps each task_id -> model_id."""
    acc = cost = toks = 0.0
    for t in tasks:
        m = pick(t)
        c = matrix.cells[t][m]
        acc += c.accuracy
        cost += c.cost_usd
        toks += c.input_tokens + c.output_tokens
    n = len(tasks) or 1
    return {"accuracy": acc / n, "cost_total": cost, "cost_mean": cost / n, "tokens_total": int(toks)}


def assemble(track: str, matrix: Matrix, curve: list[RoutedRecord], sweep: list[RoutedRecord],
             savings: dict, *, namespace: str, curve_slider: float) -> dict[str, Any]:
    base = matrix.baselines()
    premium_id = base["premium"].model_id
    cheapest_id = base["cheapest"].model_id

    # Headline = the warm KNEE operating point: among Minima's slider settings (measured warm, in
    # the sweep), the cheapest one that still retains >=90% of premium accuracy. This is the honest
    # "best value" point — low sliders save the most, high sliders chase quality; the knee is where
    # Minima delivers near-premium quality for the least money. (The full dial is the Pareto chart.)
    prem_acc = base["premium"].accuracy
    by_slider: dict[float, list] = {}
    for r in sweep:
        by_slider.setdefault(r.slider, []).append(r)
    if by_slider:
        def _agg(rs):
            return sum(x.accuracy for x in rs) / len(rs), sum(x.cost_usd for x in rs) / len(rs)
        ranked = sorted(by_slider, key=lambda s: _agg(by_slider[s])[1])  # cheapest first
        eligible = [s for s in ranked if _agg(by_slider[s])[0] >= 0.9 * prem_acc]
        op_slider = eligible[0] if eligible else max(by_slider, key=lambda s: _agg(by_slider[s])[0])
        warm = by_slider[op_slider]
    else:  # no sweep — fall back to the final epoch of the curve
        op_slider = curve_slider
        seen, warm = set(), []
        for r in reversed(curve):
            if r.task_id not in seen:
                seen.add(r.task_id); warm.append(r)
        warm.reverse()
    minima_pick = {r.task_id: r.model_id for r in warm}
    tasks = [t for t in matrix.task_order if t in minima_pick]   # unique, canonical order

    totals = {
        "premium": _policy_totals(matrix, tasks, lambda t: premium_id),
        "cheapest": _policy_totals(matrix, tasks, lambda t: cheapest_id),
        "oracle": _policy_totals(matrix, tasks, lambda t: matrix.oracle_for(t)[0]),
        "minima": _policy_totals(matrix, tasks, lambda t: minima_pick[t]),
    }

    n = len(warm) or 1
    lat = [r.latency_ms for r in warm if r.latency_ms is not None]
    margins = [matrix.oracle_for(r.task_id)[1] - matrix.cells[r.task_id][minima_pick[r.task_id]].accuracy
               for r in warm]
    minima = {
        "accuracy": totals["minima"]["accuracy"],
        "cost_total": totals["minima"]["cost_total"],
        "latency_mean": (sum(lat) / len(lat)) if lat else None,
        "margin_mean": sum(margins) / n,
        "tokens_total": totals["minima"]["tokens_total"],
    }

    prem = totals["premium"]
    head = {
        "minima_accuracy": minima["accuracy"],
        "cost_savings_pct_vs_premium": _pct(prem["cost_total"] - minima["cost_total"], prem["cost_total"]),
        "accuracy_retention_vs_premium": _ratio(minima["accuracy"], prem["accuracy"]),
        "tokens_saved_vs_premium": prem["tokens_total"] - minima["tokens_total"],
        "tokens_saved_pct": _pct(prem["tokens_total"] - minima["tokens_total"], prem["tokens_total"]),
        "avg_latency_ms": minima["latency_mean"],
        "avg_margin_to_oracle": minima["margin_mean"],
        "accuracy_lift_vs_cheapest": minima["accuracy"] - totals["cheapest"]["accuracy"],
    }

    # Learning-curve series over the full (multi-epoch) stream. Accuracy/margin use a trailing
    # window so convergence is visible; savings is cumulative (a running total ratio).
    win = max(5, len(curve) // 8)
    cum_min_cost = cum_prem_cost = 0.0
    series: dict[str, list] = {k: [] for k in
                               ("step", "task_id", "rolling_accuracy", "rolling_savings_pct",
                                "rolling_margin", "decision_basis", "model_id")}
    for i, r in enumerate(curve, 1):
        cum_min_cost += r.cost_usd
        cum_prem_cost += matrix.cells[r.task_id][premium_id].cost_usd
        window = curve[max(0, i - win):i]
        wn = len(window) or 1
        series["step"].append(i)
        series["task_id"].append(r.task_id)
        series["rolling_accuracy"].append(sum(w.accuracy for w in window) / wn)
        series["rolling_savings_pct"].append(_pct(cum_prem_cost - cum_min_cost, cum_prem_cost))
        series["rolling_margin"].append(
            sum(matrix.oracle_for(w.task_id)[1] - w.accuracy for w in window) / wn)
        series["decision_basis"].append(r.decision_basis)
        series["model_id"].append(r.model_id)

    # Pareto: Minima operating point per slider (mean over the warm sweep pass)
    sweep_points = []
    for s in sorted({r.slider for r in sweep}):
        rs = [r for r in sweep if r.slider == s]
        k = len(rs) or 1
        sweep_points.append({"slider": s,
                             "accuracy": sum(r.accuracy for r in rs) / k,
                             "cost": sum(r.cost_usd for r in rs) / k})

    # per task type (warm operating point)
    by_type = []
    for tt in sorted({matrix.task_types[t] for t in tasks}):
        ts = [t for t in tasks if matrix.task_types[t] == tt]
        by_type.append({
            "task_type": tt, "n": len(ts),
            "minima_acc": _policy_totals(matrix, ts, lambda t: minima_pick[t])["accuracy"],
            "premium_acc": _policy_totals(matrix, ts, lambda t: premium_id)["accuracy"],
            "cheapest_acc": _policy_totals(matrix, ts, lambda t: cheapest_id)["accuracy"],
            "oracle_acc": _policy_totals(matrix, ts, lambda t: matrix.oracle_for(t)[0])["accuracy"],
        })

    # per difficulty (warm operating point) — the "wide difficulty range" view
    diff_rank = {"easy": 0, "medium": 1, "hard": 2}
    present = {matrix.difficulties.get(t, "") for t in tasks} - {""}
    by_difficulty = []
    for d in sorted(present, key=lambda x: diff_rank.get(x, 9)):
        ts = [t for t in tasks if matrix.difficulties.get(t) == d]
        by_difficulty.append({
            "difficulty": d, "n": len(ts),
            "minima_acc": _policy_totals(matrix, ts, lambda t: minima_pick[t])["accuracy"],
            "premium_acc": _policy_totals(matrix, ts, lambda t: premium_id)["accuracy"],
            "cheapest_acc": _policy_totals(matrix, ts, lambda t: cheapest_id)["accuracy"],
            "oracle_acc": _policy_totals(matrix, ts, lambda t: matrix.oracle_for(t)[0])["accuracy"],
        })

    selection: dict[str, int] = {}
    for r in warm:
        selection[r.model_id] = selection.get(r.model_id, 0) + 1

    return {
        "track": track,
        "namespace": namespace,
        "curve_slider": curve_slider,
        "operating_slider": op_slider,
        "n_tasks": len(tasks),
        "models": matrix.models,
        "model_points": [vars(p) for p in matrix.model_points()],
        "baselines": {k: vars(v) for k, v in base.items()},
        "totals": totals,
        "minima": minima,
        "minima_sweep": sweep_points,
        "headline": head,
        "curve_series": series,
        "refs": {"premium_acc": base["premium"].accuracy, "oracle_acc": base["oracle"].accuracy,
                 "cheapest_acc": base["cheapest"].accuracy, "random_acc": base["random"].accuracy},
        "by_task_type": by_type,
        "by_difficulty": by_difficulty,
        "model_selection": selection,
        "savings_service": savings,
    }


def reassemble_dir(d: str | Path) -> dict[str, Any]:
    """Recompute results.json from a run's saved artifacts — fully offline (no network/keys).

    Lets the metric logic (e.g. the premium definition) be re-applied to a past run without
    re-routing or re-calling any model.
    """
    d = Path(d)
    matrix = Matrix.from_dict(json.loads((d / "matrix.json").read_text()))
    curve = [RoutedRecord(**x) for x in read_jsonl(d / "routed_curve.jsonl")]
    sweep = [RoutedRecord(**x) for x in read_jsonl(d / "routed_sweep.jsonl")]
    savings = json.loads((d / "savings.json").read_text()) if (d / "savings.json").exists() else {}
    ns = savings.get("namespace") if isinstance(savings, dict) else None
    if not ns and (d / "results.json").exists():
        ns = json.loads((d / "results.json").read_text()).get("namespace")
    slider = curve[0].slider if curve else 5.0
    track = curve[0].track if curve else "catalog"
    return assemble(track, matrix, curve, sweep, savings, namespace=ns or d.name, curve_slider=slider)


def _pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


# --- rendering (light "paper" theme) ----------------------------------------------------------

# One palette, used everywhere. Minima is the indigo hero; the three reference policies each own a hue.
MINIMA_C = "#3b5bdb"   # indigo — Minima (routed)
PREMIUM_C = "#e8590c"  # orange — all-premium
CHEAP_C = "#1098ad"    # cyan   — cheapest
ORACLE_C = "#2f9e44"   # green  — oracle (perfect)
MODEL_C = "#adb5bd"    # gray   — individual candidate models
SAVINGS_C = "#7048e8"  # violet — cumulative savings line
MARGIN_C = "#e64980"   # pink   — margin-to-oracle area
INK = "#1a1d24"        # body text / axes
MUTED = "#6b7280"      # secondary text
GRID = "#eef1f5"       # gridlines
AXLINE = "#ced4da"     # axis lines


def render(results: dict, path: str | Path) -> Path:
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs

    panels = [("Cost–quality frontier",
               "Each gray dot is a candidate model; the indigo line is Minima sweeping its "
               "cost↔quality slider. It hugs the oracle edge at a fraction of premium cost.",
               _fig_pareto(results, go))]
    if results.get("by_difficulty"):
        panels.append((
            "Accuracy by difficulty",
            "Where routing earns its keep: cheap models clear the easy tasks, only strong models "
            "clear the hard ones — Minima tracks the best policy across the whole range.",
            _fig_difficulty(results, go)))
    panels += [
        ("Learning curve",
         "Rolling accuracy (left axis) and cumulative cost savings vs all-premium (right axis) as "
         "feedback accumulates over the task stream.",
         _fig_learning(results, go)),
        ("Margin to the most-effective model",
         "Minima's accuracy gap to the perfect per-task router — lower is better (0 = perfect).",
         _fig_margin(results, go)),
        ("Accuracy by task type",
         "Minima against the three reference policies, broken down by task type.",
         _fig_by_type(results, go)),
        ("Which model Minima routed to",
         "Distribution of routing decisions across the learning-curve pass.",
         _fig_selection(results, go)),
    ]
    blocks = [(head, cap, f.to_html(full_html=False, include_plotlyjs=False,
                                    config={"displayModeBar": False})) for head, cap, f in panels]
    html = _page(results, blocks, plotlyjs=get_plotlyjs())  # inline → opens offline
    path = Path(path)
    path.write_text(html)
    return path


def _layout(go, *, height: int = 430, **kw):
    base = dict(template="plotly_white", margin=dict(l=64, r=34, t=24, b=56),
                paper_bgcolor="white", plot_bgcolor="white", height=height,
                font=dict(family="-apple-system,Segoe UI,Roboto,Helvetica,sans-serif",
                          color=INK, size=13),
                xaxis=dict(gridcolor=GRID, linecolor=AXLINE, zerolinecolor=GRID),
                yaxis=dict(gridcolor=GRID, linecolor=AXLINE, zerolinecolor=GRID),
                legend=dict(orientation="h", y=-0.18, font=dict(size=12)))
    base.update(kw)  # caller's xaxis/yaxis/etc. override the defaults
    return base


def _fig_pareto(results, go):
    fig = go.Figure()
    mp = results["model_points"]
    fig.add_trace(go.Scatter(x=[p["cost_usd"] for p in mp], y=[p["accuracy"] for p in mp],
                             mode="markers+text", text=[p["model_id"] for p in mp],
                             textposition="top center", textfont=dict(size=9, color=MUTED),
                             marker=dict(size=9, color=MODEL_C), name="candidate models",
                             hovertemplate="%{text}<br>$%{x:.4f}/task · acc %{y:.2f}<extra></extra>"))
    b = results["baselines"]
    for key, color, sym in (("premium", PREMIUM_C, "star"), ("cheapest", CHEAP_C, "square"),
                            ("oracle", ORACLE_C, "diamond")):
        fig.add_trace(go.Scatter(x=[b[key]["cost_usd"]], y=[b[key]["accuracy"]], mode="markers",
                                 marker=dict(size=17, color=color, symbol=sym,
                                             line=dict(width=1, color="white")),
                                 name=b[key]["label"]))
    sp = results["minima_sweep"]
    fig.add_trace(go.Scatter(x=[p["cost"] for p in sp], y=[p["accuracy"] for p in sp],
                             mode="lines+markers", line=dict(color=MINIMA_C, width=3),
                             marker=dict(size=11, color=MINIMA_C),
                             text=[f"slider {p['slider']}" for p in sp],
                             name="Minima (slider sweep)"))
    fig.update_layout(**_layout(go, xaxis=dict(title="mean cost per task (USD, log scale)",
                                               type="log", gridcolor=GRID, linecolor=AXLINE),
                                yaxis=dict(title="accuracy", gridcolor=GRID, linecolor=AXLINE)))
    return fig


def _fig_difficulty(results, go):
    bd = results["by_difficulty"]
    x = [f"{d['difficulty']} (n={d['n']})" for d in bd]
    fig = go.Figure()
    for key, name, color in (("minima_acc", "Minima", MINIMA_C),
                             ("premium_acc", "all-premium", PREMIUM_C),
                             ("cheapest_acc", "cheapest", CHEAP_C),
                             ("oracle_acc", "oracle", ORACLE_C)):
        fig.add_trace(go.Bar(x=x, y=[d[key] for d in bd], name=name, marker_color=color))
    fig.update_layout(**_layout(go, barmode="group", bargap=0.28,
                                yaxis=dict(title="accuracy", range=[0, 1], gridcolor=GRID,
                                           linecolor=AXLINE)))
    return fig


def _fig_learning(results, go):
    s = results["curve_series"]
    r = results["refs"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["step"], y=s["rolling_accuracy"], mode="lines",
                             line=dict(color=MINIMA_C, width=3), name="Minima rolling accuracy"))
    for label, val, color in (("all-premium", r["premium_acc"], PREMIUM_C),
                              ("oracle", r["oracle_acc"], ORACLE_C),
                              ("cheapest", r["cheapest_acc"], CHEAP_C)):
        fig.add_hline(y=val, line=dict(color=color, dash="dash", width=1.3),
                      annotation_text=label, annotation_position="right",
                      annotation_font=dict(size=10, color=color))
    fig.add_trace(go.Scatter(x=s["step"], y=s["rolling_savings_pct"], mode="lines", yaxis="y2",
                             line=dict(color=SAVINGS_C, width=2, dash="dot"),
                             name="cumulative cost savings vs premium (%)"))
    fig.update_layout(**_layout(
        go, xaxis=dict(title="tasks seen (feedback accumulated)", gridcolor=GRID, linecolor=AXLINE),
        yaxis=dict(title="rolling accuracy", range=[0, 1], gridcolor=GRID, linecolor=AXLINE),
        yaxis2=dict(title="savings vs premium (%)", overlaying="y", side="right", range=[0, 100],
                    showgrid=False)))
    return fig


def _fig_margin(results, go):
    s = results["curve_series"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["step"], y=s["rolling_margin"], mode="lines", fill="tozeroy",
                             line=dict(color=MARGIN_C, width=2), fillcolor="rgba(230,73,128,0.12)",
                             name="margin to oracle"))
    fig.update_layout(**_layout(
        go, showlegend=False,
        xaxis=dict(title="tasks seen", gridcolor=GRID, linecolor=AXLINE),
        yaxis=dict(title="accuracy gap to oracle", gridcolor=GRID, linecolor=AXLINE)))
    return fig


def _fig_by_type(results, go):
    bt = results["by_task_type"]
    x = [f"{d['task_type']} (n={d['n']})" for d in bt]
    fig = go.Figure()
    for key, name, color in (("minima_acc", "Minima", MINIMA_C),
                             ("premium_acc", "all-premium", PREMIUM_C),
                             ("cheapest_acc", "cheapest", CHEAP_C),
                             ("oracle_acc", "oracle", ORACLE_C)):
        fig.add_trace(go.Bar(x=x, y=[d[key] for d in bt], name=name, marker_color=color))
    fig.update_layout(**_layout(go, barmode="group", bargap=0.28,
                                yaxis=dict(title="accuracy", range=[0, 1], gridcolor=GRID,
                                           linecolor=AXLINE)))
    return fig


def _fig_selection(results, go):
    sel = results["model_selection"]
    items = sorted(sel.items(), key=lambda kv: -kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items],
                           orientation="h", marker_color=MINIMA_C,
                           hovertemplate="%{y}: %{x} tasks<extra></extra>"))
    fig.update_layout(**_layout(go, height=360, showlegend=False,
                                yaxis=dict(autorange="reversed", linecolor=AXLINE),
                                xaxis=dict(title="tasks routed", gridcolor=GRID, linecolor=AXLINE)))
    return fig


def _card(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="card"><div class="v">{value}</div>'
            f'<div class="l">{label}</div><div class="s">{sub}</div></div>')


def _page(results: dict, blocks: list, plotlyjs: str = "") -> str:
    h = results["headline"]
    b = results["baselines"]
    lat = h["avg_latency_ms"]
    saved = h["cost_savings_pct_vs_premium"]
    # Lead with the metrics that are always strong; cost-saved gets context (it's ~0 when the best
    # model is itself cheap — the routing value then lives in retention + lift, not the cost axis).
    cards = "".join([
        _card("accuracy retained vs premium", f"{100*h['accuracy_retention_vs_premium']:.0f}%",
              f"Minima acc {h['minima_accuracy']:.2f} · premium {b['premium']['model_id']}"),
        _card("accuracy lift vs cheapest", f"+{h['accuracy_lift_vs_cheapest']:.2f}",
              f"cheapest = {b['cheapest']['model_id']}"),
        _card("margin to oracle", f"{h['avg_margin_to_oracle']:.3f}", "0 = the perfect per-task router"),
        _card("cost saved vs all-premium", f"{saved:.0f}%",
              "premium is itself cheap here" if saved < 1 else f"vs {b['premium']['model_id']}"),
        _card("tokens saved vs premium", f"{h['tokens_saved_pct']:.0f}%",
              f"{h['tokens_saved_vs_premium']:,} tokens"),
        _card("avg latency", f"{lat:.0f} ms" if lat else "—", "routed model, realized"),
    ])
    key = "".join(
        f'<span><i style="background:{c}"></i>{lab}</span>'
        for lab, c in (("Minima (routed)", MINIMA_C), ("all-premium", PREMIUM_C),
                       ("cheapest", CHEAP_C), ("oracle (perfect)", ORACLE_C)))
    sv = results.get("savings_service", {})
    sv_note = ""
    try:
        real = sv["summary"]["realized"]
        pct = _pct(real["savings_vs_premium_est_usd"], real["est_cost_premium_usd"])
        sv_note = (f'<p class="note">Independent cross-check — Minima\'s own '
                   f'<code>GET /v1/savings</code> (estimated-cost basis) reports '
                   f'<b>{pct:.0f}% savings vs premium</b> over {real["n_reconciled"]} reconciled '
                   f'calls. Differs slightly from the cards above (different cost basis); '
                   f'full payload in <code>savings.json</code>.</p>')
    except (KeyError, TypeError):
        pass
    charts = "".join(f'<section class="chart"><h2>{head}</h2>'
                     f'<p class="cap">{cap}</p>{blk}</section>' for head, cap, blk in blocks)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Minima benchmark — {results['track']} track</title>
<script type="text/javascript">{plotlyjs}</script>
<style>
 :root{{--ink:{INK};--muted:{MUTED};--navy:#1c3d8c;--line:#e5e7eb}}
 *{{box-sizing:border-box}}
 body{{background:#f5f6f8;color:var(--ink);margin:0;padding:32px 20px;
   font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.5}}
 .wrap{{max-width:1080px;margin:0 auto}}
 h1{{font-size:24px;font-weight:700;letter-spacing:-0.01em;margin:0 0 4px}}
 h1 .track{{color:var(--navy)}}
 .sub{{color:var(--muted);margin:0 0 18px;font-size:13px}}
 .sub code{{color:var(--navy);background:#eef1fb;padding:1px 5px;border-radius:4px;font-size:12px}}
 .key{{display:flex;flex-wrap:wrap;gap:16px;margin:0 0 22px;font-size:12px;color:var(--muted)}}
 .key span{{display:inline-flex;align-items:center;gap:6px}}
 .key i{{width:11px;height:11px;border-radius:50%;display:inline-block}}
 .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin-bottom:26px}}
 .card{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px 18px 16px;
   box-shadow:0 1px 2px rgba(16,24,40,.04)}}
 .card .v{{font-size:32px;font-weight:700;color:var(--navy);letter-spacing:-0.02em}}
 .card .l{{font-size:13px;font-weight:600;margin-top:6px}}
 .card .s{{font-size:11.5px;color:var(--muted);margin-top:3px}}
 .chart{{background:#fff;border:1px solid var(--line);border-radius:14px;margin-bottom:18px;
   padding:18px 18px 8px;box-shadow:0 1px 2px rgba(16,24,40,.04)}}
 .chart h2{{font-size:16px;font-weight:650;margin:0 0 2px}}
 .chart .cap{{color:var(--muted);font-size:12.5px;margin:0 0 6px;max-width:62em}}
 .note{{color:var(--muted);font-size:12px;margin-top:6px}}
 .note code,.note b{{color:var(--navy)}}
</style></head><body><div class="wrap">
<h1>Minima — cost-aware LLM routing · <span class="track">{results['track']}</span> track</h1>
<p class="sub">{results['n_tasks']} tasks · {len(results['models'])} candidate models · headline at operating slider {results.get('operating_slider', results['curve_slider'])} (cheapest point retaining ≥90% premium accuracy) · namespace <code>{results['namespace']}</code></p>
<div class="key">{key}</div>
<div class="cards">{cards}</div>
{charts}
{sv_note}
</div></body></html>"""
