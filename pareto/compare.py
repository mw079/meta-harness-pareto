"""One plot with the Pareto curves of several runs, plus a report table.

    uv run python -m pareto.compare --runs stanford pareto-1agent pareto-3agents --out results/comparison

Labels default to each run's strategy ("Stanford Meta-Harness", "Pareto Queue (1 agent)", ...);
override with --labels. The legend lists the baselines first, then the runs in the order given.

Panels: (1) validation — all valid harnesses (faint) + each run's val frontier;
        (2) TEST, search datasets; (3) TEST, unseen datasets — Pareto among the harnesses each run
        selected on validation (see pareto.finalize). Hypervolume uses one reference point per panel
        for all runs (ref accuracy 0, ref cost 1.5x the most expensive baseline in that panel).
"""

from __future__ import annotations

import argparse

from .common import ROOT, load_config, read_json, read_jsonl
from .frontier import Point, compute_frontier, hypervolume
from .run import Run

# Categorical palette in fixed order (first three validated for colour-vision deficiency on the light
# surface); every run also gets its own marker and line style, so identity never depends on colour alone.
STYLES = [
    {"color": "#2a78d6", "marker": "o", "linestyle": "-"},
    {"color": "#eb6834", "marker": "s", "linestyle": "--"},
    {"color": "#1baf7a", "marker": "^", "linestyle": "-."},
    {"color": "#6250d6", "marker": "D", "linestyle": ":"},
]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def pareto(points: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    pts = [Point(i, a, c, k) for k, (i, a, c) in enumerate(points) if a is not None and c is not None]
    return [(p.id, p.acc, p.cost) for p in compute_frontier(pts, 0.0, 0.0)]


def hv(front: list[tuple[str, float, float]], ref_cost: float) -> float:
    return hypervolume([Point(i, a, c) for i, a, c in front if c <= ref_cost], ref_acc=0.0, ref_cost=ref_cost)


def repeat_rate(run: Run) -> float | None:
    """Pareto runs: share of candidates whose components overlap (Jaccard >= 0.5) with an EARLIER
    harness that had already been dropped (not on the frontier when it was added)."""
    hist = [h for h in read_jsonl(run.dir / "history.jsonl") if h.get("direction") != "baseline"]
    dropped: list[set] = []
    repeats = n = 0
    for h in hist:
        comps = {str(c).lower() for c in (h.get("components") or [])}
        if comps:
            n += 1
            if any(len(comps & d) / len(comps | d) >= 0.5 for d in dropped if d):
                repeats += 1
        if h.get("status") == "ok" and not h.get("on_frontier_when_added") and comps:
            dropped.append(comps)
    return repeats / n if n else None


def search_cost(run: Run) -> dict:
    """Search cost of the KEPT rounds only (rounds discarded by usage limits or rollbacks are not
    counted), at list price (LLM cache hits counted as if paid, so runs that happened to share
    cached calls are comparable). Baselines excluded (identical in every run)."""
    agents = harness = 0.0
    for m in run.metas():
        if m.get("direction") == "baseline":
            continue
        agents += (m.get("proposer_usd") or 0.0) + (m.get("review_usd") or 0.0)
        for ev_path in (run.archive / m["id"] / "smoke" / "eval.json", run.archive / m["id"] / "dev" / "eval.json",
                        run.hidden / m["id"] / "val" / "eval.json"):
            ev = read_json(ev_path) or {}
            for ds in ev.get("datasets", {}).values():
                harness += (ds.get("learn_usage") or {}).get("usd", 0.0) + (ds.get("predict_usage") or {}).get("usd", 0.0)
    return {"agents": agents, "harness": harness}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+")
    p.add_argument("--out", default="results/comparison")
    a = p.parse_args()
    cfg = load_config()
    labels = a.labels or [Run(n, cfg).label for n in a.runs]
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    data = []
    for name, label in zip(a.runs, labels, strict=True):
        run = Run(name, cfg)
        res = read_json(run.dir / "final" / "results.json")
        if not res:
            raise SystemExit(f"{name}: no final/results.json — run pareto.finalize first")
        metas = [m for m in run.metas() if m.get("status") == "ok"]
        data.append({"name": name, "label": label, "run": run, "rows": res["rows"], "metas": metas})

    panels = [("val", "Validation (search datasets)"), ("test", "Test (search datasets)"),
              ("unseen", "Unseen datasets (test)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURFACE)
    summary = {}
    for ax, (key, title) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        if key == "val":
            base_costs = [m["usd_per_question"] for d in data for m in d["metas"] if m["direction"] == "baseline"]
        else:
            base_costs = [r[f"{key}_cost"] for d in data for r in d["rows"]
                          if r["direction"] == "baseline" and r.get(f"{key}_cost")]
        ref_cost = 1.5 * max(base_costs) if base_costs else None
        drawn_baselines = False
        for d, st in zip(data, STYLES, strict=False):  # up to len(STYLES) runs
            if key == "val":
                pts = [(m["id"], m["accuracy"], m["usd_per_question"]) for m in d["metas"]]
                ax.scatter([c for _, _, c in pts], [acc for _, acc, _ in pts], s=14, color=st["color"],
                           alpha=0.25, marker=st["marker"], linewidths=0, zorder=2)
            else:
                pts = [(r["id"], r.get(f"{key}_acc"), r.get(f"{key}_cost")) for r in d["rows"]]
            front = pareto(pts)
            xs, ys = [c for _, _, c in front], [acc for _, acc, _ in front]
            ax.step(xs, ys, where="post", color=st["color"], linestyle=st["linestyle"], linewidth=2, zorder=3)
            ax.scatter(xs, ys, s=64, color=st["color"], marker=st["marker"], edgecolors=SURFACE,
                       linewidths=2, zorder=4, label=d["label"])
            summary.setdefault(d["name"], {})[f"{key}_hypervolume"] = hv(front, ref_cost) if ref_cost else None
            summary[d["name"]][f"{key}_frontier"] = [i for i, _, _ in front]
            if not drawn_baselines:
                if key == "val":
                    bl = [(m["name"], m["accuracy"], m["usd_per_question"]) for m in d["metas"]
                          if m["direction"] == "baseline"]
                else:
                    bl = [(r["name"], r.get(f"{key}_acc"), r.get(f"{key}_cost")) for r in d["rows"]
                          if r["direction"] == "baseline" and r.get(f"{key}_acc") is not None]
                ax.scatter([c for _, _, c in bl], [acc for _, acc, _ in bl], s=56, marker="x",
                           color=INK2, linewidths=2, zorder=5, label="Baselines")
                for nm, acc, c in bl:
                    ax.annotate(nm, (c, acc), textcoords="offset points", xytext=(5, -11), fontsize=8, color=INK2)
                drawn_baselines = True
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.4f}".rstrip("0")))
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("$ per question (log scale)", color=INK2, fontsize=9)
        ax.set_ylabel("accuracy", color=INK2, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    handles, leg_labels = axes[0].get_legend_handles_labels()
    # Legend order: baselines first, then the runs in the order given on the command line.
    order = sorted(range(len(leg_labels)), key=lambda i: (leg_labels[i] != "Baselines", i))
    handles, leg_labels = [handles[i] for i in order], [leg_labels[i] for i in order]
    fig.legend(handles, leg_labels, frameon=False, fontsize=9, loc="upper right", ncol=len(handles),
               bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("Accuracy vs cost: Pareto curves per search approach (higher and further left is better)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "pareto_curves.png", dpi=160, facecolor=SURFACE)

    lines = ["# Comparison: " + " vs ".join(labels), "",
             "![Pareto curves](pareto_curves.png)", "",
             "| | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]

    def row(title, fn):
        lines.append(f"| {title} | " + " | ".join(fn(d) for d in data) + " |")

    def f(x, nd=4):
        return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    row("rounds", lambda d: str(d["run"].state["iteration"]))
    row("candidates proposed", lambda d: str(d["run"].state["candidates"]))
    row("valid candidates", lambda d: str(sum(1 for m in d["metas"] if m["direction"] != "baseline")))
    row("val frontier size", lambda d: str(len(summary[d["name"]]["val_frontier"])))
    for key, _ in panels:
        row(f"hypervolume ({key}) ×1e3", lambda d, k=key: f(1e3 * summary[d["name"]][f"{k}_hypervolume"]
                                                            if summary[d["name"]][f"{k}_hypervolume"] is not None else None, 3))
    row("best test accuracy", lambda d: f(max((r["test_acc"] for r in d["rows"] if r.get("test_acc") is not None),
                                              default=None), 3))
    row("mean val→test gap (selected)", lambda d: f(
        (lambda g: sum(g) / len(g) if g else None)([r["val_to_test_gap"] for r in d["rows"]
                                                    if r.get("val_to_test_gap") is not None and r["direction"] != "baseline"]), 3))
    costs = {d["name"]: search_cost(d["run"]) for d in data}
    row("search cost: coding agents (kept rounds)", lambda d: f"${costs[d['name']]['agents']:.2f}")
    row("search cost: harness model (kept rounds, list price)", lambda d: f"${costs[d['name']]['harness']:.2f}")
    row("actual spend incl. discarded rounds, cache, final eval",
        lambda d: f"${d['run'].state['spent']['harness_usd'] + d['run'].state['spent']['proposer_usd']:.2f}")
    row("repeat rate vs dropped ideas", lambda d: f(repeat_rate(d["run"]), 2))
    lines += ["", "Hypervolume: area dominated by the curve up to (accuracy 0, 1.5× the most expensive baseline's",
              "cost), same reference for all runs within a panel. Test/unseen curves use only the harnesses each",
              "run selected on validation (its val Pareto set + baselines). Repeat rate: share of candidates whose",
              "component tags overlap (Jaccard ≥ 0.5) with an earlier harness that was off the frontier when added.",
              ""]
    for d in data:
        lines += [f"## {d['label']} — selected harnesses", "",
                  "| id | name | val acc | test acc | gap | unseen acc | $/q (test) |", "|---|---|---|---|---|---|---|"]
        for r in sorted(d["rows"], key=lambda r: r.get("test_cost") or 0):
            lines.append(f"| {r['id']} | {r['name']} | {f(r['val_acc'], 3)} | {f(r.get('test_acc'), 3)} | "
                         f"{f(r.get('val_to_test_gap'), 3)} | {f(r.get('unseen_acc'), 3)} | {f(r.get('test_cost'), 6)} |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines))
    print(f"wrote {out / 'pareto_curves.png'} and {out / 'report.md'}")


if __name__ == "__main__":
    main()
