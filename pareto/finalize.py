"""Final, one-time evaluation on held-out data. Locks each run afterwards.

    uv run python -m pareto.finalize --runs stanford pareto-1agent pareto-3agents [--yes]

Protocol (identical for both strategies): selection uses VALIDATION only. For each run we
take its validation Pareto set (exact dominance, no ε) over ALL its valid harnesses, plus the
baselines, and evaluate exactly those on
  - the TEST split of the search datasets, and
  - the TEST split of the UNSEEN datasets (each harness learns on that dataset's train split).
Results: runs/<run>/final/results.json (consumed by pareto.compare).
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

from .common import load_config, load_dotenv, read_json, write_json
from .run import Run, log, run_eval_subprocess


def selected(run: Run) -> list[dict]:
    front_ids = {m["id"] for m in run.frontier(acc_eps=0.0, cost_eps=0.0)}
    return [m for m in run.metas() if m.get("status") == "ok"
            and (m["id"] in front_ids or m.get("direction") == "baseline")]


def finalize_run(run: Run, cfg: dict) -> list[dict]:
    final = run.dir / "final"
    prev = {r["id"]: r for r in (read_json(final / "results.json") or {}).get("rows", [])}
    front_ids = {m["id"] for m in run.frontier(acc_eps=0.0, cost_eps=0.0)}
    chosen = selected(run)
    run.state["finalized"] = True  # lock before touching test data
    run.save()

    def do(m: dict) -> dict:
        if m["id"] in prev:  # idempotent: never evaluate test twice
            return prev[m["id"]]
        h = run.archive / m["id"] / "harness.py"
        st = run_eval_subprocess(h, final / m["id"] / "search_test", cfg, split="test")
        un = run_eval_subprocess(h, final / m["id"] / "unseen_test", cfg, split="test",
                                 datasets=cfg["datasets"]["unseen"])
        run.add_spend(harness_usd=st.get("real_usd_spent", 0.0) + un.get("real_usd_spent", 0.0))
        row = {
            "id": m["id"], "name": m["name"], "direction": m["direction"],
            "on_val_frontier": m["id"] in front_ids,
            "val_acc": m["accuracy"], "val_cost": m["usd_per_question"],
            "test_acc": st.get("accuracy"), "test_cost": st.get("usd_per_question"),
            "unseen_acc": un.get("accuracy"), "unseen_cost": un.get("usd_per_question"),
            "test_status": st["status"], "unseen_status": un["status"],
            "test_per_dataset": {d: v.get("accuracy") for d, v in st.get("datasets", {}).items()},
            "unseen_per_dataset": {d: v.get("accuracy") for d, v in un.get("datasets", {}).items()},
        }
        if row["test_acc"] is not None:
            row["val_to_test_gap"] = round(row["val_acc"] - row["test_acc"], 4)
        log(f"  [{run.name}] {m['id']} {m['name'][:30]}: val={row['val_acc']:.3f} "
            f"test={row['test_acc']} unseen={row['unseen_acc']}")
        return row

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(do, chosen))
    run.save()
    write_json(final / "results.json", {"run": run.name, "strategy": run.strategy, "agents": run.agents, "rows": rows,
                                        "spent": run.state["spent"], "candidates": run.state["candidates"],
                                        "iterations": run.state["iteration"]})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    a = p.parse_args()
    load_dotenv()
    cfg = load_config()
    runs = [Run(n, cfg) for n in a.runs]
    for r in runs:
        if not r.exists:
            sys.exit(f"no run named {r.name}")
    plan = {r.name: len(selected(r)) for r in runs}
    if not a.yes:
        ans = input(f"Evaluate on TEST + unseen data and lock these runs {plan}? [y/N] ")
        if ans.strip().lower() != "y":
            return
    for r in runs:
        log(f"finalizing {r.name} ({r.strategy}): {plan[r.name]} harnesses")
        finalize_run(r, cfg)
    log("done. Next: uv run python -m pareto.compare --runs " + " ".join(a.runs))


if __name__ == "__main__":
    main()
