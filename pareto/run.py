"""The search loop: two strategies on identical infrastructure.

    uv run python -m pareto.run --run stanford       --strategy stanford          --max-candidates 30
    uv run python -m pareto.run --run pareto-1agent  --strategy pareto --agents 1 --max-candidates 30
    uv run python -m pareto.run --run pareto-3agents --strategy pareto --agents 3 --max-candidates 30
    (re-running the same command resumes; --setup-only evaluates the baselines only)

pareto   (Pareto Queue): parent = next frontier member from a round-robin queue; agents write
         one candidate per direction (cheaper / accurate / explore), either as 3 parallel agents
         or as 1 agent writing all three; agents see ONLY the current frontier; the frontier is
         recomputed over all valid harnesses (noise-aware Pareto); new members join the queue.
stanford (Stanford Meta-Harness): 1 agent per round sees ALL harnesses + history and writes
         3 candidates; its goal is accuracy. Its Pareto curve is computed at the end.

Every candidate, in both strategies: depth gate -> static check -> transcript audit ->
fake-model interface check -> reviewer -> smoke eval (dev) -> dev eval (traces visible)
+ val eval (selection; traces hidden). Stops at --max-candidates or the $ budget. A round that
is cut short (usage limit, crash, Ctrl-C) is discarded as a whole; resuming starts it again.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .checks import audit_transcript, static_check
from .common import (
    BASELINES_DIR,
    ROOT,
    RUNS_DIR,
    TEMPLATE_DIR,
    append_jsonl,
    load_config,
    load_dotenv,
    read_json,
    read_jsonl,
    write_json,
)
from .data import load_splits
from .evaluate import split_learn_dev
from .frontier import ParentQueue, Point, compute_frontier, hypervolume
from .workers import (
    PARETO_SLOTS,
    AgentUnavailable,
    read_notes,
    run_pareto_agent,
    run_reviewer,
    run_stanford_agent,
)

_lock = threading.Lock()
STRATEGIES = ("pareto", "stanford")


# --------------------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_eval_subprocess(harness: Path, out: Path, cfg: dict, split: str = "val",
                        datasets: list[str] | None = None, limit_train: int | None = None,
                        limit_eval: int | None = None) -> dict:
    datasets = datasets or cfg["datasets"]["search"]
    cmd = [sys.executable, "-m", "pareto.evaluate", "--harness", str(harness), "--split", split,
           "--datasets", ",".join(datasets), "--out", str(out)]
    if limit_train is not None:
        cmd += ["--limit-train", str(limit_train)]
    if limit_eval is not None:
        cmd += ["--limit-eval", str(limit_eval)]
    out.mkdir(parents=True, exist_ok=True)
    try:
        with open(out / "stdout.log", "w") as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           timeout=cfg["evaluation"]["eval_timeout_s"])
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "evaluation timed out", "real_usd_spent": 0.0}
    summary = read_json(out / "eval.json")
    if not summary:
        tail = (out / "stdout.log").read_text()[-1500:]
        return {"status": "crashed", "error": f"no eval.json; log tail: {tail}", "real_usd_spent": 0.0}
    return summary


def interface_check_subprocess(harness: Path) -> list[str]:
    try:
        p = subprocess.run([sys.executable, "-m", "pareto.checks", str(harness)], cwd=ROOT,
                           capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return ["interface check timed out (180s) with a fake model"]
    return [] if p.returncode == 0 else [p.stdout[-2000:] + p.stderr[-1000:]]


def val_summary(ev: dict) -> dict:
    """What agents may see about the val run: aggregate numbers only (no inputs, predictions,
    targets, prompts or error messages that could quote an example)."""
    def short_error(e: str | None) -> str | None:
        return e.split(":")[0][:80] if e else None
    return {
        "status": ev.get("status"), "error_type": short_error(ev.get("error")),
        **{k: ev.get(k) for k in ("accuracy", "usd_per_question", "tokens_per_question", "calls_per_question")},
        "datasets": {
            d: {"status": v.get("status"), "n": v.get("n"), "errors": v.get("errors"),
                **{k: v.get(k) for k in ("accuracy", "usd_per_question", "tokens_per_question", "calls_per_question")}}
            for d, v in ev.get("datasets", {}).items()
        },
    }


def scores_from_eval(ev: dict) -> dict:
    return {
        "accuracy": ev.get("accuracy"),
        "usd_per_question": ev.get("usd_per_question"),
        "tokens_per_question": ev.get("tokens_per_question"),
        "calls_per_question": ev.get("calls_per_question"),
        "learn_usd": ev.get("learn_usd"),
        "per_dataset": {
            d: {"accuracy": v.get("accuracy"), "usd_per_question": v.get("usd_per_question"),
                "errors": v.get("errors"), "calls_per_question": v.get("calls_per_question")}
            for d, v in ev.get("datasets", {}).items() if v.get("status") == "ok"
        },
    }


# --------------------------------------------------------------------------- run state
class Run:
    def __init__(self, name: str, cfg: dict):
        self.cfg = cfg
        self.name = name
        self.dir = RUNS_DIR / name
        self.archive = self.dir / "archive"   # code, meta, dev traces, val summary
        self.hidden = self.dir / "hidden"     # val traces: never visible to agents
        self.state = read_json(self.dir / "state.json", default=None)

    @property
    def exists(self) -> bool:
        return self.state is not None

    @property
    def strategy(self) -> str:
        return self.state["strategy"]

    @property
    def agents(self) -> int:
        """Agents per round: 1 or 3 for pareto; stanford always uses 1."""
        return self.state.get("agents", 1)

    @property
    def label(self) -> str:
        if self.strategy == "stanford":
            return "Stanford Meta-Harness"
        return f"Pareto Queue ({self.agents} agent{'s' if self.agents > 1 else ''})"

    def save(self) -> None:
        with _lock:
            write_json(self.dir / "state.json", self.state)

    def metas(self) -> list[dict]:
        out = []
        for d in sorted(self.archive.iterdir()) if self.archive.exists() else []:
            m = read_json(d / "meta.json")
            if m:
                out.append(m)
        return out

    def new_id(self) -> str:
        with _lock:
            self.state["next_id"] += 1
            return f"h{self.state['next_id']:04d}"

    def frontier(self, acc_eps: float | None = None, cost_eps: float | None = None) -> list[dict]:
        s = self.cfg["search"]
        ae = s["acc_eps"] if acc_eps is None else acc_eps
        ce = s["cost_eps"] if cost_eps is None else cost_eps
        valid = [m for m in self.metas() if m.get("status") == "ok"]
        pts = [Point(m["id"], m["accuracy"], m["usd_per_question"], i) for i, m in enumerate(valid)]
        by_id = {m["id"]: m for m in valid}
        return [by_id[p.id] for p in compute_frontier(pts, ae, ce)]

    def write_frontier(self) -> list[dict]:
        front = self.frontier()
        keys = ("id", "name", "accuracy", "usd_per_question", "tokens_per_question",
                "calls_per_question", "per_dataset", "direction", "parent", "dev_accuracy")
        write_json(self.dir / "frontier.json", [{k: m.get(k) for k in keys} for m in front])
        return front

    def add_spend(self, harness_usd: float = 0.0, proposer_usd: float = 0.0) -> None:
        with _lock:
            self.state["spent"]["harness_usd"] += harness_usd or 0.0
            self.state["spent"]["proposer_usd"] += proposer_usd or 0.0

    def total_spent(self) -> float:
        return self.state["spent"]["harness_usd"] + self.state["spent"]["proposer_usd"]

    def evaluate(self, cid: str, harness: Path) -> tuple[dict, dict]:
        """Dev run (traces visible in archive/<id>/dev) and val run (traces hidden in
        hidden/<id>/val; only archive/<id>/val_summary.json is visible), in parallel."""
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_dev = pool.submit(run_eval_subprocess, harness, self.archive / cid / "dev", self.cfg, "dev")
            f_val = pool.submit(run_eval_subprocess, harness, self.hidden / cid / "val", self.cfg, "val")
            dev, val = f_dev.result(), f_val.result()
        self.add_spend(harness_usd=dev.get("real_usd_spent", 0.0) + val.get("real_usd_spent", 0.0))
        write_json(self.archive / cid / "val_summary.json", val_summary(val))
        return dev, val


def setup_run(run: Run, strategy: str, agents: int) -> None:
    cfg = run.cfg
    run.dir.mkdir(parents=True, exist_ok=True)
    run.archive.mkdir(exist_ok=True)
    shutil.copy(ROOT / "config.yaml", run.dir / "config.snapshot.yaml")
    # Search data visible to agents: learn + dev parts of the train split of SEARCH datasets.
    for name in cfg["datasets"]["search"]:
        train, _val, _test, _ev = load_splits(name, cfg)
        learn, dev = split_learn_dev(train, cfg["evaluation"].get("dev_fraction", 0.25))
        for split, rows in (("learn", learn), ("dev", dev)):
            p = run.dir / "data" / name / f"{split}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                for r in rows:
                    f.write(json.dumps({k: r[k] for k in ("input", "target", "raw_question") if k in r}) + "\n")
    (run.dir / "history.jsonl").touch()
    run.state = {"strategy": strategy, "agents": agents, "iteration": 0, "next_id": 0, "candidates": 0,
                 "spent": {"harness_usd": 0.0, "proposer_usd": 0.0}, "queue": {}, "finalized": False,
                 "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    run.save()

    log("evaluating baselines")

    def do_baseline(name: str) -> None:
        bid = f"b_{name}"
        d = run.archive / bid
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(BASELINES_DIR / f"{name}.py", d / "harness.py")
        dev, ev = run.evaluate(bid, d / "harness.py")
        status = ev["status"] if dev["status"] == "ok" else dev["status"]
        error = dev.get("error") or (f"val: {val_summary(ev)['error_type']}" if ev.get("error") else None)
        meta = {"id": bid, "name": name, "direction": "baseline", "parent": None, "iteration": 0,
                "status": status, "error": error, **scores_from_eval(ev),
                "dev_accuracy": dev.get("accuracy"), "dev_usd_per_question": dev.get("usd_per_question")}
        write_json(d / "meta.json", meta)
        append_jsonl(run.dir / "history.jsonl", _history_line(meta))
        log(f"  {bid}: {_fmt(meta)}")

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(do_baseline, cfg["search"]["baselines"]))
    front = run.write_frontier()
    q = ParentQueue()
    q.sync([m["id"] for m in front])
    run.state["queue"] = q.to_dict()
    run.state["setup_complete"] = True
    run.save()


def _fmt(m: dict) -> str:
    if m.get("status") != "ok":
        return f"{m.get('status')} ({(m.get('error') or '; '.join(m.get('reject_reasons') or []))[:160]})"
    return f"acc={m['accuracy']:.3f}  ${m['usd_per_question']:.6f}/q  calls/q={m.get('calls_per_question') or 0:.2f}"


def _history_line(m: dict) -> dict:
    keys = ("id", "iteration", "parent", "direction", "name", "status", "accuracy", "usd_per_question",
            "calls_per_question", "per_dataset", "dev_accuracy", "mechanism", "hypothesis", "components",
            "expected_accuracy_delta", "expected_cost_delta_pct", "actual_accuracy_delta",
            "actual_cost_delta_pct", "reject_reasons", "review", "on_frontier_when_added")
    return {k: m.get(k) for k in keys if k in m}


# --------------------------------------------------------------------------- candidate pipeline
def process_candidate(run: Run, cand_dir: Path, sandbox_root: Path, agent, direction: str, base: dict,
                      iteration: int, trace_marker: str | None, proposer_usd: float) -> dict:
    """cand_dir holds candidate.py, notes.json, parent.py (= base), diagnosis.md, harness_base.py.
    trace_marker: id whose dev traces the agent must have opened (None = any dev traces)."""
    cfg = run.cfg
    cid = run.new_id()
    with _lock:
        run.state["candidates"] += 1
    d = run.archive / cid
    d.mkdir(parents=True, exist_ok=True)
    notes = read_notes(cand_dir)
    cand = cand_dir / "candidate.py"
    if cand.exists():
        shutil.copy(cand, d / "harness.py")
    meta = {
        "id": cid, "iteration": iteration, "parent": base["id"], "direction": direction,
        "name": notes.get("name") or f"{direction}_{cid}", "worker_dir": str(cand_dir.relative_to(run.dir)),
        "proposer_usd": round(proposer_usd, 4), "proposer_seconds": agent.seconds,
        **{k: notes.get(k) for k in ("mechanism", "hypothesis", "evidence", "expected_accuracy_delta",
                                      "expected_cost_delta_pct", "risks", "components")},
    }

    def finish(status: str, **extra) -> dict:
        meta.update(status=status, **extra)
        write_json(d / "meta.json", meta)
        return meta

    # 1. did the agent produce something new?
    if not cand.exists() or filecmp.cmp(cand, cand_dir / "parent.py", shallow=False):
        if agent.exit_code != 0:
            return finish("agent_failed", error=f"exit {agent.exit_code}: {agent.error}")
        return finish("no_change", error="candidate.py missing or identical to its base")
    if not notes:
        meta["concerns"] = ["notes.json missing or invalid"]

    # 2. depth gate: a written diagnosis, and evidence the agent opened dev traces
    diag = cand_dir / "diagnosis.md"
    if diag.exists():
        shutil.copy(diag, d / "diagnosis.md")
    blobs = [json.dumps(tc.get("input", {})) for tc in agent.tool_calls]
    read_traces = any("traces" in b and (trace_marker is None or trace_marker in b) for b in blobs)
    shallow = []
    if not diag.exists() or len(diag.read_text()) < 400:
        shallow.append("diagnosis.md missing or shorter than 400 chars")
    if not read_traces:
        shallow.append("agent never opened the required dev traces")
    if shallow:
        return finish("rejected", reject_reasons=shallow)

    # 3. static rules + transcript audit + fake-model interface check (all free)
    problems = (static_check(d / "harness.py", cfg["harness_rules"])
                + audit_transcript(agent.tool_calls, cfg["isolation"]["audit_patterns"]))
    if not problems:
        problems = interface_check_subprocess(d / "harness.py")
    if problems:
        return finish("rejected", reject_reasons=problems)

    # 4. independent reviewer (same isolation as the worker)
    if cfg["proposer"].get("review", True):
        verdict, rres = run_reviewer(cand_dir, run.dir, sandbox_root, cfg)
        run.add_spend(proposer_usd=rres.cost_usd)
        meta["review"] = verdict
        meta["review_usd"] = round(rres.cost_usd, 4)
        if verdict.get("verdict") != "pass":  # anything but an explicit pass is a rejection
            return finish("rejected", reject_reasons=["reviewer: " + "; ".join(verdict.get("reasons", []))])

    # 5. smoke test with the real model on a few dev examples
    sm = cfg["evaluation"]["smoke"]
    smoke = run_eval_subprocess(d / "harness.py", d / "smoke", cfg, split="dev",
                                limit_train=sm["train"], limit_eval=sm["eval"])
    run.add_spend(harness_usd=smoke.get("real_usd_spent", 0.0))
    if smoke["status"] != "ok":
        return finish("crashed", error=f"smoke: {smoke.get('error')}")

    # 6. full evaluation: dev (visible traces) + val (selection; agents see aggregates only)
    dev, ev = run.evaluate(cid, d / "harness.py")
    if dev["status"] != "ok":
        return finish(dev["status"], error=f"dev: {dev.get('error')}")
    if ev["status"] != "ok":
        return finish(ev["status"], error=f"val: {val_summary(ev)['error_type']}")
    meta.update(dev_accuracy=dev.get("accuracy"), dev_usd_per_question=dev.get("usd_per_question"))
    scores = scores_from_eval(ev)
    deltas = {
        "actual_accuracy_delta": round(scores["accuracy"] - base["accuracy"], 4),
        "actual_cost_delta_pct": round(100 * (scores["usd_per_question"] / base["usd_per_question"] - 1), 1)
        if base.get("usd_per_question") else None,
    }
    return finish("ok", **scores, **deltas)


def _record(run: Run, results: list[dict]) -> list[dict]:
    new_front = run.write_frontier()
    front_ids = {m["id"] for m in new_front}
    for m in results:
        m["on_frontier_when_added"] = m["id"] in front_ids
        write_json(run.archive / m["id"] / "meta.json", m)
        append_jsonl(run.dir / "history.jsonl", _history_line(m))
        star = "  ★ frontier" if m["on_frontier_when_added"] else ""
        log(f"  {m['id']} [{m['direction']}] {m['name']}: {_fmt(m)}{star}")
    return new_front


# --------------------------------------------------------------------------- strategies
def _log_agent(label: str, agent) -> None:
    log(f"  agent{label}: exit={agent.exit_code} ${agent.cost_usd:.2f} {agent.seconds:.0f}s "
        f"tools={len(agent.tool_calls)}{'  USAGE LIMIT' if agent.limited else ''}")


def _stage(run: Run, cdir: Path, wdir: Path, base_id: str) -> None:
    """Give a candidate folder written by a multi-candidate agent what the pipeline and reviewer
    need: its base as parent.py, the harness contract, and the agent's shared diagnosis."""
    cdir.mkdir(exist_ok=True)
    shutil.copy(run.archive / base_id / "harness.py", cdir / "parent.py")
    shutil.copy(TEMPLATE_DIR / "harness_base.py", cdir / "harness_base.py")
    if (wdir / "diagnosis.md").exists():
        shutil.copy(wdir / "diagnosis.md", cdir / "diagnosis.md")


def iterate_pareto(run: Run) -> None:
    """Pareto Queue round: dequeue a frontier parent; 3 agents (one per direction) or 1 agent
    (all three directions) write 3 candidates; merge into the frontier; enqueue new members."""
    cfg = run.cfg
    it = run.state["iteration"]
    front = run.frontier()
    q = ParentQueue.from_dict(run.state.get("queue"))
    parent_id = q.pop([m["id"] for m in front])
    parent = next(m for m in front if m["id"] == parent_id)
    log(f"=== round {it} [pareto, {run.agents} agent(s)]: parent {parent_id} ({parent['name']}) {_fmt(parent)}")

    if run.agents == 3:
        def agent_for(direction: str):
            wdir = run.dir / "workers" / f"iter{it:03d}_{direction}"
            agent = run_pareto_agent(run.dir, wdir, it, parent, front, cfg, direction)
            run.add_spend(proposer_usd=agent.cost_usd)
            _log_agent(f" {direction}", agent)
            return direction, wdir, wdir, agent, agent.cost_usd

        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = list(pool.map(agent_for, [d for _, d in PARETO_SLOTS]))
    else:
        wdir = run.dir / "workers" / f"iter{it:03d}"
        agent = run_pareto_agent(run.dir, wdir, it, parent, front, cfg, None)
        run.add_spend(proposer_usd=agent.cost_usd)
        _log_agent("", agent)
        jobs = [(d, wdir / slot, wdir, agent, agent.cost_usd / 3) for slot, d in PARETO_SLOTS]
    if any(job[3].limited for job in jobs):
        raise AgentUnavailable("an agent hit the usage limit")
    if run.agents == 1:
        for _, cdir, wdir, _, _ in jobs:
            _stage(run, cdir, wdir, parent_id)

    def one(job) -> dict:
        direction, cdir, sandbox_root, agent, usd = job
        return process_candidate(run, cdir, sandbox_root, agent, direction, parent, it,
                                 trace_marker=parent_id, proposer_usd=usd)

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(one, jobs))
    new_front = _record(run, results)
    q.sync([m["id"] for m in new_front])
    run.state["queue"] = q.to_dict()


def iterate_stanford(run: Run) -> None:
    """Stanford round: one agent sees all harnesses + history and writes 3 candidates."""
    cfg = run.cfg
    it = run.state["iteration"]
    metas = run.metas()
    ok = [m for m in metas if m.get("status") == "ok"]
    best = max(ok, key=lambda m: m["accuracy"])
    log(f"=== round {it} [stanford]: best so far {best['id']} ({best['name']}) {_fmt(best)}")
    wdir = run.dir / "workers" / f"iter{it:03d}"
    agent = run_stanford_agent(run.dir, wdir, it, metas, read_jsonl(run.dir / "history.jsonl"), cfg)
    run.add_spend(proposer_usd=agent.cost_usd)
    _log_agent("", agent)
    if agent.limited:
        raise AgentUnavailable("the agent hit the usage limit")
    by_id = {m["id"]: m for m in ok}

    def one(k: int) -> dict:
        cdir = wdir / f"c{k}"
        base = by_id.get(str(read_notes(cdir).get("base", "")).strip(), best)
        _stage(run, cdir, wdir, base["id"])
        return process_candidate(run, cdir, wdir, agent, "stanford", base, it,
                                 trace_marker=None, proposer_usd=agent.cost_usd / 3)

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(one, (1, 2, 3)))
    _record(run, results)


ITERATE = {"pareto": iterate_pareto, "stanford": iterate_stanford}


def run_round(run: Run) -> None:
    """One round, all or nothing: if it is cut short, everything it wrote is discarded so a resume
    starts the round again and never reuses candidate ids."""
    run.state["iteration"] += 1
    try:
        ITERATE[run.strategy](run)
    except AgentUnavailable as e:
        log(f"PAUSED: {e}. Discarding round {run.state['iteration']}; re-run the same command "
            f"after the limit resets to resume.")
        rollback_to(run, run.state["iteration"] - 1)
        sys.exit(2)
    except BaseException:  # crash or Ctrl-C
        log(f"round {run.state['iteration']} interrupted; discarding it")
        rollback_to(run, run.state["iteration"] - 1)
        raise
    run.save()


def rollback_to(run: Run, n: int) -> None:
    """Discard everything from rounds > n (archive, hidden, worker dirs, history lines) and rebuild
    the counters and the queue. Used when a round was cut short by a usage limit. Money already
    spent stays in `spent` (it was really spent)."""
    removed = []
    for d in sorted(run.archive.glob("h*")):
        m = read_json(d / "meta.json")  # a dir without meta.json is an unfinished candidate
        if not m or (m.get("iteration") or 0) > n:
            shutil.rmtree(d, ignore_errors=True)
            shutil.rmtree(run.hidden / d.name, ignore_errors=True)
            removed.append(d.name)
    for d in (run.dir / "workers").glob("iter*") if (run.dir / "workers").exists() else []:
        if int(d.name[4:7]) > n:
            shutil.rmtree(d, ignore_errors=True)
    hist = [h for h in read_jsonl(run.dir / "history.jsonl") if h.get("id") not in removed]
    (run.dir / "history.jsonl").write_text("".join(json.dumps(h, default=str) + "\n" for h in hist))
    kept = [m for m in run.metas() if m.get("direction") != "baseline"]
    run.state["iteration"] = n
    run.state["candidates"] = len(kept)
    run.state["next_id"] = max((int(m["id"][1:]) for m in kept), default=0)
    q = ParentQueue()  # restart the round-robin from the current frontier (cheapest first)
    front = run.write_frontier()
    q.sync([m["id"] for m in front])
    run.state["queue"] = q.to_dict()
    run.state.setdefault("rollbacks", []).append({"to_round": n, "removed": removed,
                                                  "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    run.save()
    log(f"rolled back to round {n}: removed {len(removed)} candidates {removed}")


# --------------------------------------------------------------------------- reporting
def hv_ref_cost(run: Run) -> float | None:
    """Fixed hypervolume reference: 1.5x the most expensive baseline (comparable across rounds)."""
    if run.state.get("hv_ref_cost") is None:
        costs = [m["usd_per_question"] for m in run.metas()
                 if m.get("direction") == "baseline" and m.get("status") == "ok"]
        if costs:
            run.state["hv_ref_cost"] = 1.5 * max(costs)
            run.save()
    return run.state.get("hv_ref_cost")


def print_frontier(front: list[dict], run: Run) -> None:
    pts = [Point(m["id"], m["accuracy"], m["usd_per_question"]) for m in front]
    ref = hv_ref_cost(run)
    pts_in = [p for p in pts if ref is None or p.cost <= ref]
    log(f"frontier ({len(front)} points, hypervolume={hypervolume(pts_in, ref_cost=ref):.3e}; "
        f"{run.state['candidates']} candidates; spent ${run.total_spent():.2f} = harness "
        f"${run.state['spent']['harness_usd']:.2f} + agents ${run.state['spent']['proposer_usd']:.2f}):")
    for m in front:
        log(f"    {m['id']:<16} {m['name'][:36]:<36} {_fmt(m)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Harness search loop (Pareto Queue or Stanford Meta-Harness)")
    p.add_argument("--run", required=True, help="run name (directory under runs/)")
    p.add_argument("--strategy", choices=STRATEGIES, help="required for a new run")
    p.add_argument("--agents", type=int, choices=(1, 3),
                   help="pareto only (default 3): 3 parallel agents, one per direction, or 1 agent writing all three")
    p.add_argument("--max-candidates", type=int, help="stop when this many candidates were proposed")
    p.add_argument("--budget", type=float, help="override $ cap (harness model + agents)")
    p.add_argument("--setup-only", action="store_true", help="only create the run and evaluate baselines")
    p.add_argument("--rollback-to-round", type=int, help="discard rounds after N, then exit")
    a = p.parse_args()

    load_dotenv()
    cfg = load_config()
    budget = a.budget if a.budget is not None else cfg["search"]["budget_usd"]
    max_cands = a.max_candidates if a.max_candidates is not None else cfg["search"]["max_candidates"]
    run = Run(a.run, cfg)
    if run.exists and run.state.get("finalized"):
        sys.exit(f"run '{a.run}' is finalized (test set used); start a new run")
    if run.exists and not run.state.get("setup_complete"):
        sys.exit(f"setting up run '{a.run}' was interrupted (baselines incomplete); delete {run.dir} and start again")
    if run.exists and a.strategy and a.strategy != run.strategy:
        sys.exit(f"run '{a.run}' uses strategy '{run.strategy}', not '{a.strategy}'")
    if run.exists and a.agents is not None and a.agents != run.agents:
        sys.exit(f"run '{a.run}' uses {run.agents} agent(s) per round, not {a.agents}")
    if not run.exists:
        if not a.strategy:
            sys.exit("--strategy is required for a new run")
        if a.strategy == "stanford" and a.agents not in (None, 1):
            sys.exit("--agents applies to the pareto strategy only")
        agents = (a.agents or 3) if a.strategy == "pareto" else 1
        log(f"new run: {run.dir} (strategy={a.strategy}, agents={agents})")
        setup_run(run, a.strategy, agents)
    if a.rollback_to_round is not None:
        rollback_to(run, a.rollback_to_round)
        print_frontier(run.frontier(), run)
        return
    print_frontier(run.frontier(), run)
    if a.setup_only:
        return

    per_round = 3
    while True:
        if run.state["candidates"] + per_round > max_cands:
            log(f"candidate budget reached ({run.state['candidates']} of {max_cands}); stopping")
            break
        if run.total_spent() >= budget:
            log(f"$ budget reached (${run.total_spent():.2f} >= ${budget:.2f}); stopping")
            break
        run_round(run)
        print_frontier(run.frontier(), run)
    log(f"done. Next: uv run python -m pareto.finalize --runs {a.run} ...")


if __name__ == "__main__":
    main()
