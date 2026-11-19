"""Coding-agent workers (Claude Code, headless) and the reviewer.

Isolation, following hill-climbing-harness: every session gets a Claude Code settings
file with permission DENY rules (blocks Read/Edit/Grep/Glob tools) AND the bash SANDBOX
(blocks shell/python reads). Both are needed. Hidden: HIDDEN_PROJECT_FILES (secrets, LLM cache,
Stanford data, and every file that names the unseen datasets or reports results), other runs,
other agents' folders, and the run's own archive/hidden/history (agents get a COPY of exactly
what they may see). Agents also get no OPENAI_API_KEY, so they cannot call the real model.

What an agent sees:
  pareto   : copies of the CURRENT FRONTIER harnesses only (+ frontier.json)
  stanford : copies of ALL harnesses so far (+ history.jsonl)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .common import PROMPTS_DIR, ROOT, RUNS_DIR, TEMPLATE_DIR, read_json, write_json

VISIBLE_FILES = ("harness.py", "meta.json", "val_summary.json", "diagnosis.md")
VISIBLE_DIRS = ("dev",)


@dataclass
class AgentResult:
    exit_code: int
    text: str
    cost_usd: float
    seconds: float
    tool_calls: list[dict] = field(default_factory=list)
    error: str = ""
    limited: bool = False  # Claude usage/rate limit hit: the session's work is incomplete


class AgentUnavailable(RuntimeError):
    """Raised when a session hit the usage limit; the round must be discarded and resumed later."""


# --------------------------------------------------------------------------- isolation
# Project files agents must not read: secrets, the LLM cache, held-out data, and every file that
# names the unseen datasets or reports results (config, README, results, tests).
HIDDEN_PROJECT_FILES = (".env", ".cache", "third_party", "config.yaml", "README.md", "results", "tests")


def hidden_paths(run_dir: Path, worker_dir: Path) -> list[Path]:
    run_dir, worker_dir = run_dir.resolve(), worker_dir.resolve()
    hidden = [ROOT / name for name in HIDDEN_PROJECT_FILES]
    hidden += [p for p in RUNS_DIR.resolve().iterdir() if p.is_dir() and p != run_dir]
    for p in run_dir.iterdir():
        if p.name == "data":
            continue  # learn/dev only (agents get their own copy)
        if p.name == "workers":
            hidden += [w for w in p.iterdir() if w.resolve() != worker_dir and w.resolve() not in worker_dir.parents]
            continue
        hidden.append(p)
    return [p.resolve() for p in hidden if p.exists()]


def build_settings(run_dir: Path, worker_dir: Path) -> dict:
    hidden = hidden_paths(run_dir, worker_dir)
    rules = []
    for p in hidden:
        pat = f"/{p}/**" if p.is_dir() else f"/{p}"
        rules += [f"{tool}({pat})" for tool in ("Read", "Edit")]  # Read rules also cover Grep/Glob
    # The project's own code must not be edited by agents (sandbox covers bash; this covers Edit).
    for name in ("pareto", "pyproject.toml", "uv.lock", ".venv"):
        p = (ROOT / name).resolve()
        if p.exists():
            rules.append(f"Edit(/{p}/**)" if p.is_dir() else f"Edit(/{p})")
    return {
        "env": {"PYTHONDONTWRITEBYTECODE": "1"},
        "permissions": {"deny": rules},
        "sandbox": {"enabled": True, "allowUnsandboxedCommands": False, "failIfUnavailable": True,
                    "filesystem": {"denyRead": [str(p) for p in hidden]}},
    }


def run_claude(prompt: str, system_prompt: str, cwd: Path, cfg: dict, log_path: Path,
               tools: list[str], effort: str, timeout_s: int, settings: dict | None) -> AgentResult:
    cmd = [
        "claude", "-p", prompt,
        "--model", cfg["model"],
        "--effort", effort,
        "--output-format", "stream-json", "--verbose",
        "--dangerously-skip-permissions",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--tools", ",".join(tools),
        "--append-system-prompt", system_prompt,
    ]
    if settings is not None:
        cmd += ["--settings", json.dumps(settings)]
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "CLAUDECODE")}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.PIPE,
                                  stdin=subprocess.DEVNULL, text=True, timeout=timeout_s)
        exit_code, stderr = proc.returncode, proc.stderr
    except subprocess.TimeoutExpired:
        exit_code, stderr = 124, f"timed out after {timeout_s}s"
    except FileNotFoundError as e:
        return AgentResult(127, "", 0.0, 0.0, error=str(e))

    text_parts, tool_calls, cost, limited = [], [], 0.0, False
    for line in log_path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block["text"])
                elif isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls.append({"name": block.get("name"), "input": block.get("input", {})})
        elif ev.get("type") == "result":
            cost = float(ev.get("total_cost_usd") or 0.0)
            if ev.get("result"):
                text_parts.append(str(ev["result"]))
            if ev.get("api_error_status") == 429 or "limit" in str(ev.get("api_error", "")):
                limited = True
    if exit_code != 0 and any(s in (stderr or "").lower() for s in ("usage limit", "rate limit", "hit your")):
        limited = True
    return AgentResult(exit_code, "\n".join(text_parts), cost, round(time.time() - t0, 1),
                       tool_calls, error=(stderr or "")[-2000:] if exit_code else "", limited=limited)


# --------------------------------------------------------------------------- prompts
SELF_CHECK_RULES = "self_check_rules.json"


def self_check_cmd() -> str:
    # Project python directly (no uv: it would need write access to its cache inside the sandbox).
    # The rules come from a file in the agent's folder because config.yaml is hidden from agents.
    return f"PYTHONPATH={ROOT} {ROOT}/.venv/bin/python -m pareto.checks --rules {SELF_CHECK_RULES}"


def self_check_rules(cfg: dict) -> dict:
    """Harness rules for the agents' self-check: no dataset names (the orchestrator checks those)."""
    r = cfg["harness_rules"]
    return {"max_lines": r["max_lines"], "allowed_imports": r["allowed_imports"]}


def _fill(text: str, cfg: dict) -> str:
    hm, hr = cfg["harness_model"], cfg["harness_rules"]
    stdlib = ", ".join(m for m in hr["allowed_imports"] if m not in ("harness_base", "__future__", "concurrent"))
    return (text.replace("{self_check_cmd}", self_check_cmd())
            .replace("{max_lines}", str(hr["max_lines"])).replace("{allowed_imports}", stdlib)
            .replace("{model}", hm["model"]).replace("{effort}", hm["reasoning_effort"])
            .replace("{price_in}", f"{hm['price_per_million']['input']:.2f}")
            .replace("{price_out}", f"{hm['price_per_million']['output']:.2f}"))


# One agent writing all three directions puts them in these folders (and in this order).
PARETO_SLOTS = (("c1", "cheaper"), ("c2", "accurate"), ("c3", "explore"))
LONG_SESSION = "multi_candidate_timeout_s"  # config key: sessions that write 3 candidates


def system_prompt(role: str, cfg: dict, direction: str | None = None) -> str:
    """role: 'pareto' (one direction per agent), 'pareto_single' (one agent, all three
    directions) or 'stanford'."""
    parts = [(PROMPTS_DIR / f"role_{role}.md").read_text(), (PROMPTS_DIR / "rules.md").read_text()]
    if direction:
        parts.append((PROMPTS_DIR / f"direction_{direction}.md").read_text())
    if role == "pareto_single":
        for slot, d in PARETO_SLOTS:
            parts.append(f"# Brief for `{slot}/`\n\n" + (PROMPTS_DIR / f"direction_{d}.md").read_text())
    return _fill("\n\n".join(parts), cfg)


def _table(rows_in: list[dict], mark: str | None = None, sort_by_acc: bool = False) -> str:
    rows = ["| id | name | status | val accuracy | $/question | tokens/q | calls/q |",
            "|---|---|---|---|---|---|---|"]
    items = sorted(rows_in, key=lambda m: -(m.get("accuracy") or -1)) if sort_by_acc else rows_in
    for f in items:
        tag = " ← parent" if f["id"] == mark else ""
        if f.get("status", "ok") == "ok" and f.get("accuracy") is not None:
            rows.append(f"| {f['id']}{tag} | {f.get('name', '')} | ok | {f['accuracy']:.3f} | "
                        f"{f['usd_per_question']:.6f} | {f.get('tokens_per_question') or 0:.0f} | "
                        f"{f.get('calls_per_question') or 0:.2f} |")
        else:
            rows.append(f"| {f['id']} | {f.get('name', '')} | {f.get('status')} | – | – | – | – |")
    return "\n".join(rows)


def render_task_pareto(iteration: int, parent: dict, frontier: list[dict], direction: str | None) -> str:
    """direction set: one agent, one candidate. direction None: one agent, three candidates."""
    per_ds = "\n".join(f"- {d}: val accuracy {v['accuracy']:.3f}, ${v['usd_per_question']:.6f}/question"
                       for d, v in parent.get("per_dataset", {}).items())
    dev = parent.get("dev_accuracy")
    pid = parent["id"]
    if direction:
        title = f"direction: {direction.upper()}"
        copied = "copied to `parent.py` and `candidate.py`."
        deliverables = (f"1. `diagnosis.md`, 2. `candidate.py` (passes `{self_check_cmd()} candidate.py`),\n"
                        f"3. `notes.json` with `\"base\": \"{pid}\"`. Then stop.")
    else:
        title = "three directions from one parent"
        copied = "copied to `parent.py` and to\n`c1/candidate.py`, `c2/candidate.py`, `c3/candidate.py`."
        deliverables = ("1. `diagnosis.md`\n"
                        "2. `c1/candidate.py` (CHEAPER), `c2/candidate.py` (MORE ACCURATE), `c3/candidate.py` (EXPLORE);\n"
                        f"   each passes `{self_check_cmd()} c<k>/candidate.py`\n"
                        f"3. `c1/notes.json`, `c2/notes.json`, `c3/notes.json`, each with `\"base\": \"{pid}\"`. Then stop.")
    return f"""# Round {iteration} — {title}

Parent: **{pid}** (`{parent.get('name', '')}`), {copied}
Validation (aggregate only): accuracy {parent['accuracy']:.3f}, ${parent['usd_per_question']:.6f} per question.
{per_ds}
Dev accuracy: {dev if dev is None else f'{dev:.3f}'} — its traces are in `archive/{pid}/dev/traces/`.

## Current frontier (cheapest first)

{_table(frontier, mark=pid)}

## Deliverables
{deliverables}
"""


def render_task_stanford(iteration: int, metas: list[dict]) -> str:
    return f"""# Iteration {iteration}

Write 3 new harnesses: `c1/candidate.py`, `c2/candidate.py`, `c3/candidate.py`, each with a
`notes.json` next to it, plus one `diagnosis.md` in this directory.
Self-check each: `{self_check_cmd()} c1/candidate.py` (and c2, c3).

## Leaderboard (all harnesses so far, best validation accuracy first)

{_table(metas, sort_by_acc=True)}
"""


# --------------------------------------------------------------------------- workspace
def _copy_visible(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in VISIBLE_FILES:
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    for name in VISIBLE_DIRS:
        if (src / name).is_dir():
            shutil.copytree(src / name, dst / name, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("stdout.log"))


def build_view(run_dir: Path, worker_dir: Path, ids: list[str], extra_files: dict[str, object]) -> None:
    """Copy exactly what the agent may see into its directory."""
    worker_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE_DIR / "harness_base.py", worker_dir / "harness_base.py")
    for cid in ids:
        _copy_visible(run_dir / "archive" / cid, worker_dir / "archive" / cid)
    data_copy = worker_dir / "data"
    if not data_copy.exists():
        shutil.copytree(run_dir / "data", data_copy)
    for name, content in extra_files.items():
        p = worker_dir / name
        if isinstance(content, str):
            p.write_text(content)
        else:
            write_json(p, content)


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, default=str) + "\n" for r in rows)


def _start(wdir: Path, task: str, prompt: str, cfg: dict, run_dir: Path, timeout_key: str) -> AgentResult:
    (wdir / "TASK.md").write_text(task)
    pc = cfg["proposer"]
    return run_claude("Read TASK.md and complete it.\n\n" + task, prompt, wdir, pc, wdir / "agent_log.jsonl",
                      pc["tools"], pc["effort"], pc[timeout_key], build_settings(run_dir, wdir))


def run_pareto_agent(run_dir: Path, wdir: Path, iteration: int, parent: dict, frontier: list[dict],
                     cfg: dict, direction: str | None) -> AgentResult:
    """Pareto Queue agent. It sees only the current frontier.
    direction given: writes one candidate (`candidate.py`) in that direction (3-agent mode).
    direction None : writes c1 cheaper / c2 accurate / c3 explore (1-agent mode)."""
    keys = ("id", "name", "accuracy", "usd_per_question", "tokens_per_question", "calls_per_question",
            "per_dataset", "dev_accuracy")
    build_view(run_dir, wdir, [m["id"] for m in frontier],
               {"frontier.json": [{k: m.get(k) for k in keys} for m in frontier],
                SELF_CHECK_RULES: self_check_rules(cfg)})
    parent_code = run_dir / "archive" / parent["id"] / "harness.py"
    shutil.copy(parent_code, wdir / "parent.py")
    targets = [wdir] if direction else [wdir / slot for slot, _ in PARETO_SLOTS]
    for t in targets:
        t.mkdir(exist_ok=True)
        shutil.copy(parent_code, t / "candidate.py")
    task = render_task_pareto(iteration, parent, frontier, direction)
    if direction:
        return _start(wdir, task, system_prompt("pareto", cfg, direction), cfg, run_dir, "timeout_s")
    return _start(wdir, task, system_prompt("pareto_single", cfg), cfg, run_dir, LONG_SESSION)


def run_stanford_agent(run_dir: Path, wdir: Path, iteration: int, metas: list[dict], history: list[dict],
                       cfg: dict) -> AgentResult:
    """Stanford agent: sees all harnesses and the history; writes c1, c2, c3."""
    build_view(run_dir, wdir, [m["id"] for m in metas],
               {"history.jsonl": _jsonl(history), SELF_CHECK_RULES: self_check_rules(cfg)})
    for k in (1, 2, 3):
        (wdir / f"c{k}").mkdir(exist_ok=True)
    return _start(wdir, render_task_stanford(iteration, metas), system_prompt("stanford", cfg), cfg, run_dir,
                  LONG_SESSION)


def run_reviewer(cand_dir: Path, run_dir: Path, sandbox_root: Path, cfg: dict) -> tuple[dict, AgentResult]:
    """cand_dir holds candidate.py, notes.json, parent.py, harness_base.py."""
    pc = cfg["proposer"]
    res = run_claude(
        "Review candidate.py (with notes.json, parent.py, harness_base.py) and output your verdict JSON line.",
        (PROMPTS_DIR / "reviewer.md").read_text(), cand_dir, pc, cand_dir / "review_log.jsonl",
        ["Read", "Grep", "Glob"], pc.get("review_effort", "low"), pc.get("review_timeout_s", 600),
        build_settings(run_dir, sandbox_root),
    )
    if res.limited:
        raise AgentUnavailable("reviewer hit the usage limit")
    # No parseable verdict = not reviewed = not allowed to be evaluated.
    verdict = {"verdict": "fail", "reasons": ["reviewer produced no verdict"], "concerns": []}
    for line in reversed(res.text.strip().splitlines()):
        line = line.strip().strip("`")
        if line.startswith("{") and "verdict" in line:
            try:
                verdict = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    return verdict, res


def read_notes(cand_dir: Path) -> dict:
    notes = read_json(cand_dir / "notes.json", default=None)
    return notes if isinstance(notes, dict) else {}
