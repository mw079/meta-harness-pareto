"""Verification of candidate harnesses, cheapest checks first.

1. static_check     - AST rules: single file, allowed imports, no file/OS/network access,
                      no dataset names, exactly one harness class, size limit.
2. audit_transcript - did the agent touch forbidden paths (test data, unseen datasets, .env)?
3. interface_check  - run the harness against a fake model (free): learn, round-trip state,
                      predict, thread-safety smoke.

Dataset names and forbidden path patterns live in config.yaml, which agents cannot read, so
this file reveals nothing about the unseen datasets.

CLI:
    python -m pareto.checks candidate.py                         # orchestrator: rules from config.yaml
    python -m pareto.checks --rules self_check_rules.json c.py   # agents' self-check (no dataset names)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import load_config, read_json

# Harness code is executed by the evaluator WITHOUT a sandbox, so these rules block the routes
# from allowed modules to files, the OS and the interpreter (e.g. `threading._os`,
# `re.enum.sys.modules`, `__builtins__["open"]`). They guard against cheating, not against a
# determined adversary; the reviewer and the transcript audit are the other layers.
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "__builtins__", "globals", "locals", "vars",
    "breakpoint", "memoryview", "setattr", "delattr",
}
FORBIDDEN_ATTRS = {
    "_llm", "__subclasses__", "__globals__", "__builtins__", "__code__", "__dict__", "__import__",
    "sys", "os", "subprocess", "builtins", "importlib", "modules", "system", "popen", "open",
}


def static_check(path: Path, rules: dict) -> list[str]:
    """rules: config["harness_rules"] (max_lines, allowed_imports, optional dataset_patterns:
    regexes; an inline (?i) makes one case-insensitive)."""
    problems: list[str] = []
    path = Path(path)
    if not path.exists():
        return [f"{path.name} does not exist"]
    src = path.read_text()
    n_lines = src.count("\n") + 1
    if n_lines > rules["max_lines"]:
        problems.append(f"too long: {n_lines} lines > {rules['max_lines']}")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    allowed = set(rules["allowed_imports"])
    modules = set()  # names bound by imports (modules or names imported from them)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules |= {(a.asname or a.name).split(".")[0] for a in node.names}
    harness_classes = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in allowed:
                    problems.append(f"import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                problems.append("relative imports not allowed (single-file harness)")
            elif node.module not in allowed:
                problems.append(f"import not allowed: from {node.module}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"forbidden builtin: {node.id} (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            problems.append(f"forbidden attribute: .{node.attr} (line {node.lineno})")
        elif (isinstance(node, ast.Attribute) and node.attr.startswith("_")
              and isinstance(node.value, ast.Name) and node.value.id in modules):
            problems.append(f"private module attribute not allowed: {node.value.id}.{node.attr} (line {node.lineno})")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
              and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in modules):
            problems.append(f"getattr on an imported module not allowed (line {node.lineno})")
        elif isinstance(node, ast.ClassDef):
            if any((isinstance(b, ast.Name) and b.id == "MemorySystem")
                   or (isinstance(b, ast.Attribute) and b.attr == "MemorySystem") for b in node.bases):
                harness_classes += 1
    if harness_classes != 1:
        problems.append(f"need exactly one class extending MemorySystem, found {harness_classes}")

    for pattern in rules.get("dataset_patterns", []):
        m = re.search(pattern, src)
        if m:
            problems.append(f"dataset-specific reference not allowed: '{m.group(0)}'")
    return sorted(set(problems))


def audit_transcript(tool_calls: list[dict], patterns: list[str]) -> list[str]:
    """tool_calls: [{"name":..., "input": {...}}] from an agent's stream-json log.
    patterns: config["isolation"]["audit_patterns"] (regexes)."""
    regexes = [re.compile(p) for p in patterns]
    hits = []
    for tc in tool_calls:
        blob = json.dumps(tc.get("input", {}))
        m = next((m for r in regexes if (m := r.search(blob))), None)
        if m:
            hits.append(f"{tc.get('name')} touched forbidden path pattern '{m.group(0)}': {blob[:200]}")
    return hits


def interface_check(path: Path) -> list[str]:
    """Exercise the harness with a fake model. Returns a list of problems."""
    from .llm import FakeLLM, UsageTracker
    from .loader import load_harness_class

    problems: list[str] = []
    try:
        cls = load_harness_class(path)
    except Exception as e:  # noqa: BLE001
        return [f"import failed: {type(e).__name__}: {e}"]

    batch = [
        {"input": f"Question {i}: classify item {i}.", "prediction": f"label_{i % 2}",
         "ground_truth": f"label_{i % 2}", "was_correct": True, "raw_question": f"item {i}", "context": ""}
        for i in range(6)
    ]
    try:
        cold = cls(FakeLLM(UsageTracker()))
        ans, meta = cold.predict("Cold start question?")
        if not isinstance(ans, str) or not isinstance(meta, dict):
            problems.append("predict must return (str, dict)")
    except Exception as e:  # noqa: BLE001
        problems.append(f"cold-start predict failed: {type(e).__name__}: {e}")

    try:
        h = cls(FakeLLM(UsageTracker()))
        for i in range(0, len(batch), 2):
            h.learn_from_batch(batch[i : i + 2])
        state = h.get_state()
        if not isinstance(state, str):
            problems.append("get_state must return str")
        h2 = cls(FakeLLM(UsageTracker()))
        h2.set_state(state)
        with ThreadPoolExecutor(max_workers=4) as ex:
            outs = list(ex.map(lambda q: h2.predict(q), [f"New question {i}?" for i in range(8)]))
        if any(not isinstance(a, str) for a, _ in outs):
            problems.append("predict returned a non-str answer after learning")
    except Exception as e:  # noqa: BLE001
        problems.append(f"learn/state/predict failed: {type(e).__name__}: {e}")
    return problems


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("harness")
    p.add_argument("--rules", help="JSON with harness rules (agents' self-check); default: config.yaml")
    a = p.parse_args()
    rules = read_json(Path(a.rules)) if a.rules else load_config()["harness_rules"]
    if rules is None:
        raise SystemExit(f"cannot read rules file {a.rules}")
    path = Path(a.harness)
    problems = static_check(path, rules)
    if not any(x.startswith("syntax error") for x in problems):
        problems += interface_check(path)
    if problems:
        print("SELF-CHECK FAILED:")
        for x in problems:
            print(f"  - {x}")
        raise SystemExit(1)
    print("SELF-CHECK PASSED (static rules + fake-model interface test)")


if __name__ == "__main__":
    main()
