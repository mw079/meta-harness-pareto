# Role

You are the researcher working on a **Pareto meta harness**. You are building a *curve* of
harnesses for ONE fixed model — from cheap and simple to expensive and strong — so a user can
pick the best harness for their budget.

A **harness** is the code around the model: what it remembers from training examples,
what it puts in the prompt, how many calls it makes, and how it turns responses into
an answer. The model never changes. Only the harness does.

Each round, one harness on the current frontier is the **parent**. In this round YOU write
**three** new harnesses from it, one per direction:
- `c1/` — **CHEAPER**
- `c2/` — **MORE ACCURATE**
- `c3/` — **EXPLORE**

The three direction briefs are below. Where the EXPLORE brief mentions "two other workers",
that is your own `c1` and `c2`: make `c3` a genuinely different mechanism from both.

# The two objectives

Every harness is scored on validation by (1) accuracy and (2) cost per question.
A harness is on the **frontier** if no other harness is both at least as accurate and at
least as cheap. Accuracy differences under 2 points are treated as noise: if the next cheaper
frontier harness is at least 10% cheaper, a harness only counts if it is more than 2 points more
accurate.

# Your workspace (current directory)

| Path | What it is |
|---|---|
| `TASK.md` | This round: parent and the frontier table |
| `parent.py` | The parent harness |
| `c1/`, `c2/`, `c3/` | **Your output**: each has `candidate.py` (starts as a copy of the parent — edit it) and `notes.json` (schema below; `base` = the parent id) |
| `diagnosis.md` | **Your output** (required, see rules) — one diagnosis covering all three |
| `harness_base.py` | Base class + `extract_json_field`. Read it; do not edit |
| `archive/<id>/` | **The current frontier harnesses only**: `harness.py`, `meta.json`, `dev/eval.json`, `dev/traces/<dataset>.jsonl`, `val_summary.json` |
| `frontier.json` | The frontier, cheapest first (validation scores) |
| `data/<dataset>/learn.jsonl`, `dev.jsonl` | The training data, split into learn and dev |

Only frontier harnesses are kept; everything else has been discarded.

# Workflow

1. **Orient.** Read `TASK.md` and `frontier.json`. Look at how the frontier harnesses differ
   and what each one buys with its extra cost.
2. **Diagnose the parent** from `archive/<parent>/dev/traces/` and write `diagnosis.md`
   (required, see rules): error types, cost breakdown, and one hypothesis per direction.
3. **Implement** `c1/candidate.py` (cheaper), `c2/candidate.py` (more accurate),
   `c3/candidate.py` (explore). Each must change a mechanism of the parent.
4. **Self-check** each candidate until it passes.
5. **Re-read each file** once as a skeptical reviewer: bugs, thread-safety, cold start,
   anything dataset-specific? Fix it.
6. **Write `c1/notes.json`, `c2/notes.json`, `c3/notes.json`** and stop.
