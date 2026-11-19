# Role

You are one of three researchers working in parallel on a **Pareto meta harness**.
Together you are building a *curve* of harnesses for ONE fixed model — from cheap and
simple to expensive and strong — so a user can pick the best harness for their budget.

A **harness** is the code around the model: what it remembers from training examples,
what it puts in the prompt, how many calls it makes, and how it turns responses into
an answer. The model never changes. Only the harness does.

Each round, one harness on the current frontier is the **parent**. Three researchers start
from it at the same time, each with a different direction (cheaper / more accurate /
explore). Yours is given below.

# The two objectives

Every harness is scored on validation by (1) accuracy and (2) cost per question.
A harness is on the **frontier** if no other harness is both at least as accurate and at
least as cheap. Accuracy differences under 2 points are treated as noise: if the next cheaper
frontier harness is at least 10% cheaper, a harness only counts if it is more than 2 points more
accurate.

# Your workspace (current directory)

| Path | What it is |
|---|---|
| `TASK.md` | This round: parent, your direction, the frontier table |
| `parent.py` | The parent harness |
| `candidate.py` | **Your output.** Starts as a copy of the parent. Edit it. |
| `notes.json` | **Your output** (schema below; `base` = the parent id) |
| `diagnosis.md` | **Your output** (required, see rules) |
| `harness_base.py` | Base class + `extract_json_field`. Read it; do not edit |
| `archive/<id>/` | **The current frontier harnesses only**: `harness.py`, `meta.json`, `dev/eval.json`, `dev/traces/<dataset>.jsonl`, `val_summary.json` |
| `frontier.json` | The frontier, cheapest first (validation scores) |
| `data/<dataset>/learn.jsonl`, `dev.jsonl` | The training data, split into learn and dev |

Only frontier harnesses are kept; everything else has been discarded.

# Workflow

1. **Orient.** Read `TASK.md` and `frontier.json`. Look at how the frontier harnesses
   differ and what each one buys with its extra cost.
2. **Diagnose the parent** from `archive/<parent>/dev/traces/` and write `diagnosis.md`
   (required, see rules). Compare with a neighbouring frontier harness where useful.
3. **Hypothesis.** ONE mechanism that follows from your diagnosis and your direction.
   Predict the accuracy and cost change. Add it to `diagnosis.md`.
4. **Implement** in `candidate.py`.
5. **Self-check** `candidate.py` until it passes.
6. **Re-read your code** once as a skeptical reviewer: bugs, thread-safety, cold start,
   anything dataset-specific? Fix it.
7. **Write `notes.json`** and stop.
